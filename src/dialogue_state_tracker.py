"""
Dialogue State Tracker (DST)
Tracks conversation state and encodes utterances
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Dict, List, Tuple
import numpy as np


class DialogueStateTracker(nn.Module):
    """
    Tracks dialogue state over conversation turns
    Uses BERT for utterance encoding and LSTMs for context modeling
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        dst_config = config['model']['dialogue_state_tracker']
        
        # BERT encoder for utterances
        self.bert = BertModel.from_pretrained(dst_config['bert_model'])
        self.bert_hidden_size = self.bert.config.hidden_size  # 768
        
        # Freeze BERT if specified
        if dst_config.get('freeze_bert', False):
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Intent classifier
        self.intent_classifier = nn.Sequential(
            nn.Linear(self.bert_hidden_size, dst_config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(dst_config['dropout']),
            nn.Linear(dst_config['hidden_dim'], dst_config['num_intents'])
        )
        
        # Slot extractor
        self.slot_extractor = nn.Sequential(
            nn.Linear(self.bert_hidden_size, dst_config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(dst_config['dropout']),
            nn.Linear(dst_config['hidden_dim'], dst_config['num_slots'])
        )
        
        # Context encoder (bidirectional LSTM)
        self.context_encoder = nn.LSTM(
            input_size=self.bert_hidden_size,
            hidden_size=dst_config['hidden_dim'],
            num_layers=dst_config['num_layers'],
            batch_first=True,
            bidirectional=True,
            dropout=dst_config['dropout'] if dst_config['num_layers'] > 1 else 0
        )
        
        # Projection to state representation
        self.state_projector = nn.Sequential(
            nn.Linear(dst_config['hidden_dim'] * 2, dst_config['state_dim']),  # *2 for bidirectional
            nn.LayerNorm(dst_config['state_dim']),
            nn.ReLU()
        )
        
        # Sentiment analyzer
        self.sentiment_analyzer = nn.Sequential(
            nn.Linear(self.bert_hidden_size, dst_config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(dst_config['dropout']),
            nn.Linear(dst_config['hidden_dim'], 3)  # negative, neutral, positive
        )
        
        self.hidden_dim = dst_config['hidden_dim']
        self.state_dim = dst_config['state_dim']
        
        # Vocabulary for tokenization
        self.tokenizer = BertTokenizer.from_pretrained(dst_config['bert_model'])
    
    def forward(self, utterances: List[str], history: List[Dict] = None) -> Dict:
        """
        Process utterances and dialogue history
        
        Args:
            utterances: List of utterances (batch)
            history: Optional dialogue history
            
        Returns:
            Dictionary with state representation, intents, slots, sentiment
        """
        batch_size = len(utterances)
        device = next(self.parameters()).device
        
        # Tokenize utterances
        encoded = self.tokenizer(
            utterances,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Encode with BERT
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS] token representation
        cls_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        
        # Classify intent
        intent_logits = self.intent_classifier(cls_embeddings)  # (batch, num_intents)
        intent_probs = torch.softmax(intent_logits, dim=-1)
        
        # Extract slots
        slot_logits = self.slot_extractor(cls_embeddings)  # (batch, num_slots)
        slot_probs = torch.sigmoid(slot_logits)  # Multi-label
        
        # Analyze sentiment
        sentiment_logits = self.sentiment_analyzer(cls_embeddings)  # (batch, 3)
        sentiment_probs = torch.softmax(sentiment_logits, dim=-1)
        
        # Encode context if history provided
        if history and len(history) > 0:
            # Encode history utterances
            history_texts = [turn.get('utterance', '') for turn in history]
            history_encoded = self.tokenizer(
                history_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            history_ids = history_encoded['input_ids'].to(device)
            history_mask = history_encoded['attention_mask'].to(device)
            
            history_outputs = self.bert(
                input_ids=history_ids,
                attention_mask=history_mask
            )
            
            history_embeddings = history_outputs.last_hidden_state[:, 0, :]  # (history_len, 768)
            
            # Combine current and history
            combined = torch.cat([history_embeddings, cls_embeddings], dim=0)  # (history_len + batch, 768)
            combined = combined.unsqueeze(0)  # (1, seq_len, 768)
            
            # Encode with LSTM
            context_encoded, (h_n, c_n) = self.context_encoder(combined)
            
            # Take last hidden state
            context_repr = context_encoded[:, -batch_size:, :]  # (1, batch, hidden*2)
            context_repr = context_repr.squeeze(0)  # (batch, hidden*2)
        else:
            # No history, use current utterance only
            combined = cls_embeddings.unsqueeze(0)  # (1, batch, 768)
            context_encoded, (h_n, c_n) = self.context_encoder(combined)
            context_repr = context_encoded.squeeze(0)  # (batch, hidden*2)
        
        # Project to state representation
        state = self.state_projector(context_repr)  # (batch, state_dim)
        
        return {
            'state': state,
            'intent_logits': intent_logits,
            'intent_probs': intent_probs,
            'slot_logits': slot_logits,
            'slot_probs': slot_probs,
            'sentiment_logits': sentiment_logits,
            'sentiment_probs': sentiment_probs,
            'utterance_embedding': cls_embeddings
        }
    
    def encode_batch(self, conversations: List[Dict]) -> torch.Tensor:
        """
        Encode a batch of full conversations
        
        Args:
            conversations: List of conversation dicts with 'turns'
            
        Returns:
            Batch of state representations (batch, state_dim)
        """
        batch_states = []
        
        for conv in conversations:
            turns = conv.get('turns', [])
            if not turns:
                # Empty conversation, return zero state
                batch_states.append(torch.zeros(self.state_dim))
                continue
            
            # Get last utterance
            last_turn = turns[-1]
            last_utterance = last_turn.get('utterance', '')
            
            # Get history
            history = turns[:-1] if len(turns) > 1 else None
            
            # Encode
            with torch.no_grad():
                output = self.forward([last_utterance], history)
                state = output['state'][0]  # First (only) in batch
                batch_states.append(state)
        
        return torch.stack(batch_states)
    
    def get_intent_name(self, intent_id: int) -> str:
        """Map intent ID to human-readable name"""
        intent_names = [
            'request_recommendation',
            'provide_preference',
            'accept',
            'reject',
            'inquire',
            'inform',
            'chitchat',
            'acknowledge',
            'thank',
            'goodbye'
        ]
        
        if 0 <= intent_id < len(intent_names):
            return intent_names[intent_id]
        return 'unknown'
    
    def get_slot_names(self, slot_ids: List[int]) -> List[str]:
        """Map slot IDs to human-readable names"""
        slot_names = [
            'genre',
            'actor',
            'director',
            'year',
            'rating',
            'mood',
            'occasion',
            'previously_watched',
            'liked',
            'disliked'
        ]
        
        return [slot_names[idx] for idx in slot_ids if 0 <= idx < len(slot_names)]


class BeliefStateTracker(nn.Module):
    """
    Maintains belief state over dialogue
    Tracks user preferences, mentioned items, conversation goals
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.state_dim = config['model']['dialogue_state_tracker']['state_dim']
        self.hidden_dim = config['model']['dialogue_state_tracker']['hidden_dim']
        
        # Belief state components
        self.preference_tracker = nn.GRU(
            input_size=self.state_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Item mention tracker
        self.item_tracker = nn.LSTM(
            input_size=self.state_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Goal tracker
        self.goal_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 5)  # 5 conversation goals
        )
    
    def forward(self, dialogue_states: torch.Tensor) -> Dict:
        """
        Track belief state from dialogue states
        
        Args:
            dialogue_states: (batch, seq_len, state_dim)
            
        Returns:
            Belief state representation
        """
        batch_size, seq_len, _ = dialogue_states.shape
        
        # Track preferences
        pref_output, pref_hidden = self.preference_tracker(dialogue_states)
        preference_state = pref_hidden[-1]  # Last layer hidden state
        
        # Track items
        item_output, (item_hidden, item_cell) = self.item_tracker(dialogue_states)
        item_state = item_hidden[-1]
        
        # Predict goal
        combined = torch.cat([preference_state, item_state], dim=-1)
        goal_logits = self.goal_predictor(combined)
        goal_probs = torch.softmax(goal_logits, dim=-1)
        
        return {
            'preference_state': preference_state,
            'item_state': item_state,
            'goal_probs': goal_probs,
            'belief_state': combined
        }


if __name__ == "__main__":
    # Test DST
    import yaml
    import os
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create DST
    print("Creating DST...")
    dst = DialogueStateTracker(config)
    
    # Test forward pass
    test_utterances = [
        "I'm looking for a good action movie",
        "Something with Tom Hanks would be great"
    ]
    
    print(f"\nTesting with utterances: {test_utterances}")
    
    output = dst(test_utterances)
    
    print(f"\nOutput shapes:")
    print(f"  State: {output['state'].shape}")
    print(f"  Intent probs: {output['intent_probs'].shape}")
    print(f"  Slot probs: {output['slot_probs'].shape}")
    print(f"  Sentiment probs: {output['sentiment_probs'].shape}")
    
    # Test intent detection
    intent_ids = torch.argmax(output['intent_probs'], dim=-1)
    print(f"\nDetected intents:")
    for i, intent_id in enumerate(intent_ids):
        intent_name = dst.get_intent_name(intent_id.item())
        print(f"  Utterance {i}: {intent_name} (confidence: {output['intent_probs'][i, intent_id]:.3f})")
    
    # Test belief state tracker
    print("\n\nTesting Belief State Tracker...")
    bst = BeliefStateTracker(config)
    
    # Create dummy dialogue sequence
    dialogue_states = torch.randn(2, 5, config['model']['dialogue_state_tracker']['state_dim'])
    belief_output = bst(dialogue_states)
    
    print(f"Belief state shape: {belief_output['belief_state'].shape}")
    print(f"Goal probabilities shape: {belief_output['goal_probs'].shape}")
    
    print("\n✓ DST tests passed!")
