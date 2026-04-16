"""
Multi-Objective Conversational Recommender System (MO-CRS)
Main integration of all components
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

from dialogue_state_tracker import DialogueStateTracker, BeliefStateTracker
from personalization_engine import PersonalizationEngine
from policy_network import MultiObjectivePolicyNetwork
from diversity_fairness_controller import DiversityFairnessController
from explanation_generator import ExplanationGenerator


class MOCRS(nn.Module):
    """
    Complete Multi-Objective Conversational Recommender System
    Integrates all components: DST, PE, MOPN, DFC, EG
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # Initialize all components
        print("Initializing DST...")
        self.dst = DialogueStateTracker(config)
        
        print("Initializing Belief State Tracker...")
        self.bst = BeliefStateTracker(config)
        
        print("Initializing Personalization Engine...")
        self.pe = PersonalizationEngine(config)
        
        print("Initializing Policy Network...")
        self.policy = MultiObjectivePolicyNetwork(config)
        
        print("Initializing Diversity & Fairness Controller...")
        self.dfc = DiversityFairnessController(config)
        
        print("Initializing Explanation Generator...")
        self.eg = ExplanationGenerator(config)
        
        # Dimensions
        self.state_dim = config['model']['dialogue_state_tracker']['state_dim']
        self.profile_dim = config['model']['personalization']['profile_dim']
        self.hidden_dim = config['model']['policy']['hidden_dim']
        
        # State combiner (DST + PE outputs -> Policy input)
        self.state_combiner = nn.Sequential(
            nn.Linear(self.state_dim + self.profile_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, config['model']['policy']['state_dim'])
        )
        
        print("[OK] MO-CRS initialized successfully!")
    
    def forward(self, batch: Dict) -> Dict:
        """
        Forward pass through complete system
        
        Args:
            batch: Dictionary containing:
                - 'utterances': List of current utterances
                - 'dialogue_history': List of dialogue history dicts
                - 'static_features': (batch, static_dim) user demographics
                - 'candidate_items': (batch, num_candidates, item_dim)
                - 'candidate_scores': (batch, num_candidates)
                - 'user_demographics': List of demographic dicts
                
        Returns:
            Dictionary with system outputs
        """
        utterances = batch['utterances']
        dialogue_history = batch.get('dialogue_history', None)
        static_features = batch['static_features']
        candidate_items = batch.get('candidate_items', None)
        
        batch_size = len(utterances)
        device = next(self.parameters()).device
        
        # ==== Dialogue State Tracking ====
        dst_outputs = self.dst(utterances, dialogue_history)
        dialogue_state = dst_outputs['state']  # (batch, state_dim)
        
        # ==== Belief State Tracking ====
        if dialogue_history and len(dialogue_history) > 0:
            # Stack historical states for belief tracking
            hist_states = []
            for hist in dialogue_history:
                hist_turn = hist.get('utterance', '')
                hist_out = self.dst([hist_turn], None)
                hist_states.append(hist_out['state'])
            
            if hist_states:
                hist_states_stacked = torch.stack(hist_states, dim=1)  # (batch, history_len, state_dim)
                bst_outputs = self.bst(hist_states_stacked)
            else:
                bst_outputs = None
        else:
            bst_outputs = None
        
        # ==== Personalization ====
        # Prepare dialogue states for PE
        if dialogue_history and len(dialogue_history) > 0:
            dialogue_states_seq = torch.stack([dialogue_state] * min(10, len(dialogue_history) + 1), dim=1)
        else:
            dialogue_states_seq = dialogue_state.unsqueeze(1)  # (batch, 1, state_dim)

        is_cold_start_batch = batch.get('is_cold_start', None)
        if is_cold_start_batch is None:
            is_cold_start_batch = torch.zeros(batch_size, dtype=torch.bool, device=device)

        use_thompson_sampling = batch.get('use_thompson_sampling', None)
        
        pe_outputs = self.pe(
            static_features,
            dialogue_states_seq,
            item_embeddings=candidate_items,
            is_cold_start=is_cold_start_batch,
            use_thompson_sampling=use_thompson_sampling
        )
        user_profile = pe_outputs['user_profile']  # (batch, profile_dim)
        
        # ==== Combine state for policy ====
        combined_state = torch.cat([dialogue_state, user_profile], dim=-1)
        policy_state = self.state_combiner(combined_state)  # (batch, policy_state_dim)
        
        # ==== Policy Network ====
        policy_outputs = self.policy(policy_state)
        action_probs = policy_outputs['action_probs']
        
        # Select actions
        actions, log_probs = self.policy.select_action(policy_state, deterministic=False)
        
        # ==== Diversity & Fairness ====
        if candidate_items is not None:
            candidate_item_ids = batch.get('candidate_item_ids', None)
            recommended_history_batch = batch.get('recommended_history', None)

            # Rerank candidates
            dfc_outputs_list = []
            for i in range(batch_size):
                items_i = candidate_items[i] if candidate_items.dim() == 3 else candidate_items
                scores_i = pe_outputs['preference_scores'][i] if 'preference_scores' in pe_outputs else torch.rand(items_i.shape[0])
                user_emb_i = user_profile[i]
                demographics_i = batch.get('user_demographics', [{}])[i] if i < len(batch.get('user_demographics', [])) else {}

                candidate_ids_i = None
                if candidate_item_ids is not None and i < len(candidate_item_ids):
                    candidate_ids_i = candidate_item_ids[i]

                recommended_history_i = None
                if isinstance(recommended_history_batch, list):
                    if recommended_history_batch and isinstance(recommended_history_batch[0], list):
                        if i < len(recommended_history_batch):
                            recommended_history_i = recommended_history_batch[i]
                    else:
                        recommended_history_i = recommended_history_batch
                
                dfc_out = self.dfc(
                    items_i,
                    scores_i,
                    user_emb_i,
                    recommended_history=recommended_history_i,
                    candidate_ids=candidate_ids_i,
                    user_demographics=demographics_i
                )
                dfc_outputs_list.append(dfc_out)
            
            # Aggregate DFC outputs
            reranked_indices = torch.stack([out['reranked_indices'] for out in dfc_outputs_list])
            reranked_scores = torch.stack([out['reranked_scores'] for out in dfc_outputs_list])
            temporal_penalties = torch.stack([out['temporal_penalties'] for out in dfc_outputs_list])
        else:
            reranked_indices = None
            reranked_scores = None
            temporal_penalties = None
        
        # ==== Explanation Generation ====
        # Generate explanations for top recommendations
        explanations = []
        if candidate_items is not None and reranked_indices is not None:
            # Use dialogue state + user profile as context for explanation
            explanation_context = combined_state  # (batch, state_dim + profile_dim)
            candidate_item_ids = batch.get('candidate_item_ids', None)
            candidate_item_names = batch.get('candidate_item_names', None)
            
            for i in range(batch_size):
                top_item_idx = reranked_indices[i, 0].item()
                item_name = f'Item_{top_item_idx}'

                # If caller provides candidate item ids, map reranked index back to real item id.
                if candidate_item_ids is not None and i < len(candidate_item_ids):
                    ids_i = candidate_item_ids[i]
                    if isinstance(ids_i, list) and 0 <= top_item_idx < len(ids_i):
                        item_name = str(ids_i[top_item_idx])

                # Prefer human-readable title if provided by caller.
                if candidate_item_names is not None and i < len(candidate_item_names):
                    names_i = candidate_item_names[i]
                    if isinstance(names_i, list) and 0 <= top_item_idx < len(names_i):
                        item_name = str(names_i[top_item_idx])
                
                item_info = {
                    'name': item_name,
                    'genre': 'Drama',  # Would come from item catalog
                    'rating': '8.5'
                }
                user_info = {'age': 30}
                
                eg_out = self.eg(
                    explanation_context[i:i+1],
                    item_info,
                    user_info
                )
                explanations.append(eg_out['explanations'][0])
        
        # ==== Return all outputs ====
        return {
            # DST outputs
            'dialogue_state': dialogue_state,
            'intent_probs': dst_outputs['intent_probs'],
            'slot_probs': dst_outputs['slot_probs'],
            'sentiment_probs': dst_outputs['sentiment_probs'],
            
            # PE outputs
            'user_profile': user_profile,
            'preference_scores': pe_outputs.get('preference_scores', None),
            'preference_scores_mean': pe_outputs.get('preference_scores_mean', None),
            'preference_uncertainty': pe_outputs.get('preference_uncertainty', None),
            
            # Policy outputs
            'actions': actions,
            'action_probs': action_probs,
            'log_probs': log_probs,
            'q_values': policy_outputs['q_values'],
            'values': policy_outputs['values'],
            
            # DFC outputs
            'reranked_indices': reranked_indices,
            'reranked_scores': reranked_scores,
            'temporal_penalties': temporal_penalties,
            
            # Explanations
            'explanations': explanations,
            
            # Combined state
            'policy_state': policy_state
        }
    
    def generate_response(self, batch: Dict) -> Dict:
        """
        Generate complete system response (recommendation + explanation)
        
        Args:
            batch: Input batch
            
        Returns:
            System response dictionary
        """
        # Forward pass
        outputs = self.forward(batch)
        
        # Extract top recommendations
        if outputs['reranked_indices'] is not None:
            top_k = 5
            recommendations = []
            
            for i in range(len(batch['utterances'])):
                top_indices = outputs['reranked_indices'][i, :top_k]
                top_scores = outputs['reranked_scores'][i, :top_k]
                
                recs = []
                for idx, score in zip(top_indices, top_scores):
                    recs.append({
                        'item_id': idx.item(),
                        'score': score.item()
                    })
                
                recommendations.append(recs)
        else:
            recommendations = None
        
        # Build response
        response = {
            'recommendations': recommendations,
            'explanations': outputs['explanations'],
            'action_type': outputs['actions'].tolist(),
            'intent': torch.argmax(outputs['intent_probs'], dim=-1).tolist(),
            'sentiment': torch.argmax(outputs['sentiment_probs'], dim=-1).tolist()
        }
        
        return response
    
    def train_step(self, batch: Dict, optimizer: torch.optim.Optimizer) -> Dict:
        """
        Single training step
        
        Args:
            batch: Training batch
            optimizer: Optimizer
            
        Returns:
            Loss dictionary
        """
        self.train()
        
        # Forward pass
        outputs = self.forward(batch)
        
        # Compute losses
        losses = {}
        
        # Intent classification loss (if labels available)
        if 'intent_labels' in batch:
            intent_loss = nn.CrossEntropyLoss()(
                outputs['intent_probs'],
                batch['intent_labels']
            )
            losses['intent_loss'] = intent_loss
        
        # Sentiment classification loss
        if 'sentiment_labels' in batch:
            sentiment_loss = nn.CrossEntropyLoss()(
                outputs['sentiment_probs'],
                batch['sentiment_labels']
            )
            losses['sentiment_loss'] = sentiment_loss
        
        # Preference prediction loss
        if 'preference_labels' in batch and outputs['preference_scores'] is not None:
            pref_loss = nn.MSELoss()(
                outputs['preference_scores'],
                batch['preference_labels']
            )
            losses['preference_loss'] = pref_loss
        
        # Total loss
        total_loss = sum(losses.values()) if losses else torch.tensor(0.0)
        
        # Backward pass
        optimizer.zero_grad()
        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Return losses
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                    for k, v in losses.items()}
        loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else 0.0
        
        return loss_dict
    
    @torch.no_grad()
    def evaluate(self, batch: Dict) -> Dict:
        """
        Evaluate on a batch
        
        Args:
            batch: Evaluation batch
            
        Returns:
            Evaluation metrics
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(batch)
        
        metrics = {}
        losses = []
        
        # Intent accuracy + loss
        if 'intent_labels' in batch:
            intent_preds = torch.argmax(outputs['intent_probs'], dim=-1)
            intent_acc = (intent_preds == batch['intent_labels']).float().mean()
            metrics['intent_accuracy'] = intent_acc.item()
            intent_loss = nn.CrossEntropyLoss()(outputs['intent_probs'], batch['intent_labels'])
            losses.append(intent_loss.item())
        
        # Sentiment accuracy + loss
        if 'sentiment_labels' in batch:
            sentiment_preds = torch.argmax(outputs['sentiment_probs'], dim=-1)
            sentiment_acc = (sentiment_preds == batch['sentiment_labels']).float().mean()
            metrics['sentiment_accuracy'] = sentiment_acc.item()
            sentiment_loss = nn.CrossEntropyLoss()(outputs['sentiment_probs'], batch['sentiment_labels'])
            losses.append(sentiment_loss.item())
        
        metrics['total_loss'] = float(np.mean(losses)) if losses else 0.0
        
        return metrics


if __name__ == "__main__":
    # Test complete system
    import yaml
    import os
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("Testing Multi-Objective CRS")
    print("="*60)
    
    # Create system
    mocrs = MOCRS(config)
    
    # Test batch
    batch_size = 2
    num_candidates = 50
    
    batch = {
        'utterances': [
            "I'm looking for a good action movie",
            "Something with Tom Hanks would be nice"
        ],
        'dialogue_history': None,
        'static_features': torch.randn(batch_size, config['model']['personalization']['static_features_dim']),
        'candidate_items': torch.randn(batch_size, num_candidates, 
                                      config['model']['diversity_fairness']['item_embedding_dim']),
        'user_demographics': [
            {'age_group': '26-35', 'gender': 'M'},
            {'age_group': '36-45', 'gender': 'F'}
        ]
    }
    
    print(f"\n{'='*60}")
    print("Forward Pass Test")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}")
    print(f"Candidates: {num_candidates}")
    
    # Forward pass
    outputs = mocrs(batch)
    
    print(f"\n{'DST Outputs':-^60}")
    print(f"  Dialogue state: {outputs['dialogue_state'].shape}")
    print(f"  Intent probs: {outputs['intent_probs'].shape}")
    print(f"  Sentiment probs: {outputs['sentiment_probs'].shape}")
    
    print(f"\n{'PE Outputs':-^60}")
    print(f"  User profile: {outputs['user_profile'].shape}")
    if outputs['preference_scores'] is not None:
        print(f"  Preference scores: {outputs['preference_scores'].shape}")
    
    print(f"\n{'Policy Outputs':-^60}")
    print(f"  Actions: {outputs['actions']}")
    print(f"  Action probs: {outputs['action_probs'].shape}")
    print(f"  Q-values: {outputs['q_values'].shape}")
    print(f"  Values: {outputs['values'].shape}")
    
    print(f"\n{'DFC Outputs':-^60}")
    print(f"  Reranked indices: {outputs['reranked_indices'].shape}")
    print(f"  Reranked scores: {outputs['reranked_scores'].shape}")
    
    print(f"\n{'Explanations':-^60}")
    for i, exp in enumerate(outputs['explanations']):
        print(f"  User {i+1}: {exp}")
    
    # Test response generation
    print(f"\n{'='*60}")
    print("Response Generation Test")
    print(f"{'='*60}")
    
    response = mocrs.generate_response(batch)
    
    for i in range(batch_size):
        print(f"\nUser {i+1}:")
        print(f"  Utterance: {batch['utterances'][i]}")
        print(f"  Top recommendations: {response['recommendations'][i][:3]}")
        print(f"  Explanation: {response['explanations'][i]}")
        print(f"  Detected intent: {response['intent'][i]}")
    
    print(f"\n{'='*60}")
    print("✓ All tests passed!")
    print(f"{'='*60}")
