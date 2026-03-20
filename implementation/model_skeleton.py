"""
Multi-Objective Conversational Recommender System (MO-CRS)
PyTorch Implementation Skeleton

This file provides the main structure for implementing the MO-CRS model.
Based on the architecture described in model_architecture.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from collections import defaultdict, deque
from transformers import BertModel, BertTokenizer


# ============================================================================
# 1. DIALOGUE STATE TRACKER (DST)
# ============================================================================

class DialogueStateTracker(nn.Module):
    """Tracks conversation state and user preferences"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Utterance encoder (BERT)
        self.utterance_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Intent classifier
        self.intent_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.num_intents)
        )
        
        # Context encoding with GRU
        self.context_encoder = nn.GRU(
            input_size=768,
            hidden_size=config.hidden_dim,
            batch_first=True
        )
        
        # Preference tracker
        self.preference_history = []
        self.dialogue_history = []
        
    def encode_utterance(self, utterance):
        """Encode user utterance using BERT"""
        inputs = self.tokenizer(utterance, return_tensors='pt', 
                               padding=True, truncation=True)
        outputs = self.utterance_encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token
    
    def classify_intent(self, utterance_embedding):
        """Classify user intent"""
        intent_logits = self.intent_classifier(utterance_embedding)
        intent = torch.argmax(intent_logits, dim=-1)
        return intent
    
    def update_state(self, user_utterance, system_action):
        """Update dialogue state"""
        # Encode utterance
        utt_emb = self.encode_utterance(user_utterance)
        
        # Classify intent
        intent = self.classify_intent(utt_emb)
        
        # Update dialogue history
        self.dialogue_history.append({
            'utterance': user_utterance,
            'embedding': utt_emb,
            'intent': intent,
            'system_action': system_action
        })
        
        # Update context with GRU
        history_embs = torch.stack([h['embedding'] for h in self.dialogue_history])
        context_vector, _ = self.context_encoder(history_embs)
        
        # Build state representation
        state = {
            'context_vector': context_vector[-1, :],
            'intent': intent,
            'turn': len(self.dialogue_history),
            'dialogue_history': self.dialogue_history
        }
        
        return state


# ============================================================================
# 2. PERSONALIZATION ENGINE (PE)
# ============================================================================

class PersonalizationEngine(nn.Module):
    """Creates personalized user representations"""
    
    def __init__(self, config):
        super().__init__()
        
        # Static profile encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(config.static_profile_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Session encoder
        self.session_encoder = nn.Sequential(
            nn.Linear(config.session_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Attention over user history
        self.preference_attention = nn.MultiheadAttention(
            embed_dim=config.item_embedding_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + config.item_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.user_embedding_dim)
        )
        
    def forward(self, static_profile, session_profile, interaction_history, context):
        """Generate personalized user representation"""
        # Encode profiles
        static_emb = self.static_encoder(static_profile)
        session_emb = self.session_encoder(session_profile)
        
        # Attention over interaction history
        query = context.unsqueeze(1)  # Current context as query
        attended_prefs, attention_weights = self.preference_attention(
            query, interaction_history, interaction_history
        )
        
        # Fuse representations
        combined = torch.cat([
            static_emb,
            session_emb,
            attended_prefs.squeeze(1)
        ], dim=-1)
        
        user_representation = self.fusion(combined)
        
        return user_representation, attention_weights


# ============================================================================
# 3. MULTI-OBJECTIVE POLICY NETWORK (MOPN)
# ============================================================================

class MultiObjectivePolicyNetwork(nn.Module):
    """RL policy with multi-objective Q-networks"""
    
    def __init__(self, config):
        super().__init__()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Multi-head Q-networks (one per objective)
        self.q_accuracy = self._build_q_head(256, config.action_dim)
        self.q_diversity = self._build_q_head(256, config.action_dim)
        self.q_fairness = self._build_q_head(256, config.action_dim)
        self.q_engagement = self._build_q_head(256, config.action_dim)
        
        # Objective weights (learnable or fixed)
        self.objective_weights = nn.Parameter(
            torch.tensor([0.4, 0.25, 0.2, 0.15])  # acc, div, fair, eng
        )
        
        # Value networks for PPO
        self.value_accuracy = nn.Linear(256, 1)
        self.value_diversity = nn.Linear(256, 1)
        self.value_fairness = nn.Linear(256, 1)
        self.value_engagement = nn.Linear(256, 1)
        
    def _build_q_head(self, input_dim, output_dim):
        """Build Q-network head"""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, state):
        """Forward pass returns Q-values for each objective"""
        # Encode state
        z = self.state_encoder(state)
        
        # Compute Q-values for each objective
        q_acc = self.q_accuracy(z)
        q_div = self.q_diversity(z)
        q_fair = self.q_fairness(z)
        q_eng = self.q_engagement(z)
        
        # Scalarized Q-value (weighted sum)
        weights = F.softmax(self.objective_weights, dim=0)
        q_total = (weights[0] * q_acc + 
                  weights[1] * q_div + 
                  weights[2] * q_fair + 
                  weights[3] * q_eng)
        
        # Value estimates
        values = {
            'accuracy': self.value_accuracy(z),
            'diversity': self.value_diversity(z),
            'fairness': self.value_fairness(z),
            'engagement': self.value_engagement(z)
        }
        
        return q_total, {
            'q_acc': q_acc,
            'q_div': q_div,
            'q_fair': q_fair,
            'q_eng': q_eng
        }, values
    
    def select_action(self, state, epsilon=0.0):
        """Select action using epsilon-greedy"""
        if np.random.random() < epsilon:
            # Random action (exploration)
            action = torch.randint(0, self.q_accuracy[-1].out_features, (1,))
        else:
            # Greedy action
            q_total, _, _ = self.forward(state)
            action = torch.argmax(q_total, dim=-1)
        
        return action


# ============================================================================
# 4. DIVERSITY & FAIRNESS CONTROLLER (DFC)
# ============================================================================

class DiversityFairnessController:
    """Post-processes recommendations for diversity and fairness"""
    
    def __init__(self, config):
        self.config = config
        self.temporal_window = deque(maxlen=config.temporal_window_size)
        self.exposure_counts = defaultdict(int)
        self.group_statistics = defaultdict(lambda: {'recs': 0, 'accepts': 0})
        
    def mmr_rerank(self, candidates, k, lambda_mmr=0.7):
        """Maximal Marginal Relevance re-ranking"""
        selected = []
        remaining = candidates.copy()
        
        # Select most relevant first
        best_idx = np.argmax([c['score'] for c in remaining])
        selected.append(remaining.pop(best_idx))
        
        # Select remaining k-1 items
        for _ in range(k - 1):
            if not remaining:
                break
            
            mmr_scores = []
            for candidate in remaining:
                relevance = candidate['score']
                
                # Max similarity to selected items
                max_sim = max([
                    self.similarity(candidate, s) for s in selected
                ])
                
                # MMR score
                mmr = lambda_mmr * relevance - (1 - lambda_mmr) * max_sim
                mmr_scores.append(mmr)
            
            # Select best MMR
            best_idx = np.argmax(mmr_scores)
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def similarity(self, item1, item2):
        """Compute similarity between items"""
        # Cosine similarity of embeddings
        emb1 = torch.tensor(item1['embedding'])
        emb2 = torch.tensor(item2['embedding'])
        return F.cosine_similarity(emb1, emb2, dim=0).item()
    
    def apply_temporal_diversity(self, candidates):
        """Penalize items similar to recent recommendations"""
        adjusted = []
        
        for candidate in candidates:
            penalty = 0
            for hist_item in self.temporal_window:
                sim = self.similarity(candidate, hist_item)
                penalty += sim * 0.3  # Penalty factor
            
            new_score = candidate['score'] * (1 - penalty)
            adjusted.append({**candidate, 'score': new_score})
        
        return adjusted
    
    def enforce_fairness(self, recommendations, user_group):
        """Apply fairness constraints"""
        # Track statistics
        self.group_statistics[user_group]['recs'] += len(recommendations)
        
        # Check demographic parity
        group_rates = {
            g: stats['recs'] for g, stats in self.group_statistics.items()
        }
        
        # If this group is underrepresented, boost
        if len(group_rates) > 1:
            mean_rate = np.mean(list(group_rates.values()))
            if group_rates[user_group] < mean_rate * 0.9:
                # Apply boost to recommendations
                for rec in recommendations:
                    rec['score'] *= 1.2
        
        return recommendations
    
    def process_recommendations(self, candidates, user_group, k=5):
        """Main processing pipeline"""
        # Step 1: Temporal diversity
        candidates = self.apply_temporal_diversity(candidates)
        
        # Step 2: MMR re-ranking
        diverse_recs = self.mmr_rerank(candidates, k=k)
        
        # Step 3: Fairness enforcement
        fair_recs = self.enforce_fairness(diverse_recs, user_group)
        
        # Update temporal window
        self.temporal_window.extend(fair_recs)
        
        return fair_recs


# ============================================================================
# 5. EXPLANATION GENERATOR (EG)
# ============================================================================

class ExplanationGenerator(nn.Module):
    """Generates natural language explanations"""
    
    def __init__(self, config):
        super().__init__()
        
        # Could use template-based or neural generation
        # Here we show a simple seq2seq approach
        
        self.explanation_encoder = nn.Linear(config.item_dim + config.user_dim, 256)
        
        # Simple GRU decoder
        self.decoder = nn.GRU(
            input_size=config.vocab_dim,
            hidden_size=256,
            batch_first=True
        )
        
        self.output_projection = nn.Linear(256, config.vocab_size)
        
        # Templates for different explanation types
        self.templates = {
            'feature': "Recommended because it has {feature}: {value}",
            'comparison': "Similar to {ref_item} but with {difference}",
            'social': "Users like you rated this {rating}/5",
            'diversity': "For variety, here's a highly-rated {category}",
            'fairness': "To show diverse options, we're including this quality choice"
        }
    
    def select_explanation_type(self, context):
        """Select appropriate explanation type"""
        if context.get('recommendation_reason') == 'diversity':
            return 'diversity'
        elif context.get('recommendation_reason') == 'fairness':
            return 'fairness'
        elif context.get('turn') < 3:
            return 'feature'
        else:
            return 'social'
    
    def generate_explanation(self, item, user, context):
        """Generate explanation (template-based for simplicity)"""
        exp_type = self.select_explanation_type(context)
        template = self.templates[exp_type]
        
        # Fill template based on type
        if exp_type == 'feature':
            # Find matching feature
            matching_feature = self.find_matching_feature(item, user)
            explanation = template.format(**matching_feature)
        elif exp_type == 'diversity':
            explanation = template.format(category=item.get('category', 'option'))
        else:
            explanation = template
        
        return explanation
    
    def find_matching_feature(self, item, user):
        """Find item feature matching user preference"""
        # Simplified: return first matching feature
        for feature, value in item.get('features', {}).items():
            if feature in user.get('preferences', []):
                return {'feature': feature, 'value': value}
        
        return {'feature': 'quality', 'value': 'high rating'}


# ============================================================================
# 6. COMPLETE MO-CRS SYSTEM
# ============================================================================

class MOCRS:
    """Complete Multi-Objective Conversational Recommender System"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize components
        self.dst = DialogueStateTracker(config)
        self.personalization_engine = PersonalizationEngine(config)
        self.policy_network = MultiObjectivePolicyNetwork(config)
        self.dfc = DiversityFairnessController(config)
        self.explanation_generator = ExplanationGenerator(config)
        
        # Optimizers
        self.policy_optimizer = Adam(
            self.policy_network.parameters(), 
            lr=config.learning_rate
        )
        
    def converse(self, user, max_turns=20):
        """Main conversation loop"""
        conversation_log = []
        user_satisfied = False
        
        for turn in range(max_turns):
            # Get user utterance
            user_utterance = user.speak()
            
            # Update dialogue state
            state = self.dst.update_state(user_utterance, None)
            
            # Generate personalized user representation
            user_repr, _ = self.personalization_engine(
                user.static_profile,
                user.session_profile,
                user.interaction_history,
                state['context_vector']
            )
            
            # Select action using policy
            state_tensor = self.prepare_state_tensor(state, user_repr)
            action = self.policy_network.select_action(state_tensor)
            
            # Execute action
            if action == 'recommend':
                # Get candidate items
                candidates = self.get_candidate_items(user, state)
                
                # Apply diversity & fairness
                final_recs = self.dfc.process_recommendations(
                    candidates, 
                    user.group,
                    k=5
                )
                
                # Generate explanations
                explanations = [
                    self.explanation_generator.generate_explanation(
                        item, user, state
                    )
                    for item in final_recs
                ]
                
                # Present to user
                system_response = {
                    'type': 'recommendation',
                    'items': final_recs,
                    'explanations': explanations
                }
            else:
                # Other actions (ask question, clarify, etc.)
                system_response = self.generate_system_response(action, state)
            
            # Get user feedback
            feedback = user.respond(system_response)
            
            # Log
            conversation_log.append({
                'turn': turn,
                'user_utterance': user_utterance,
                'system_response': system_response,
                'feedback': feedback
            })
            
            # Check if satisfied
            if feedback['satisfied']:
                user_satisfied = True
                break
        
        return conversation_log, user_satisfied
    
    def prepare_state_tensor(self, state, user_repr):
        """Prepare state tensor for policy network"""
        # Concatenate all state components
        state_tensor = torch.cat([
            state['context_vector'],
            user_repr,
            torch.tensor([state['turn']], dtype=torch.float32)
        ])
        return state_tensor
    
    def get_candidate_items(self, user, state):
        """Retrieve candidate items for recommendation"""
        # This would interface with item database/retrieval system
        # Placeholder implementation
        return []
    
    def generate_system_response(self, action, state):
        """Generate system response for non-recommendation actions"""
        # Placeholder
        return {'type': 'question', 'text': 'What features are important to you?'}


# ============================================================================
# 7. TRAINING PROCEDURE
# ============================================================================

def train_mo_crs(model, env, config):
    """Training loop for MO-CRS"""
    
    for episode in range(config.num_episodes):
        # Collect trajectory
        state = env.reset()
        trajectory = []
        
        for t in range(config.max_turns):
            # Select action
            action = model.policy_network.select_action(
                state, epsilon=config.epsilon
            )
            
            # Execute
            next_state, rewards, done, info = env.step(action)
            
            # Store
            trajectory.append({
                'state': state,
                'action': action,
                'rewards': rewards,  # Multi-objective rewards
                'next_state': next_state,
                'done': done
            })
            
            state = next_state
            
            if done:
                break
        
        # Update policy (PPO or DQN)
        loss = compute_policy_loss(model.policy_network, trajectory)
        
        model.policy_optimizer.zero_grad()
        loss.backward()
        model.policy_optimizer.step()
        
        # Log
        if episode % 100 == 0:
            print(f"Episode {episode}: Loss = {loss.item():.4f}")


def compute_policy_loss(policy, trajectory):
    """Compute multi-objective policy loss"""
    # Placeholder for actual PPO/DQN loss computation
    # See algorithms/rl_training.md for details
    return torch.tensor(0.0, requires_grad=True)


# ============================================================================
# 8. CONFIGURATION
# ============================================================================

class Config:
    """Configuration class"""
    def __init__(self):
        # Model dimensions
        self.state_dim = 1024
        self.action_dim = 100
        self.user_embedding_dim = 128
        self.item_embedding_dim = 128
        self.hidden_dim = 256
        
        # Training
        self.learning_rate = 3e-4
        self.num_episodes = 10000
        self.max_turns = 20
        self.epsilon = 0.1
        
        # Diversity & Fairness
        self.temporal_window_size = 10
        self.lambda_mmr = 0.7
        
        # Other
        self.num_intents = 10
        self.vocab_size = 10000
        self.vocab_dim = 256


# ============================================================================
# 9. MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Initialize configuration
    config = Config()
    
    # Initialize model
    model = MOCRS(config)
    
    print("MO-CRS Model initialized successfully!")
    print(f"Policy Network: {sum(p.numel() for p in model.policy_network.parameters())} parameters")
