"""
Personalization Engine (PE)
Learns and maintains user profiles with cold-start handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


class PersonalizationEngine(nn.Module):
    """
    Learns user representations and personalizes recommendations
    Supports both static and dynamic user profiles
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        pe_config = config['model']['personalization']
        
        self.hidden_dim = pe_config['hidden_dim']
        self.num_heads = pe_config['attention_heads']
        self.state_dim = config['model']['dialogue_state_tracker']['state_dim']
        self.profile_dim = pe_config['profile_dim']
        
        # Static profile encoder (demographics, long-term preferences)
        self.static_encoder = nn.Sequential(
            nn.Linear(pe_config['static_features_dim'], self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(pe_config['dropout']),
            nn.Linear(self.hidden_dim, self.profile_dim)
        )
        
        # Dynamic profile encoder (current conversation)
        self.dynamic_encoder = nn.LSTM(
            input_size=self.state_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=pe_config['dropout'],
            bidirectional=True
        )
        
        # Multi-head attention for user modeling
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,  # Bidirectional LSTM
            num_heads=self.num_heads,
            dropout=pe_config['dropout'],
            batch_first=True
        )
        
        # Profile fusion
        self.profile_fusion = nn.Sequential(
            nn.Linear(self.profile_dim + self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(pe_config['dropout']),
            nn.Linear(self.hidden_dim, self.profile_dim)
        )
        
        # Item preference predictor
        self.preference_predictor = nn.Sequential(
            nn.Linear(self.profile_dim + pe_config['item_embedding_dim'], self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(pe_config['dropout']),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)  # Preference score
        )
        
        # Cold-start handler (meta-learning component)
        self.cold_start_handler = ColdStartHandler(config)
        
        # Thompson sampling for exploration
        self.thompson_sampler = ThompsonSampler(config)
    
    def forward(self, static_features: torch.Tensor, 
                dialogue_states: torch.Tensor,
                item_embeddings: torch.Tensor = None,
                is_cold_start: bool = False) -> Dict:
        """
        Generate user profile and predict preferences
        
        Args:
            static_features: (batch, static_features_dim) - demographics, etc.
            dialogue_states: (batch, seq_len, state_dim) - dialogue history
            item_embeddings: (batch, num_items, item_dim) - candidate items
            is_cold_start: Whether user is cold-start
            
        Returns:
            User profile and preference predictions
        """
        batch_size = static_features.shape[0]
        
        # Encode static profile
        static_profile = self.static_encoder(static_features)  # (batch, profile_dim)
        
        # Encode dynamic profile from dialogue
        dynamic_output, (h_n, c_n) = self.dynamic_encoder(dialogue_states)  # (batch, seq_len, hidden*2)
        
        # Apply attention to focus on important turns
        attended, attention_weights = self.attention(
            dynamic_output,
            dynamic_output,
            dynamic_output
        )  # (batch, seq_len, hidden*2)
        
        # Aggregate attended outputs
        dynamic_profile = torch.mean(attended, dim=1)  # (batch, hidden*2)
        
        # Fuse static and dynamic profiles
        combined = torch.cat([static_profile, dynamic_profile], dim=-1)
        user_profile = self.profile_fusion(combined)  # (batch, profile_dim)
        
        # Handle cold-start if needed
        if is_cold_start:
            user_profile = self.cold_start_handler(user_profile, static_features)
        
        outputs = {
            'user_profile': user_profile,
            'static_profile': static_profile,
            'dynamic_profile': dynamic_profile,
            'attention_weights': attention_weights
        }
        
        # Predict preferences if items provided
        if item_embeddings is not None:
            preference_scores = self.predict_preferences(user_profile, item_embeddings)
            outputs['preference_scores'] = preference_scores
        
        return outputs
    
    def predict_preferences(self, user_profile: torch.Tensor, 
                           item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict user preferences for items
        
        Args:
            user_profile: (batch, profile_dim)
            item_embeddings: (batch, num_items, item_dim) or (num_items, item_dim)
            
        Returns:
            Preference scores (batch, num_items)
        """
        if item_embeddings.dim() == 2:
            # Expand to batch
            item_embeddings = item_embeddings.unsqueeze(0).expand(
                user_profile.shape[0], -1, -1
            )
        
        batch_size, num_items, item_dim = item_embeddings.shape
        
        # Expand user profile for each item
        user_expanded = user_profile.unsqueeze(1).expand(-1, num_items, -1)  # (batch, num_items, profile_dim)
        
        # Concatenate user and item
        combined = torch.cat([user_expanded, item_embeddings], dim=-1)  # (batch, num_items, profile_dim + item_dim)
        
        # Predict preference
        combined_flat = combined.view(-1, combined.shape[-1])  # (batch*num_items, ...)
        scores_flat = self.preference_predictor(combined_flat)  # (batch*num_items, 1)
        scores = scores_flat.view(batch_size, num_items)  # (batch, num_items)
        
        return scores
    
    def update_profile(self, user_profile: torch.Tensor, 
                      feedback: Dict) -> torch.Tensor:
        """
        Update user profile based on feedback
        
        Args:
            user_profile: (batch, profile_dim)
            feedback: Dictionary with feedback information
            
        Returns:
            Updated user profile
        """
        # Simple update: weighted average with feedback signal
        feedback_signal = feedback.get('signal', torch.zeros_like(user_profile))
        update_rate = feedback.get('learning_rate', 0.1)
        
        updated_profile = (1 - update_rate) * user_profile + update_rate * feedback_signal
        
        return updated_profile


class ColdStartHandler(nn.Module):
    """
    Handles cold-start users with meta-learning
    Based on MAML (Model-Agnostic Meta-Learning)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        pe_config = config['model']['personalization']
        self.profile_dim = pe_config['profile_dim']
        self.hidden_dim = pe_config['hidden_dim']
        
        # Meta-learner network
        self.meta_learner = nn.Sequential(
            nn.Linear(self.profile_dim + pe_config['static_features_dim'], 
                     self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.profile_dim)
        )
        
        # Prototype bank for similar users
        self.num_prototypes = 20
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.profile_dim))
        
    def forward(self, initial_profile: torch.Tensor, 
                static_features: torch.Tensor) -> torch.Tensor:
        """
        Enhance cold-start profile
        
        Args:
            initial_profile: (batch, profile_dim)
            static_features: (batch, static_features_dim)
            
        Returns:
            Enhanced profile (batch, profile_dim)
        """
        # Concatenate profile and features
        combined = torch.cat([initial_profile, static_features], dim=-1)
        
        # Meta-learning enhancement
        meta_enhanced = self.meta_learner(combined)
        
        # Prototype matching
        similarities = F.cosine_similarity(
            initial_profile.unsqueeze(1),  # (batch, 1, profile_dim)
            self.prototypes.unsqueeze(0),   # (1, num_prototypes, profile_dim)
            dim=-1
        )  # (batch, num_prototypes)
        
        # Weighted combination with prototypes
        weights = F.softmax(similarities, dim=-1)  # (batch, num_prototypes)
        prototype_contribution = torch.matmul(weights, self.prototypes)  # (batch, profile_dim)
        
        # Combine meta-learning and prototype
        enhanced_profile = 0.7 * meta_enhanced + 0.3 * prototype_contribution
        
        return enhanced_profile


class ThompsonSampler(nn.Module):
    """
    Thompson sampling for exploration-exploitation
    Maintains uncertainty estimates for recommendations
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        pe_config = config['model']['personalization']
        
        self.profile_dim = pe_config['profile_dim']
        self.hidden_dim = pe_config['hidden_dim']
        
        # Uncertainty estimator
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.profile_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
    def sample(self, user_profile: torch.Tensor, 
              preference_scores: torch.Tensor,
              num_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample preferences with Thompson sampling
        
        Args:
            user_profile: (batch, profile_dim)
            preference_scores: (batch, num_items) - mean preferences
            num_samples: Number of samples to draw
            
        Returns:
            Sampled preferences and uncertainties
        """
        batch_size, num_items = preference_scores.shape
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_estimator(user_profile)  # (batch, 1)
        uncertainty = uncertainty.expand(-1, num_items)  # (batch, num_items)
        
        # Sample from Gaussian
        samples = []
        for _ in range(num_samples):
            noise = torch.randn_like(preference_scores) * uncertainty
            sampled = preference_scores + noise
            samples.append(sampled)
        
        if num_samples == 1:
            return samples[0], uncertainty
        else:
            return torch.stack(samples), uncertainty


class UserProfileMemory(nn.Module):
    """
    Memory module for storing and retrieving user profiles
    Implements attention-based memory access
    """
    
    def __init__(self, config: Dict, memory_size: int = 1000):
        super().__init__()
        
        pe_config = config['model']['personalization']
        self.profile_dim = pe_config['profile_dim']
        self.memory_size = memory_size
        
        # Memory bank
        self.memory = nn.Parameter(torch.randn(memory_size, self.profile_dim))
        
        # Memory attention
        self.query_proj = nn.Linear(self.profile_dim, self.profile_dim)
        self.key_proj = nn.Linear(self.profile_dim, self.profile_dim)
        self.value_proj = nn.Linear(self.profile_dim, self.profile_dim)
        
    def retrieve(self, query_profile: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        """
        Retrieve similar profiles from memory
        
        Args:
            query_profile: (batch, profile_dim)
            top_k: Number of profiles to retrieve
            
        Returns:
            Retrieved profiles (batch, top_k, profile_dim)
        """
        batch_size = query_profile.shape[0]
        
        # Project
        query = self.query_proj(query_profile)  # (batch, profile_dim)
        keys = self.key_proj(self.memory)  # (memory_size, profile_dim)
        values = self.value_proj(self.memory)  # (memory_size, profile_dim)
        
        # Compute attention scores
        scores = torch.matmul(query, keys.t())  # (batch, memory_size)
        scores = scores / (self.profile_dim ** 0.5)
        
        # Get top-k
        top_scores, top_indices = torch.topk(scores, top_k, dim=-1)  # (batch, top_k)
        
        # Retrieve values
        retrieved = values[top_indices]  # (batch, top_k, profile_dim)
        
        # Apply attention weights
        weights = F.softmax(top_scores, dim=-1).unsqueeze(-1)  # (batch, top_k, 1)
        weighted = (retrieved * weights).sum(dim=1)  # (batch, profile_dim)
        
        return weighted


if __name__ == "__main__":
    # Test Personalization Engine
    import yaml
    import os
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create PE
    print("Creating Personalization Engine...")
    pe = PersonalizationEngine(config)
    
    # Test data
    batch_size = 4
    seq_len = 10
    num_items = 50
    
    static_features = torch.randn(batch_size, config['model']['personalization']['static_features_dim'])
    dialogue_states = torch.randn(batch_size, seq_len, config['model']['dialogue_state_tracker']['state_dim'])
    item_embeddings = torch.randn(num_items, config['model']['personalization']['item_embedding_dim'])
    
    print(f"\nTesting forward pass...")
    print(f"  Static features: {static_features.shape}")
    print(f"  Dialogue states: {dialogue_states.shape}")
    print(f"  Item embeddings: {item_embeddings.shape}")
    
    # Forward pass
    outputs = pe(static_features, dialogue_states, item_embeddings)
    
    print(f"\nOutput shapes:")
    print(f"  User profile: {outputs['user_profile'].shape}")
    print(f"  Static profile: {outputs['static_profile'].shape}")
    print(f"  Dynamic profile: {outputs['dynamic_profile'].shape}")
    print(f"  Preference scores: {outputs['preference_scores'].shape}")
    print(f"  Attention weights: {outputs['attention_weights'].shape}")
    
    # Test cold-start
    print(f"\nTesting cold-start handling...")
    outputs_cold = pe(static_features, dialogue_states, item_embeddings, is_cold_start=True)
    print(f"  Cold-start profile: {outputs_cold['user_profile'].shape}")
    
    # Test Thompson sampling
    print(f"\nTesting Thompson sampling...")
    sampled_prefs, uncertainties = pe.thompson_sampler.sample(
        outputs['user_profile'],
        outputs['preference_scores'],
        num_samples=3
    )
    print(f"  Sampled preferences: {sampled_prefs.shape}")
    print(f"  Uncertainties: {uncertainties.shape}")
    
    # Test memory
    print(f"\nTesting profile memory...")
    memory = UserProfileMemory(config)
    retrieved = memory.retrieve(outputs['user_profile'], top_k=5)
    print(f"  Retrieved profiles: {retrieved.shape}")
    
    print("\n✓ Personalization Engine tests passed!")
