"""
Multi-Objective Policy Network (MOPN)
PPO-based policy with multiple objective Q-heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


class MultiObjectivePolicyNetwork(nn.Module):
    """
    Policy network with multiple objectives:
    - Accuracy (recommendation quality)
    - Diversity (avoid redundancy)
    - Fairness (equitable treatment)
    - Engagement (user satisfaction)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        policy_config = config['model']['policy']
        
        self.state_dim = policy_config['state_dim']
        self.action_dim = policy_config['action_dim']
        self.hidden_dim = policy_config['hidden_dim']
        self.num_objectives = policy_config['num_objectives']
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(policy_config['dropout']),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Multiple Q-heads for each objective
        self.q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, self.action_dim)
            )
            for _ in range(self.num_objectives)
        ])
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(policy_config['dropout']),
            nn.Linear(self.hidden_dim // 2, self.action_dim)
        )
        
        # Value head (critic) - one for each objective
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 1)
            )
            for _ in range(self.num_objectives)
        ])
        
        # Objective weights predictor (dynamic weighting)
        self.weight_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.num_objectives),
            nn.Softmax(dim=-1)
        )
        
        # Scalarization layer for multi-objective combination
        self.scalarization = ScalarizationLayer(self.num_objectives)
        
    def forward(self, state: torch.Tensor, 
                objective_weights: torch.Tensor = None) -> Dict:
        """
        Forward pass through policy network
        
        Args:
            state: (batch, state_dim)
            objective_weights: (batch, num_objectives) or None for automatic
            
        Returns:
            Dictionary with actions, values, Q-values
        """
        batch_size = state.shape[0]
        
        # Extract features
        features = self.feature_extractor(state)  # (batch, hidden_dim)
        
        # Compute Q-values for each objective
        q_values = []
        for q_head in self.q_heads:
            q = q_head(features)  # (batch, action_dim)
            q_values.append(q)
        q_values = torch.stack(q_values, dim=1)  # (batch, num_objectives, action_dim)
        
        # Compute value for each objective
        values = []
        for value_head in self.value_heads:
            v = value_head(features)  # (batch, 1)
            values.append(v)
        values = torch.stack(values, dim=1).squeeze(-1)  # (batch, num_objectives)
        
        # Predict objective weights if not provided
        if objective_weights is None:
            objective_weights = self.weight_predictor(features)  # (batch, num_objectives)
        
        # Scalarize Q-values
        scalarized_q = self.scalarization(q_values, objective_weights)  # (batch, action_dim)
        
        # Compute policy logits
        policy_logits = self.policy_head(features)  # (batch, action_dim)
        
        # Action probabilities
        action_probs = F.softmax(policy_logits, dim=-1)  # (batch, action_dim)
        
        return {
            'action_probs': action_probs,
            'policy_logits': policy_logits,
            'q_values': q_values,
            'scalarized_q': scalarized_q,
            'values': values,
            'objective_weights': objective_weights,
            'features': features
        }
    
    def select_action(self, state: torch.Tensor, 
                     deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select action from policy
        
        Args:
            state: (batch, state_dim)
            deterministic: Whether to select deterministically
            
        Returns:
            actions: (batch,)
            log_probs: (batch,)
        """
        outputs = self.forward(state)
        action_probs = outputs['action_probs']
        
        if deterministic:
            actions = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from categorical distribution
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()
        
        # Compute log probabilities
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-10)
        
        return actions, log_probs
    
    def evaluate_actions(self, state: torch.Tensor, 
                        actions: torch.Tensor) -> Dict:
        """
        Evaluate actions for PPO update
        
        Args:
            state: (batch, state_dim)
            actions: (batch,)
            
        Returns:
            Dictionary with log_probs, values, entropy
        """
        outputs = self.forward(state)
        action_probs = outputs['action_probs']
        
        # Log probabilities of taken actions
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-10)
        
        # Entropy for exploration bonus
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=-1)
        
        return {
            'log_probs': log_probs,
            'values': outputs['values'],
            'entropy': entropy,
            'q_values': outputs['q_values'],
            'objective_weights': outputs['objective_weights']
        }


class ScalarizationLayer(nn.Module):
    """
    Scalarization layer for multi-objective Q-values
    Supports: weighted sum, Chebyshev, hypervolume
    """
    
    def __init__(self, num_objectives: int, method: str = 'weighted_sum'):
        super().__init__()
        
        self.num_objectives = num_objectives
        self.method = method
        
        # Reference point for Chebyshev scalarization
        self.register_buffer('reference_point', torch.zeros(num_objectives))
        
    def forward(self, q_values: torch.Tensor, 
                weights: torch.Tensor) -> torch.Tensor:
        """
        Scalarize multi-objective Q-values
        
        Args:
            q_values: (batch, num_objectives, action_dim)
            weights: (batch, num_objectives)
            
        Returns:
            Scalarized Q-values (batch, action_dim)
        """
        if self.method == 'weighted_sum':
            # Weighted sum scalarization
            weights_expanded = weights.unsqueeze(-1)  # (batch, num_objectives, 1)
            scalarized = (q_values * weights_expanded).sum(dim=1)  # (batch, action_dim)
            
        elif self.method == 'chebyshev':
            # Chebyshev scalarization
            weights_expanded = weights.unsqueeze(-1)  # (batch, num_objectives, 1)
            reference_expanded = self.reference_point.unsqueeze(0).unsqueeze(-1)  # (1, num_objectives, 1)
            
            weighted_diff = weights_expanded * (q_values - reference_expanded)
            scalarized = torch.min(weighted_diff, dim=1)[0]  # (batch, action_dim)
            
        else:
            # Default to weighted sum
            weights_expanded = weights.unsqueeze(-1)
            scalarized = (q_values * weights_expanded).sum(dim=1)
        
        return scalarized


class PPOAgent:
    """
    PPO agent for training the policy network
    """
    
    def __init__(self, policy: MultiObjectivePolicyNetwork, config: Dict):
        self.policy = policy
        self.config = config
        training_config = config['training']
        ppo_config = training_config['ppo']
        
        # PPO hyperparameters
        self.clip_epsilon = ppo_config['clip_epsilon']
        self.ppo_epochs = ppo_config['epochs']
        self.batch_size = training_config['batch_size']
        self.gamma = ppo_config['gamma']
        self.gae_lambda = ppo_config['lambda_gae']
        self.value_loss_coef = ppo_config['value_loss_coef']
        self.entropy_coef = ppo_config['entropy_coef']
        constraint_cfg = training_config.get('rl', {}).get('constraint_aware', {})
        self.constraint_aware_enabled = bool(constraint_cfg.get('enabled', False))
        self.constraint_penalty_coef = float(constraint_cfg.get('penalty_coef', 0.1))
        
        # Optimizers
        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=training_config['learning_rate']
        )
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_div': []
        }
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                   dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation
        
        Args:
            rewards: (batch, seq_len, num_objectives)
            values: (batch, seq_len, num_objectives)
            dones: (batch, seq_len)
            
        Returns:
            advantages: (batch, seq_len, num_objectives)
            returns: (batch, seq_len, num_objectives)
        """
        batch_size, seq_len, num_objectives = rewards.shape
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # For each objective
        for obj in range(num_objectives):
            gae = 0
            for t in reversed(range(seq_len)):
                if t == seq_len - 1:
                    next_value = 0
                else:
                    next_value = values[:, t + 1, obj] * (1 - dones[:, t])
                
                delta = rewards[:, t, obj] + self.gamma * next_value - values[:, t, obj]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[:, t]) * gae
                
                advantages[:, t, obj] = gae
                returns[:, t, obj] = gae + values[:, t, obj]
        
        return advantages, returns
    
    def update(self, states: torch.Tensor, actions: torch.Tensor, 
              old_log_probs: torch.Tensor, returns: torch.Tensor,
              advantages: torch.Tensor,
              constraint_penalty: torch.Tensor = None) -> Dict:
        """
        Update policy using PPO
        
        Args:
            states: (batch, state_dim)
            actions: (batch,)
            old_log_probs: (batch,)
            returns: (batch, num_objectives)
            advantages: (batch, num_objectives)
            
        Returns:
            Dictionary with loss components
        """
        total_losses = []
        
        for epoch in range(self.ppo_epochs):
            # Evaluate current policy
            eval_outputs = self.policy.evaluate_actions(states, actions)
            
            new_log_probs = eval_outputs['log_probs']
            values = eval_outputs['values']
            entropy = eval_outputs['entropy']
            
            # Ratio for PPO
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Scalarize advantages (weighted sum)
            weights = eval_outputs['objective_weights']
            advantages_scalar = (advantages * weights).sum(dim=-1)
            
            # Surrogate losses
            surr1 = ratio * advantages_scalar
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_scalar
            
            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (for all objectives)
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Optional constraint-aware PPO penalty (fairness/diversity violations).
            penalty_term = torch.tensor(0.0, device=states.device)
            if self.constraint_aware_enabled and constraint_penalty is not None:
                penalty_term = constraint_penalty.mean()

                 # Total loss
            loss = (policy_loss + 
                   self.value_loss_coef * value_loss + 
                     self.entropy_coef * entropy_loss +
                     self.constraint_penalty_coef * penalty_term)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_losses.append(loss.item())
            
            # Log statistics
            self.training_stats['policy_loss'].append(policy_loss.item())
            self.training_stats['value_loss'].append(value_loss.item())
            self.training_stats['entropy'].append(-entropy_loss.item())
            
            # Compute KL divergence for early stopping
            with torch.no_grad():
                kl_div = (old_log_probs - new_log_probs).mean()
                self.training_stats['kl_div'].append(kl_div.item())
                
                # Early stopping if KL divergence too large
                if kl_div > 0.02:
                    break
        
        return {
            'loss': np.mean(total_losses),
            'policy_loss': self.training_stats['policy_loss'][-1],
            'value_loss': self.training_stats['value_loss'][-1],
            'entropy': self.training_stats['entropy'][-1],
            'kl_div': self.training_stats['kl_div'][-1]
        }


class ParetoOptimizer:
    """
    Pareto-optimal policy selection
    Maintains a set of non-dominated policies
    """
    
    def __init__(self, num_objectives: int):
        self.num_objectives = num_objectives
        self.pareto_front = []
    
    def is_dominated(self, point1: np.ndarray, point2: np.ndarray) -> bool:
        """Check if point1 is dominated by point2"""
        return np.all(point2 >= point1) and np.any(point2 > point1)
    
    def update_front(self, new_point: np.ndarray, policy_state: Dict):
        """Update Pareto front with new point"""
        # Check if new point is dominated
        dominated = False
        for point, _ in self.pareto_front:
            if self.is_dominated(new_point, point):
                dominated = True
                break
        
        if not dominated:
            # Remove dominated points
            self.pareto_front = [
                (point, state) for point, state in self.pareto_front
                if not self.is_dominated(point, new_point)
            ]
            
            # Add new point
            self.pareto_front.append((new_point, policy_state))
    
    def get_best_policy(self, preferences: np.ndarray) -> Dict:
        """Get best policy from Pareto front given preferences"""
        if not self.pareto_front:
            return None
        
        # Weighted sum of objectives
        best_score = -float('inf')
        best_policy = None
        
        for point, policy_state in self.pareto_front:
            score = np.dot(point, preferences)
            if score > best_score:
                best_score = score
                best_policy = policy_state
        
        return best_policy


if __name__ == "__main__":
    # Test Policy Network
    import yaml
    import os
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create policy network
    print("Creating Multi-Objective Policy Network...")
    policy = MultiObjectivePolicyNetwork(config)
    
    # Test forward pass
    batch_size = 8
    state = torch.randn(batch_size, config['model']['policy']['state_dim'])
    
    print(f"\nTesting forward pass...")
    print(f"  State shape: {state.shape}")
    
    outputs = policy(state)
    
    print(f"\nOutput shapes:")
    print(f"  Action probs: {outputs['action_probs'].shape}")
    print(f"  Q-values: {outputs['q_values'].shape}")
    print(f"  Values: {outputs['values'].shape}")
    print(f"  Objective weights: {outputs['objective_weights'].shape}")
    
    # Test action selection
    print(f"\nTesting action selection...")
    actions, log_probs = policy.select_action(state, deterministic=False)
    print(f"  Actions: {actions.shape}")
    print(f"  Log probs: {log_probs.shape}")
    
    # Test PPO agent
    print(f"\nTesting PPO agent...")
    agent = PPOAgent(policy, config)
    
    # Dummy training data
    old_log_probs = torch.randn(batch_size)
    returns = torch.randn(batch_size, config['model']['policy']['num_objectives'])
    advantages = torch.randn(batch_size, config['model']['policy']['num_objectives'])
    
    update_info = agent.update(state, actions, old_log_probs, returns, advantages)
    
    print(f"  Update info: {update_info}")
    
    # Test Pareto optimizer
    print(f"\nTesting Pareto optimizer...")
    pareto = ParetoOptimizer(num_objectives=4)
    
    # Add some points
    pareto.update_front(np.array([0.8, 0.6, 0.7, 0.9]), {'epoch': 1})
    pareto.update_front(np.array([0.9, 0.5, 0.6, 0.8]), {'epoch': 2})
    pareto.update_front(np.array([0.7, 0.8, 0.9, 0.7]), {'epoch': 3})
    
    print(f"  Pareto front size: {len(pareto.pareto_front)}")
    
    # Get best policy
    preferences = np.array([0.3, 0.2, 0.3, 0.2])
    best = pareto.get_best_policy(preferences)
    print(f"  Best policy: {best}")
    
    print("\n✓ Policy Network tests passed!")
