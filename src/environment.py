"""
Conversational Recommendation Environment
Simulates user interactions for RL training
"""

import random
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from data_utils import UserSimulator, ItemCatalog


class ConversationalRecommenderEnv(gym.Env):
    """
    Gym environment for conversational recommendation
    Supports RL training with multi-objective rewards
    """
    
    def __init__(self, config: Dict, item_catalog: ItemCatalog, mode: str = 'train'):
        """
        Args:
            config: Configuration dictionary
            item_catalog: ItemCatalog instance
            mode: 'train', 'val', or 'test'
        """
        super().__init__()
        
        self.config = config
        self.item_catalog = item_catalog
        self.mode = mode
        
        # Environment parameters
        self.max_turns = config['environment']['max_turns']
        self.reward_config = config['environment']
        
        # Action space: [action_type, item_indices]
        # Action types: 0=ask_preference, 1=recommend, 2=clarify, 3=end
        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(4),
            'items': spaces.MultiBinary(config['data']['num_items'])
        })
        
        # Observation space: high-dimensional continuous state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config['model']['policy']['state_dim'],),
            dtype=np.float32
        )
        
        # User simulator
        self.user = UserSimulator(item_catalog, config)
        
        # Episode state
        self.current_turn = 0
        self.dialogue_history = []
        self.recommended_items = []
        self.user_preferences = []
        self.conversation_success = False
        self.current_user_profile = {}
        
        # Statistics tracking
        self.episode_stats = {
            'demographic_group': None,
            'items_shown': [],
            'diversity_scores': [],
            'fairness_violations': []
        }
        
    def reset(self, user_profile: Dict = None) -> np.ndarray:
        """Reset environment for new episode"""
        # Reset episode variables
        self.current_turn = 0
        self.dialogue_history = []
        self.recommended_items = []
        self.user_preferences = []
        self.conversation_success = False
        
        # Keep a copy for state construction/debugging
        self.current_user_profile = user_profile or {}

        # Reset user
        self.user.reset(user_profile)
        
        # Reset statistics
        self.episode_stats = {
            'demographic_group': self.user.age_group,
            'items_shown': [],
            'diversity_scores': [],
            'fairness_violations': []
        }
        
        # Return initial state
        initial_state = self._get_state()
        return initial_state
    
    def step(self, action: Dict) -> Tuple[np.ndarray, Dict, bool, Dict]:
        """
        Execute action and return next state, rewards, done, info
        
        Returns:
            next_state: State representation
            rewards: Dictionary of multi-objective rewards
            done: Whether episode is finished
            info: Additional information
        """
        self.current_turn += 1
        
        # Process action
        action_type = action['action_type']
        items = action.get('items', [])
        
        # Log action
        self.dialogue_history.append({
            'turn': self.current_turn,
            'action_type': action_type,
            'items': items
        })
        
        # Get user response
        user_response = self._simulate_user_response(action_type, items)
        
        # Compute rewards
        rewards = self._compute_rewards(action_type, items, user_response)
        
        # Check if done
        done = self._is_done(user_response)
        
        if done and user_response['action'] == 'accept':
            self.conversation_success = True
        
        # Get next state
        next_state = self._get_state()
        
        # Additional info
        info = {
            'turn': self.current_turn,
            'user_response': user_response,
            'success': self.conversation_success,
            'demographic_group': self.user.age_group,
            'episode_stats': self.episode_stats
        }
        
        return next_state, rewards, done, info
    
    def _simulate_user_response(self, action_type: int, items: List[str]) -> Dict:
        """Simulate user response to system action"""
        
        if action_type == 0:  # ask_preference
            preference = self.user.provide_preference()
            self.user_preferences.append(preference)
            return {
                'action': 'inform',
                'utterance': preference,
                'satisfied': False
            }
        
        elif action_type == 1:  # recommend
            self.recommended_items.extend(items)
            self.episode_stats['items_shown'].extend(items)
            response = self.user.respond_to_recommendation(items, self.current_turn)
            return response
        
        elif action_type == 2:  # clarify
            return {
                'action': 'clarify_response',
                'utterance': 'Let me clarify...',
                'satisfied': False
            }
        
        else:  # end conversation
            return {
                'action': 'goodbye',
                'utterance': 'Goodbye',
                'satisfied': False
            }
    
    def _compute_rewards(self, action_type: int, items: List[str], 
                        user_response: Dict) -> Dict:
        """Compute multi-objective rewards"""
        rewards = {}
        
        # ==== Accuracy Reward ====
        if action_type == 1:  # recommendation
            if user_response['action'] == 'accept':
                rewards['accuracy'] = self.reward_config['reward_success']
            elif user_response['action'] == 'reject':
                rewards['accuracy'] = self.reward_config['reward_reject']
            else:
                rewards['accuracy'] = 0.0
        else:
            # Efficiency penalty for non-recommendation actions
            rewards['accuracy'] = self.reward_config['reward_per_turn']
        
        # ==== Diversity Reward ====
        if action_type == 1 and len(items) > 1:
            diversity_score = self._compute_diversity(items)
            rewards['diversity'] = diversity_score * self.reward_config['reward_diversity_factor']
            self.episode_stats['diversity_scores'].append(diversity_score)
        else:
            rewards['diversity'] = 0.0
        
        # ==== Fairness Reward ====
        if action_type == 1 and items:
            fairness_score = self._compute_fairness(items)
            rewards['fairness'] = fairness_score * self.reward_config['reward_fairness_factor']
        else:
            rewards['fairness'] = 0.0
        
        # ==== Engagement Reward ====
        if user_response['action'] == 'ask_more':
            rewards['engagement'] = 5.0
        elif user_response['action'] == 'leave':
            rewards['engagement'] = -10.0
        elif user_response['action'] == 'inform':
            rewards['engagement'] = 2.0
        else:
            rewards['engagement'] = 1.0
        
        return rewards
    
    def _compute_diversity(self, items: List[str]) -> float:
        """Compute intra-list diversity"""
        if len(items) <= 1:
            return 0.0
        
        # Category diversity
        categories = set()
        for item_id in items:
            item = self.item_catalog.get_item(item_id)
            if item:
                categories.add(item.get('category', 'Unknown'))
        
        category_diversity = len(categories) / len(items)
        
        # Embedding distance diversity
        embeddings = []
        for item_id in items:
            emb = self.item_catalog.get_item_embedding(item_id)
            embeddings.append(emb)
        
        if len(embeddings) > 1:
            distances = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    dist = np.linalg.norm(embeddings[i] - embeddings[j])
                    distances.append(dist)
            
            avg_distance = np.mean(distances)
            # Normalize (assuming embeddings are normalized)
            distance_diversity = min(avg_distance / 2.0, 1.0)
        else:
            distance_diversity = 0.0
        
        # Combined diversity score
        diversity = 0.5 * category_diversity + 0.5 * distance_diversity
        
        return diversity
    
    def _compute_fairness(self, items: List[str]) -> float:
        """Compute fairness score"""
        # Simple fairness: check if we're showing diverse item providers
        # and not over-representing popular items
        
        if not items:
            return 1.0
        
        fairness_score = 0.0
        
        # Provider diversity (if available)
        providers = set()
        for item_id in items:
            item = self.item_catalog.get_item(item_id)
            if item:
                providers.add(item.get('provider', 'default'))
        
        if providers:
            provider_diversity = len(providers) / len(items)
            fairness_score += 0.5 * provider_diversity
        else:
            fairness_score += 0.5
        
        # Popularity balance (prefer less popular items sometimes)
        popularities = []
        for item_id in items:
            item = self.item_catalog.get_item(item_id)
            if item:
                popularities.append(item.get('popularity', 500))
        
        if popularities:
            avg_popularity = np.mean(popularities)
            # Reward showing less popular items (inversely proportional)
            popularity_balance = 1.0 - (avg_popularity / 1000.0)
            fairness_score += 0.5 * popularity_balance
        else:
            fairness_score += 0.5
        
        return fairness_score
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        state_dim = self.config['model']['policy']['state_dim']
        state = np.zeros(state_dim, dtype=np.float32)

        # Dialogue progress features
        state[0] = self.current_turn / self.max_turns  # Normalized turn
        state[1] = len(self.dialogue_history) / self.max_turns
        state[2] = float(self.conversation_success)

        # User profile features from the sampled training conversation.
        prefs = self.current_user_profile.get('preferences', [])
        liked = self.user.liked_items
        disliked = self.user.disliked_items
        state[3] = min(len(prefs), 20) / 20.0
        state[4] = min(len(liked), 10) / 10.0
        state[5] = min(len(disliked), 10) / 10.0

        # Encode demographics as coarse numeric features.
        age_group = self.user.age_group or 'unknown'
        age_map = {'18-25': 0.2, '26-35': 0.4, '26-40': 0.4, '36-45': 0.6, '40+': 0.8, '46-55': 0.8, '55+': 1.0}
        gender_map = {'M': 0.2, 'F': 0.6, 'Other': 0.9, 'U': 0.5}
        state[6] = age_map.get(age_group, 0.5)
        state[7] = gender_map.get(self.user.gender, 0.5)

        # Lightweight deterministic noise based on interaction counts.
        seed_val = (len(self.recommended_items) * 131 + self.current_turn * 17) % 10000
        rng = np.random.RandomState(seed_val)
        state[8:] = rng.randn(state_dim - 8).astype(np.float32) * 0.03
        
        return state
    
    def _is_done(self, user_response: Dict) -> bool:
        """Check if episode should end"""
        if user_response['action'] in ['accept', 'leave', 'goodbye']:
            return True
        
        if self.current_turn >= self.max_turns:
            return True
        
        return False
    
    def render(self, mode='human'):
        """Render environment state"""
        print(f"\n=== Turn {self.current_turn} ===")
        print(f"User: {self.user.age_group}, {self.user.gender}")
        print(f"History length: {len(self.dialogue_history)}")
        print(f"Items shown: {len(self.recommended_items)}")
        print(f"Success: {self.conversation_success}")
    
    def get_episode_stats(self) -> Dict:
        """Get statistics for the episode"""
        return {
            'turns': self.current_turn,
            'success': self.conversation_success,
            'items_recommended': len(set(self.recommended_items)),
            'avg_diversity': np.mean(self.episode_stats['diversity_scores']) 
                           if self.episode_stats['diversity_scores'] else 0.0,
            'demographic_group': self.episode_stats['demographic_group'],
        }


class BatchedConversationalEnv:
    """Batched version of the environment for parallel training"""
    
    def __init__(self, config: Dict, item_catalog: ItemCatalog, 
                 num_envs: int = 4, mode: str = 'train'):
        """
        Args:
            config: Configuration dictionary
            item_catalog: ItemCatalog instance
            num_envs: Number of parallel environments
            mode: 'train', 'val', or 'test'
        """
        self.num_envs = num_envs
        self.envs = [
            ConversationalRecommenderEnv(config, item_catalog, mode)
            for _ in range(num_envs)
        ]
    
    def reset(self, user_profiles: List[Dict] = None) -> np.ndarray:
        """Reset all environments with optional per-env user profiles."""
        states = []
        for idx, env in enumerate(self.envs):
            profile = None
            if user_profiles is not None and idx < len(user_profiles):
                profile = user_profiles[idx]
            state = env.reset(profile)
            states.append(state)
        return np.array(states)
    
    def step(self, actions: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments"""
        next_states = []
        all_rewards = []
        dones = []
        infos = []
        
        for env, action in zip(self.envs, actions):
            next_state, rewards, done, info = env.step(action)
            next_states.append(next_state)
            
            # Convert rewards dict to array
            reward_array = np.array([
                rewards['accuracy'],
                rewards['diversity'],
                rewards['fairness'],
                rewards['engagement']
            ])
            all_rewards.append(reward_array)
            
            dones.append(done)
            infos.append(info)
        
        return (np.array(next_states), 
                np.array(all_rewards), 
                np.array(dones), 
                infos)
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()


if __name__ == "__main__":
    # Test environment
    import yaml
    import os
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create catalog
    catalog = ItemCatalog(config['data']['catalog_file'])
    
    # Create environment
    env = ConversationalRecommenderEnv(config, catalog, mode='train')
    
    # Test episode
    print("Testing environment...")
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    done = False
    total_rewards = {'accuracy': 0, 'diversity': 0, 'fairness': 0, 'engagement': 0}
    
    for turn in range(5):
        # Random action
        action = {
            'action_type': random.randint(0, 3),
            'items': catalog.sample_items(5) if random.random() > 0.5 else []
        }
        
        next_state, rewards, done, info = env.step(action)
        
        print(f"\nTurn {turn + 1}:")
        print(f"  Action type: {action['action_type']}")
        print(f"  Rewards: {rewards}")
        print(f"  Done: {done}")
        
        for obj in rewards:
            total_rewards[obj] += rewards[obj]
        
        if done:
            print(f"\nEpisode finished!")
            print(f"Success: {info['success']}")
            break
    
    print(f"\nTotal rewards: {total_rewards}")
    print(f"Episode stats: {env.get_episode_stats()}")
