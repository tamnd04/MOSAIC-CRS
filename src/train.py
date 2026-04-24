"""
Training script for MO-CRS
Implements PPO training with multi-objective rewards
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
import json
from tqdm import tqdm
import wandb
from typing import Dict, List, Optional
import argparse
import random
from mocrs import MOCRS
from policy_network import PPOAgent
from environment import ConversationalRecommenderEnv, BatchedConversationalEnv
from data_utils import ReDialDataset, ItemCatalog, apply_dataset_paths, create_dataloaders, create_stronger_validation_splits
from off_policy_evaluation import off_policy_evaluate
from test_evaluation_suite import evaluate_full_test_suite
class MOCRSTrainer:
    """
    Trainer for Multi-Objective CRS
    Handles both supervised pre-training and RL fine-tuning
    """
    @staticmethod
    def _infer_sentiment_label(utterance: str) -> int:
        """Infer coarse sentiment label: 0=negative, 1=neutral, 2=positive."""
        text = str(utterance or '').lower()
        pos_markers = ['love', 'like', 'great', 'awesome', 'good', 'yes', 'thanks', 'enjoy']
        neg_markers = ['hate', 'dislike', 'bad', 'awful', 'terrible', 'no', 'boring', 'worse']
        pos = sum(1 for token in pos_markers if token in text)
        neg = sum(1 for token in neg_markers if token in text)
        if pos > neg:
            return 2
        if neg > pos:
            return 0
        return 1
    @staticmethod
    def _build_static_features_from_profile(profile: Dict, feature_dim: int) -> np.ndarray:
        """Encode user profile into deterministic static feature vector."""
        vec = np.zeros(feature_dim, dtype=np.float32)
        age_map = {
            '18-25': 0.2,
            '26-35': 0.4,
            '26-40': 0.4,
            '36-45': 0.6,
            '40+': 0.8,
            '46-55': 0.8,
            '55+': 1.0,
        }
        gender = str(profile.get('gender', 'U')).upper()
        prefs = profile.get('preferences', [])
        pref_count = len(prefs) if isinstance(prefs, list) else 0
        vec[0] = age_map.get(str(profile.get('age_group', 'unknown')), 0.5)
        vec[1] = 1.0 if gender == 'F' else 0.0
        vec[2] = 1.0 if gender == 'M' else 0.0
        vec[3] = min(pref_count, 20) / 20.0
        return vec
    @staticmethod
    def _extract_preference_signal(conversation: Dict) -> float:
        """Compute preference supervision signal from accepted/recommended overlap."""
        turns = conversation.get('turns', []) if isinstance(conversation, dict) else []
        accepted_items = set(str(x) for x in conversation.get('accepted_items', []) if x is not None)
        mentioned_items = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            for key in ['mentioned_items', 'items_mentioned', 'recommended_items']:
                values = turn.get(key, [])
                if isinstance(values, list):
                    mentioned_items.extend(str(v) for v in values if v is not None)
        unique_mentioned = set(mentioned_items)
        if not accepted_items and not unique_mentioned:
            return 0.0
        overlap = len(accepted_items.intersection(unique_mentioned))
        precision = overlap / max(len(unique_mentioned), 1)
        recall = overlap / max(len(accepted_items), 1) if accepted_items else 0.0
        if precision + recall == 0:
            return 0.0
        return float((2.0 * precision * recall) / (precision + recall))
    
    def __init__(self, config: Dict, use_wandb: bool = False):
        self.config = config
        self.use_wandb = use_wandb
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        print("Initializing MO-CRS...")
        self.model = MOCRS(config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # PPO agent for RL training
        self.ppo_agent = PPOAgent(self.model.policy, config)
        
        # Item catalog
        print("Loading item catalog...")
        self.item_catalog = ItemCatalog(config['data']['catalog_file'])
        self._initialize_catalog_tensor_cache()
        
        # Training statistics
        self.train_stats = {
            'epoch': 0,
            'global_step': 0,
            'best_val_loss': float('inf'),
            'best_val_metrics': {},
            'supervised_history': [],
            'rl_history': [],
            'rl_eval_history': []
        }
        
        # Initialize wandb
        if self.use_wandb:
            log_cfg = config.get('logging', {})
            wandb.init(
                project=log_cfg.get('wandb_project', 'mocrs-thesis'),
                config=config,
                name=log_cfg.get('experiment_name', 'mocrs-baseline')
            )
    def _initialize_catalog_tensor_cache(self) -> None:
        """Build one-time catalog caches for faster candidate sampling and fairness shaping."""
        self._catalog_item_ids = list(self.item_catalog.item_ids)
        self._catalog_size = len(self._catalog_item_ids)
        self._catalog_title_by_id = {
            str(item_id): str((self.item_catalog.get_item(item_id) or {}).get('title', item_id))
            for item_id in self._catalog_item_ids
        }
        self._fast_candidate_sampling = False
        self._catalog_embedding_matrix = None
        self._catalog_popularity = np.zeros((0,), dtype=np.float32)
        self._catalog_popularity_by_id = {}
        self._tail_sampling_probs = None
        self._head_item_set = set()
        self._tail_item_set = set()
        self._mid_item_set = set()
        if self._catalog_size == 0:
            return
        emb_dim = int(self.config.get('model', {}).get('diversity_fairness', {}).get('item_embedding_dim', 128))
        emb_matrix = np.zeros((self._catalog_size, emb_dim), dtype=np.float32)
        popularity = np.zeros((self._catalog_size,), dtype=np.float32)
        for idx, item_id in enumerate(self._catalog_item_ids):
            item = self.item_catalog.get_item(item_id) or {}
            emb = np.asarray(self.item_catalog.get_item_embedding(item_id), dtype=np.float32).reshape(-1)
            copy_dim = min(emb_dim, emb.shape[0])
            if copy_dim > 0:
                emb_matrix[idx, :copy_dim] = emb[:copy_dim]
            popularity[idx] = float(item.get('mentions', item.get('popularity', 0.0)) or 0.0)
        self._catalog_embedding_matrix = torch.as_tensor(emb_matrix, dtype=torch.float32, device=self.device)
        self._catalog_popularity = popularity
        self._catalog_popularity_by_id = {
            str(item_id): float(popularity[idx]) for idx, item_id in enumerate(self._catalog_item_ids)
        }
        self._fast_candidate_sampling = True
        tail_weights = 1.0 / np.power(popularity + 1.0, 0.75)
        tail_weights = tail_weights / max(np.sum(tail_weights), 1e-8)
        self._tail_sampling_probs = tail_weights
        order = np.argsort(popularity)[::-1]
        head_cut = max(1, int(0.2 * self._catalog_size))
        tail_cut = max(1, int(0.5 * self._catalog_size))
        self._head_item_set = {self._catalog_item_ids[int(i)] for i in order[:head_cut]}
        self._tail_item_set = {self._catalog_item_ids[int(i)] for i in order[-tail_cut:]}
        self._mid_item_set = set(self._catalog_item_ids) - self._head_item_set - self._tail_item_set
        # Register popularity priors inside the reranker so fairness can directly penalize head items.
        if hasattr(self.model, 'dfc') and hasattr(self.model.dfc, 'set_catalog_popularity'):
            self.model.dfc.set_catalog_popularity(self._catalog_popularity_by_id)
        print(f"Fast catalog cache ready: items={self._catalog_size}, emb_dim={emb_dim}")
    def _item_popularity(self, item_id: str) -> float:
        item = self.item_catalog.get_item(item_id) or {}
        return float(item.get('mentions', item.get('popularity', 0.0)) or 0.0)
    def _build_policy_utterance(self, user_profile: Optional[Dict]) -> str:
        if not user_profile:
            return 'I am looking for a movie recommendation.'
        prefs = [str(x) for x in user_profile.get('preferences', []) if x is not None]
        if not prefs:
            return 'I am open to movie recommendations.'
        titles = []
        categories = []
        for item_id in prefs[:3]:
            item = self.item_catalog.get_item(item_id) or {}
            title = str(item.get('title', item_id))
            if title:
                titles.append(title)
            category = str(item.get('category', ''))
            if category:
                categories.append(category)
        if titles:
            return f"I liked {', '.join(titles[:2])}. Recommend me something similar."
        if categories:
            return f"I usually enjoy {categories[0]} movies."
        return 'I am looking for something good to watch.'
    def _build_candidate_annotations(self, candidate_ids_batch: List[List[str]]):
        candidate_item_names = []
        candidate_item_metadata = []
        for ids in candidate_ids_batch:
            names = []
            metadata = []
            for item_id in ids:
                item = self.item_catalog.get_item(item_id) or {}
                names.append(str(item.get('title', item_id)))
                genres = item.get('genres', item.get('genre', []))
                if isinstance(genres, str):
                    genres = [g.strip() for g in genres.split('|') if g.strip()]
                metadata.append({
                    'id': str(item_id),
                    'name': str(item.get('title', item_id)),
                    'title': str(item.get('title', item_id)),
                    'genre': '|'.join(genres) if isinstance(genres, list) and genres else str(item.get('genre', item.get('category', 'Unknown'))),
                    'genres': genres if isinstance(genres, list) else [],
                    'category': str(item.get('category', 'Unknown')),
                    'year': item.get('year', ''),
                    'rating': item.get('rating', 0.0),
                    'mentions': self._item_popularity(item_id),
                })
            candidate_item_names.append(names)
            candidate_item_metadata.append(metadata)
        return candidate_item_names, candidate_item_metadata
    def _fairness_tail_bonus(self, items: List[str]) -> float:
        if not items:
            return 0.0
        head = sum(1 for item in items if item in self._head_item_set)
        tail = sum(1 for item in items if item in self._tail_item_set)
        non_head = max(len(items) - head, 0)
        categories = set()
        avg_pop = 0.0
        for item_id in items:
            item = self.item_catalog.get_item(item_id) or {}
            categories.add(str(item.get('category', 'Unknown')))
            avg_pop += self._item_popularity(item_id)
        avg_pop = avg_pop / max(len(items), 1)
        pop_norm = 1.0 / (1.0 + np.log1p(max(avg_pop, 0.0)))
        mix_bonus = len(categories) / max(len(items), 1)
        return (
            0.45 * ((tail - head) / max(len(items), 1)) +
            0.15 * ((non_head / max(len(items), 1)) - 0.5) +
            0.10 * mix_bonus +
            0.15 * pop_norm
        )
    def _compute_action_shaping(self, env_actions: List[Dict], current_turn: int, recommended_counts: np.ndarray) -> np.ndarray:
        bonuses = np.zeros((len(env_actions), 4), dtype=np.float32)
        for i, env_action in enumerate(env_actions):
            action_type = int(env_action.get('action_type', 0))
            rec_count = int(recommended_counts[i]) if i < len(recommended_counts) else 0
            items = [str(x) for x in env_action.get('items', [])]
            if current_turn == 1 and rec_count == 0:
                if action_type == 0:
                    bonuses[i, 3] += 0.35
                elif action_type == 1:
                    bonuses[i, 3] -= 0.12
            if current_turn <= 2 and rec_count == 0 and action_type == 2:
                bonuses[i, 3] += 0.05
            if action_type == 1 and items:
                bonuses[i, 2] += self._fairness_tail_bonus(items)
                head_share = sum(1 for item in items if item in self._head_item_set) / max(len(items), 1)
                if head_share > 0.6:
                    bonuses[i, 2] -= 0.20
                elif head_share < 0.35:
                    bonuses[i, 2] += 0.10
                if rec_count == 0 and current_turn <= 2:
                    bonuses[i, 3] -= 0.05
            if action_type == 3 and rec_count == 0:
                bonuses[i, 3] -= 0.60
        return bonuses
    def supervised_pretrain(self, train_loader: DataLoader, 
                           val_loader: DataLoader, 
                           num_epochs: int):
        """
        Pre-train with supervised learning on conversation logs
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
        """
        print("\n" + "="*60)
        print("Starting Supervised Pre-training")
        print("="*60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self._train_epoch(train_loader)
            
            # Validation
            val_metrics = self._validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['total_loss'])
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Save checkpoint
            if val_metrics['total_loss'] < self.train_stats['best_val_loss']:
                self.train_stats['best_val_loss'] = val_metrics['total_loss']
                self.save_checkpoint('best_model.pt')
                print(f"[OK] Saved new best model (val_loss: {val_metrics['total_loss']:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config['training']['save_frequency'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        print("\n[OK] Pre-training completed!")
    def behavioral_cloning_warmstart(self, conversations: List[Dict], epochs: int, learning_rate: float):
        """Behavioral cloning warm-start to imitate logged conversation actions."""
        if not conversations:
            print("[WARN] No conversations found for behavioral cloning warm-start")
            return
        print("\n" + "="*60)
        print("Starting Behavioral Cloning Warm-Start")
        print("="*60)
        policy = self.model.policy
        policy.train()
        bc_optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        batch_size = int(self.config['training'].get('batch_size', 64))
        for epoch in range(epochs):
            np.random.shuffle(conversations)
            epoch_losses = []
            for start in range(0, len(conversations), batch_size):
                batch_convs = conversations[start:start + batch_size]
                states_np, labels_np = self._build_bc_batch(batch_convs)
                states = torch.FloatTensor(states_np).to(self.device)
                labels = torch.LongTensor(labels_np).to(self.device)
                outputs = policy(states)
                logits = outputs['policy_logits']
                loss = nn.CrossEntropyLoss()(logits, labels)
                bc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                bc_optimizer.step()
                epoch_losses.append(loss.item())
            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            print(f"BC Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")
        print("[OK] Behavioral cloning warm-start completed")
    
    def _train_epoch(self, data_loader: DataLoader) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = []
        epoch_metrics = {
            'intent_loss': [],
            'sentiment_loss': [],
            'preference_loss': [],
            'total_loss': []
        }
        
        pbar = tqdm(data_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Training step
            losses = self.model.train_step(batch, self.optimizer)
            
            # Accumulate losses
            for key, value in losses.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({'loss': losses.get('total_loss', 0.0)})
            
            self.train_stats['global_step'] += 1
        
        # Average metrics
        avg_metrics = {
            key: np.mean(values) if values else 0.0
            for key, values in epoch_metrics.items()
        }
        
        return avg_metrics
    
    @torch.no_grad()
    def _validate_epoch(self, data_loader: DataLoader) -> Dict:
        """Validate for one epoch"""
        self.model.eval()
        
        epoch_metrics = {
            'intent_accuracy': [],
            'sentiment_accuracy': [],
            'total_loss': []
        }
        
        pbar = tqdm(data_loader, desc="Validation")
        
        for batch in pbar:
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Evaluation
            metrics = self.model.evaluate(batch)
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {
            key: np.mean(values) if values else 0.0
            for key, values in epoch_metrics.items()
        }
        
        return avg_metrics
    
    def rl_finetune(self, num_episodes: int = 1000):
        """Fine-tune with reinforcement learning."""
        print("\n" + "="*60)
        print("Starting RL Fine-tuning")
        print("="*60)
        env = BatchedConversationalEnv(
            self.config,
            self.item_catalog,
            num_envs=self.config['training']['num_envs']
        )
        rl_cfg = self.config['training'].get('rl', {})
        rollout_horizon = int(rl_cfg.get('rollout_horizon', self.config['environment']['max_turns']))
        candidate_pool_size = int(rl_cfg.get('candidate_pool_size', 100))
        top_k_recommendations = int(rl_cfg.get('top_k_recommendations', 10))
        log_interval = int(self.config['training'].get('log_interval', 10))
        eval_interval = int(rl_cfg.get('eval_interval_episodes', self.config['training'].get('eval_interval', 100)))
        save_interval = int(rl_cfg.get('save_interval_episodes', self.config['training'].get('save_interval', 500)))
        val_file = self.config.get('data', {}).get('val_file')
        best_dr = -float('inf')
        conversations = self._load_rl_conversations()
        if not conversations:
            raise ValueError("RL fine-tuning requires data.train_file with at least one conversation")
        target_conversations = len(conversations)
        data_passes = 1
        base_episodes = num_episodes
        if rl_cfg.get('use_full_dataset', True):
            max_convs = int(rl_cfg.get('max_conversations', target_conversations))
            target_conversations = min(target_conversations, max_convs)
        num_envs = self.config['training']['num_envs']
        if rl_cfg.get('use_full_dataset', True):
            base_episodes = int(np.ceil(target_conversations / max(num_envs, 1)))
            data_passes = max(1, int(rl_cfg.get('passes_over_data', 1)))
            num_episodes = base_episodes * data_passes
        print(
            f"RL data setup: target_conversations={target_conversations}, "
            f"num_envs={num_envs}, episodes={num_episodes}, "
            f"base_episodes={base_episodes}, passes_over_data={data_passes}"
        )
        reward_w_cfg = rl_cfg.get('reward_weights', {})
        reward_weights = np.array([
            reward_w_cfg.get('accuracy', 0.35),
            reward_w_cfg.get('diversity', 0.15),
            reward_w_cfg.get('fairness', 0.25),
            reward_w_cfg.get('engagement', 0.25)
        ], dtype=np.float32)
        eps_start = float(self.config['training'].get('epsilon_start', 0.2))
        eps_end = float(self.config['training'].get('epsilon_end', 0.02))
        eps_decay = float(self.config['training'].get('epsilon_decay', 0.997))
        exploration_recommend_bias = float(rl_cfg.get('exploration_recommend_bias', 0.6))
        min_turns_before_end = int(rl_cfg.get('min_turns_before_end', self.config.get('environment', {}).get('min_turns_before_end', 0)))
        min_recommendations_before_end = int(rl_cfg.get('min_recommendations_before_end', self.config.get('environment', {}).get('min_recommendations_before_end', 0)))
        fallback_action_before_end = int(rl_cfg.get('fallback_action_before_end', 1))
        ask_before_recommend_turns = int(rl_cfg.get('ask_before_recommend_turns', 1))
        seen_conversations = set()
        online_success_hist, online_turns_hist, online_diversity_hist, online_fairness_hist = [], [], [], []
        coverage_announced = False
        div_scale = float(self.config.get('environment', {}).get('reward_diversity_factor', 1.0))
        fair_scale = float(self.config.get('environment', {}).get('reward_fairness_factor', 1.0))
        pe_cfg = self.config.get('model', {}).get('personalization', {})
        use_thompson_sampling = bool(pe_cfg.get('thompson_sampling', {}).get('enabled', True))
        episode_iterator = tqdm(range(num_episodes), desc='RL Fine-tuning', unit='ep')
        for episode in episode_iterator:
            current_epsilon = max(eps_end, eps_start * (eps_decay ** episode))
            user_profiles = []
            for env_idx in range(num_envs):
                conv_idx = (episode * num_envs + env_idx) % target_conversations
                seen_conversations.add(conv_idx)
                user_profiles.append(self._conversation_to_user_profile(conversations[conv_idx], conv_idx))
            states_np = env.reset(user_profiles=user_profiles)
            states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
            traj_states, traj_actions, traj_log_probs, traj_values, traj_rewards, traj_dones = [], [], [], [], [], []
            scalar_rewards = []
            episode_length = 0
            done = False
            episode_diversity_values, episode_fairness_values = [], []
            episode_action_counts = np.zeros(4, dtype=np.int32)
            episode_recommend_counts = np.zeros(num_envs, dtype=np.int32)
            last_infos = []
            while not done and episode_length < rollout_horizon:
                candidate_ids_batch, candidate_embeddings = self._sample_candidate_items(
                    num_envs=len(states),
                    candidate_pool_size=candidate_pool_size,
                    user_profiles=user_profiles,
                )
                batch = self._create_rl_batch(
                    states,
                    candidate_embeddings,
                    candidate_ids_batch,
                    user_profiles=user_profiles,
                    is_cold_start=(episode_length == 0),
                    use_thompson_sampling=use_thompson_sampling,
                )
                with torch.no_grad():
                    outputs = self.model(batch)
                    action_probs = outputs.get('action_probs')
                    values = outputs['values']
                    reranked_indices = outputs.get('reranked_indices')
                    if action_probs is None:
                        raise RuntimeError('Policy outputs must include action_probs for RL fine-tuning.')
                    dist = torch.distributions.Categorical(action_probs)
                    actions = dist.sample()
                if current_epsilon > 0.0:
                    random_mask = torch.rand(actions.shape, device=actions.device) < current_epsilon
                    random_actions = torch.randint(0, 4, actions.shape, device=actions.device)
                    if exploration_recommend_bias > 0.0:
                        rec_mask = torch.rand(actions.shape, device=actions.device) < exploration_recommend_bias
                        random_actions = torch.where(rec_mask, torch.ones_like(random_actions), random_actions)
                    actions = torch.where(random_mask, random_actions, actions)
                env_actions = self._build_env_actions(
                    actions.detach().cpu().numpy(),
                    candidate_ids_batch,
                    reranked_indices,
                    top_k=top_k_recommendations,
                    current_turn=episode_length + 1,
                    min_turns_before_end=min_turns_before_end,
                    recommended_counts=episode_recommend_counts.tolist(),
                    min_recommendations_before_end=min_recommendations_before_end,
                    fallback_action_type=fallback_action_before_end,
                    ask_before_recommend_turns=ask_before_recommend_turns,
                )
                executed_actions_np = np.array([int(a.get('action_type', 0)) for a in env_actions], dtype=np.int64)
                executed_actions = torch.as_tensor(executed_actions_np, dtype=torch.long, device=self.device)
                executed_log_probs = torch.log(action_probs.gather(1, executed_actions.unsqueeze(-1)).squeeze(-1) + 1e-10)
                for env_idx, env_action in enumerate(env_actions):
                    action_type = int(env_action.get('action_type', 0))
                    if 0 <= action_type < 4:
                        episode_action_counts[action_type] += 1
                    if action_type == 1 and env_action.get('items', []):
                        episode_recommend_counts[env_idx] += 1
                next_states_np, rewards_np, dones_np, infos = env.step(env_actions)
                rewards_np = rewards_np + self._compute_action_shaping(env_actions, episode_length + 1, episode_recommend_counts)
                next_states = torch.as_tensor(next_states_np, dtype=torch.float32, device=self.device)
                last_infos = infos
                traj_states.append(states.detach())
                traj_actions.append(executed_actions.detach())
                traj_log_probs.append(executed_log_probs.detach())
                traj_values.append(values.detach())
                traj_rewards.append(torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device))
                traj_dones.append(torch.as_tensor(dones_np.astype(np.float32), dtype=torch.float32, device=self.device))
                scalar_rewards.append((rewards_np * reward_weights).sum(axis=1).mean())
                episode_diversity_values.append(rewards_np[:, 1] / max(div_scale, 1e-8))
                episode_fairness_values.append(rewards_np[:, 2] / max(fair_scale, 1e-8))
                states = next_states
                episode_length += 1
                done = bool(dones_np.all())
            if traj_states:
                update_stats = self._update_policy_from_rollout(traj_states, traj_actions, traj_log_probs, traj_values, traj_rewards, traj_dones)
            else:
                update_stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'loss': 0.0, 'kl_div': 0.0, 'entropy': 0.0}
            if last_infos:
                episode_success = float(np.mean([1.0 if info.get('success', False) else 0.0 for info in last_infos]))
                episode_turns_avg = float(np.mean([float(info.get('turn', episode_length)) for info in last_infos]))
                diversity_from_info = []
                for info in last_infos:
                    div_scores = info.get('episode_stats', {}).get('diversity_scores', [])
                    if div_scores:
                        diversity_from_info.append(float(np.mean(div_scores)))
                if diversity_from_info:
                    episode_diversity = float(np.mean(diversity_from_info))
                elif episode_diversity_values:
                    episode_diversity = float(np.mean(np.concatenate(episode_diversity_values)))
                else:
                    episode_diversity = 0.0
            else:
                episode_success = 0.0
                episode_turns_avg = float(episode_length)
                episode_diversity = float(np.mean(np.concatenate(episode_diversity_values))) if episode_diversity_values else 0.0
            episode_fairness = float(np.mean(np.concatenate(episode_fairness_values))) if episode_fairness_values else 0.0
            total_actions = int(episode_action_counts.sum())
            action_ratios = (episode_action_counts.astype(np.float32) / float(total_actions)) if total_actions > 0 else np.zeros(4, dtype=np.float32)
            online_success_hist.append(episode_success)
            online_turns_hist.append(episode_turns_avg)
            online_diversity_hist.append(episode_diversity)
            online_fairness_hist.append(episode_fairness)
            self.train_stats.setdefault('rl_history', []).append({
                'episode': int(episode + 1),
                'avg_reward': float(np.mean(scalar_rewards)) if scalar_rewards else 0.0,
                'episode_length': int(episode_length),
                'epsilon': float(current_epsilon),
                'policy_loss': float(update_stats.get('policy_loss', 0.0)),
                'value_loss': float(update_stats.get('value_loss', 0.0)),
                'online_success': float(episode_success),
                'online_turns': float(episode_turns_avg),
                'online_diversity': float(episode_diversity),
                'online_fairness': float(episode_fairness),
                'seen_conversations': int(len(seen_conversations)),
                'action_ask': float(action_ratios[0]),
                'action_recommend': float(action_ratios[1]),
                'action_clarify': float(action_ratios[2]),
                'action_end': float(action_ratios[3]),
            })
            if (episode + 1) % log_interval == 0:
                print(
                    f"Episode {episode + 1}/{num_episodes}: "
                    f"Avg Reward={float(np.mean(scalar_rewards)):.3f}, "
                    f"Turns={episode_length}, "
                    f"Eps={current_epsilon:.3f}, "
                    f"Policy Loss={update_stats['policy_loss']:.4f}, "
                    f"Seen Conversations={len(seen_conversations)}/{target_conversations}, "
                    f"Online Success={episode_success:.3f}, Online Turns={episode_turns_avg:.2f}, "
                    f"Online Diversity={episode_diversity:.3f}, Online Fairness={episode_fairness:.3f}, "
                    f"Actions[ask={action_ratios[0]:.2f}, rec={action_ratios[1]:.2f}, "
                    f"clar={action_ratios[2]:.2f}, end={action_ratios[3]:.2f}]"
                )
            episode_iterator.set_postfix({'reward': f"{float(np.mean(scalar_rewards)) if scalar_rewards else 0.0:.3f}", 'success': f"{episode_success:.2f}", 'seen': f"{len(seen_conversations)}/{target_conversations}"})
            if save_interval > 0 and (episode + 1) % save_interval == 0:
                self.save_checkpoint(f'rl_checkpoint_ep{episode+1}.pt')
            if eval_interval > 0 and (episode + 1) % eval_interval == 0 and val_file and os.path.exists(val_file):
                ope_metrics = off_policy_evaluate(self.model, val_file, self.item_catalog, self.config, self.device)
                dr_score = float(ope_metrics.get('dr', -float('inf')))
                ips_val = float(ope_metrics.get('ips', 0.0))
                snips_val = float(ope_metrics.get('snips', 0.0))
                dm_val = float(ope_metrics.get('dm', 0.0))
                dr_ci_low = float(ope_metrics.get('dr_ci_low', dr_score))
                dr_ci_high = float(ope_metrics.get('dr_ci_high', dr_score))
                dm_ci_low = float(ope_metrics.get('dm_ci_low', dm_val))
                dm_ci_high = float(ope_metrics.get('dm_ci_high', dm_val))
                ci_level = float(ope_metrics.get('ci_level', 0.95))
                print(f"  [Eval@{episode + 1}] DR={dr_score:.6f}, IPS={ips_val:.3e}, SNIPS={snips_val:.3e}, DR_CI{int(ci_level * 100)}=[{dr_ci_low:.6f}, {dr_ci_high:.6f}], DM={dm_val:.6f}, DM_CI{int(ci_level * 100)}=[{dm_ci_low:.6f}, {dm_ci_high:.6f}]")
                self.train_stats.setdefault('rl_eval_history', []).append({'episode': int(episode + 1), 'dr': float(dr_score), 'ips': float(ips_val), 'snips': float(snips_val), 'dm': float(dm_val), 'dr_ci_low': float(dr_ci_low), 'dr_ci_high': float(dr_ci_high), 'dm_ci_low': float(dm_ci_low), 'dm_ci_high': float(dm_ci_high), 'ci_level': float(ci_level)})
                if dr_score > best_dr:
                    best_dr = dr_score
                    self.save_checkpoint('best_rl_model.pt')
                    print(f"  [OK] New best RL checkpoint saved (DR={best_dr:.6f})")
            if rl_cfg.get('use_full_dataset', True) and len(seen_conversations) >= target_conversations and not coverage_announced:
                if data_passes > 1:
                    print(f"Covered all configured RL conversations ({target_conversations}) once; continuing remaining passes (total episodes={num_episodes}).")
                else:
                    print(f"Covered all configured RL conversations ({target_conversations}).")
                    break
                coverage_announced = True
        env.close()
        print("\n[OK] RL fine-tuning completed!")
        summary = {'success_rate': float(np.mean(online_success_hist)) if online_success_hist else 0.0, 'avg_turns': float(np.mean(online_turns_hist)) if online_turns_hist else 0.0, 'diversity': float(np.mean(online_diversity_hist)) if online_diversity_hist else 0.0, 'fairness': float(np.mean(online_fairness_hist)) if online_fairness_hist else 0.0, 'num_episodes': float(len(online_success_hist))}
        print("Online simulator metrics:")
        print(f"  success_rate: {summary['success_rate']:.6f}")
        print(f"  avg_turns: {summary['avg_turns']:.3f}")
        print(f"  diversity: {summary['diversity']:.6f}")
        print(f"  fairness: {summary['fairness']:.6f}")
        print(f"  num_episodes: {int(summary['num_episodes'])}")
        return summary
    def _build_bc_batch(self, batch_convs: List[Dict]):
        """Create states and action labels for behavioral cloning from conversation turns."""
        states = []
        labels = []
        state_dim = self.config['model']['policy']['state_dim']
        for conv in batch_convs:
            turns = conv.get('turns', []) if isinstance(conv, dict) else []
            profile = conv.get('user_profile', {}) if isinstance(conv, dict) else {}
            turn_count = max(1, len(turns))
            accepted = any(bool(t.get('accepted', False)) for t in turns if isinstance(t, dict))
            asks = sum(1 for t in turns if isinstance(t, dict) and 'preference' in str(t.get('intent', '')).lower())
            recs = sum(1 for t in turns if isinstance(t, dict) and 'recommend' in str(t.get('intent', '')).lower())
            vec = np.zeros(state_dim, dtype=np.float32)
            vec[0] = min(turn_count, 20) / 20.0
            vec[1] = min(asks, 10) / 10.0
            vec[2] = min(recs, 10) / 10.0
            vec[3] = float(accepted)
            age_map = {'18-25': 0.2, '26-35': 0.4, '26-40': 0.4, '36-45': 0.6, '40+': 0.8, '46-55': 0.8, '55+': 1.0}
            vec[4] = age_map.get(profile.get('age_group', 'unknown'), 0.5)
            vec[5] = 0.6 if profile.get('gender', 'U') == 'F' else 0.2 if profile.get('gender', 'U') == 'M' else 0.5
            # Intent-to-action mapping into the first 4 policy actions.
            # 0=ask_preference, 1=recommend, 2=clarify, 3=end
            if accepted:
                label = 1
            elif recs == 0:
                label = 0
            elif turn_count >= self.config['environment']['max_turns'] - 1:
                label = 3
            else:
                label = 2 if asks > recs else 1
            states.append(vec)
            labels.append(label)
        return np.array(states), np.array(labels)
    def _load_rl_conversations(self) -> List[Dict]:
        """Load RL training conversations from configured train file."""
        data_cfg = self.config.get('data', {})
        train_file = data_cfg.get('train_file')
        if not train_file:
            train_file = os.path.join(data_cfg.get('data_dir', './data'), 'train_data.json')
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"RL train file not found: {train_file}")
        with open(train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ['dialogues', 'conversations', 'data']:
                if key in data and isinstance(data[key], list):
                    return data[key]
        raise ValueError(f"Unsupported conversation format in {train_file}")
    def _conversation_to_user_profile(self, conversation: Dict, conv_idx: int) -> Dict:
        """Create a user profile for simulator reset from conversation metadata."""
        profile = conversation.get('user_profile', {}) if isinstance(conversation, dict) else {}
        turns = conversation.get('turns', []) if isinstance(conversation, dict) else []
        mentioned_items = []
        accepted_items = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            for key in ['mentioned_items', 'items_mentioned', 'recommended_items']:
                values = turn.get(key, [])
                if isinstance(values, list):
                    mentioned_items.extend([str(v) for v in values])
            if bool(turn.get('accepted', False)):
                accepted_items.extend([str(v) for v in turn.get('mentioned_items', [])])
        preferences = []
        if isinstance(profile.get('preferences', []), list):
            preferences.extend([str(v) for v in profile.get('preferences', [])])
        preferences.extend(accepted_items)
        preferences.extend(mentioned_items)
        # Keep order while removing duplicates.
        seen = set()
        deduped_preferences = []
        for item_id in preferences:
            if item_id not in seen:
                seen.add(item_id)
                deduped_preferences.append(item_id)
        if not deduped_preferences:
            deduped_preferences = self.item_catalog.sample_items(5)
        return {
            'user_id': profile.get('user_id', f'user_rl_{conv_idx}'),
            'age_group': profile.get('age_group', '26-35'),
            'gender': profile.get('gender', 'U'),
            'preferences': deduped_preferences[:20]
        }
    def _sample_candidate_items(self, num_envs: int, candidate_pool_size: int, user_profiles: Optional[List[Dict]] = None):
        """Sample candidate items with a contextual mix that explicitly controls head concentration."""
        sample_size = min(int(candidate_pool_size), int(self._catalog_size))
        candidate_ids_batch = []
        head_list = list(self._head_item_set)
        tail_list = list(self._tail_item_set)
        mid_list = list(self._mid_item_set) if self._mid_item_set else list(set(self._catalog_item_ids) - self._head_item_set)
        for env_idx in range(num_envs):
            profile = user_profiles[env_idx] if user_profiles is not None and env_idx < len(user_profiles) else {}
            seen = set()
            selected = []

            def add_item(item_id: str):
                item_id = str(item_id)
                if item_id in self.item_catalog.catalog and item_id not in seen and len(selected) < sample_size:
                    seen.add(item_id)
                    selected.append(item_id)

            prefs = [str(x) for x in profile.get('preferences', []) if str(x) in self.item_catalog.catalog]
            for item_id in prefs[:max(4, sample_size // 10)]:
                add_item(item_id)

            pref_categories = set()
            for item_id in prefs[:12]:
                item = self.item_catalog.get_item(item_id) or {}
                pref_categories.add(str(item.get('category', 'Unknown')))

            category_candidates = []
            for cat in pref_categories:
                category_candidates.extend(self.item_catalog.get_items_by_category(cat, limit=max(12, sample_size // 2)))
            category_candidates = [str(x) for x in category_candidates if str(x) in self.item_catalog.catalog]
            category_candidates = sorted(set(category_candidates), key=lambda iid: (iid in self._head_item_set, self._item_popularity(iid), iid))

            # Keep category matches, but prefer non-head items first.
            for item_id in category_candidates[:max(12, sample_size // 3)]:
                add_item(item_id)

            # Explicit stratified quotas: ~50% tail, ~30% mid, ~20% head.
            remaining = sample_size - len(selected)
            tail_quota = max(10, int(0.50 * remaining))
            mid_quota = max(6, int(0.30 * remaining))
            head_quota = max(2, remaining - tail_quota - mid_quota)

            random.shuffle(tail_list)
            random.shuffle(mid_list)
            random.shuffle(head_list)

            for item_id in tail_list:
                add_item(item_id)
                if sum(1 for x in selected if x in self._tail_item_set) >= tail_quota:
                    break

            for item_id in mid_list:
                add_item(item_id)
                if sum(1 for x in selected if x in self._mid_item_set) >= mid_quota:
                    break

            # Keep some head items for accuracy, but cap them.
            for item_id in head_list:
                add_item(item_id)
                if sum(1 for x in selected if x in self._head_item_set) >= head_quota:
                    break

            # Final fill prefers tail-weighted samples, not uniform sampling.
            if self._tail_sampling_probs is not None and self._catalog_size > 0 and len(selected) < sample_size:
                draw_size = min(self._catalog_size, sample_size * 4)
                tail_indices = np.random.choice(self._catalog_size, size=draw_size, replace=False, p=self._tail_sampling_probs)
                for idx in tail_indices:
                    add_item(self._catalog_item_ids[int(idx)])
                    if len(selected) >= sample_size:
                        break

            while len(selected) < sample_size:
                add_item(random.choice(tail_list if tail_list else self._catalog_item_ids))

            candidate_ids_batch.append(selected[:sample_size])

        if self._fast_candidate_sampling and self._catalog_embedding_matrix is not None:
            id_to_idx = {item_id: idx for idx, item_id in enumerate(self._catalog_item_ids)}
            row_indices = [[id_to_idx[item_id] for item_id in ids] for ids in candidate_ids_batch]
            sampled_idx_t = torch.as_tensor(np.asarray(row_indices, dtype=np.int64), dtype=torch.long, device=self.device)
            candidate_embeddings = self._catalog_embedding_matrix[sampled_idx_t]
        else:
            candidate_embeddings = []
            for ids in candidate_ids_batch:
                embs = [self.item_catalog.get_item_embedding(item_id) for item_id in ids]
                candidate_embeddings.append(embs)
            candidate_embeddings = torch.FloatTensor(np.asarray(candidate_embeddings, dtype=np.float32)).to(self.device)

        return candidate_ids_batch, candidate_embeddings
    def _create_rl_batch(self, states: torch.Tensor, candidate_embeddings: torch.Tensor,
                         candidate_ids_batch: List[List[str]] = None,
                         user_profiles: Optional[List[Dict]] = None,
                         is_cold_start: bool = False,
                         use_thompson_sampling: bool = None,
                         generate_explanations: bool = False) -> Dict:
        """Create model batch for RL rollout with candidate item embeddings and profile-grounded text."""
        batch_size = states.shape[0]
        static_dim = self.config['model']['personalization']['static_features_dim']
        static_features = torch.zeros(batch_size, static_dim, device=self.device)
        copy_dim = min(states.shape[1], static_dim)
        static_features[:, :copy_dim] = states[:, :copy_dim]
        if candidate_ids_batch is None:
            candidate_ids_batch = [self.item_catalog.sample_items(candidate_embeddings.shape[1]) for _ in range(batch_size)]
        if user_profiles is None:
            user_profiles = [{} for _ in range(batch_size)]
        utterances = [self._build_policy_utterance(user_profiles[i] if i < len(user_profiles) else {}) for i in range(batch_size)]
        candidate_item_names, candidate_item_metadata = self._build_candidate_annotations(candidate_ids_batch)
        return {
            'utterances': utterances,
            'dialogue_history': None,
            'static_features': static_features,
            'candidate_items': candidate_embeddings,
            'candidate_item_ids': candidate_ids_batch,
            'candidate_item_names': candidate_item_names,
            'candidate_item_metadata': candidate_item_metadata,
            'generate_explanations': bool(generate_explanations),
            'is_cold_start': torch.full((batch_size,), bool(is_cold_start), dtype=torch.bool, device=self.device),
            'use_thompson_sampling': bool(use_thompson_sampling) if use_thompson_sampling is not None else None,
            'user_demographics': [
                {
                    'age_group': (user_profiles[i] or {}).get('age_group', '26-35'),
                    'gender': (user_profiles[i] or {}).get('gender', 'U')
                }
                for i in range(batch_size)
            ]
        }
    def _build_env_actions(self, actions: np.ndarray, candidate_ids_batch: List[List[str]],
                           reranked_indices: torch.Tensor, top_k: int = 10,
                           current_turn: int = None,
                           min_turns_before_end: int = 0,
                           recommended_counts: List[int] = None,
                           min_recommendations_before_end: int = 0,
                           fallback_action_type: int = 0,
                           ask_before_recommend_turns: int = 0) -> List[Dict]:
        """Convert model outputs into environment actions with top-k reranked recommendations."""
        env_actions = []
        for i, action in enumerate(actions):
            action_type = int(np.clip(action, 0, 3))
            rec_count_i = int(recommended_counts[i]) if isinstance(recommended_counts, list) and i < len(recommended_counts) else 0
            if (ask_before_recommend_turns > 0 and action_type == 1 and rec_count_i == 0 and current_turn is not None and int(current_turn) <= int(ask_before_recommend_turns)):
                action_type = 0
            if action_type == 3 and (((min_turns_before_end > 0 and current_turn is not None and int(current_turn) < int(min_turns_before_end))) or ((min_recommendations_before_end > 0 and rec_count_i < int(min_recommendations_before_end)))):
                action_type = int(np.clip(fallback_action_type, 0, 3))
            recommended_items = []
            if action_type == 1:
                if reranked_indices is not None:
                    ranked_pos = reranked_indices[i, :top_k].detach().cpu().tolist()
                    recommended_items = [candidate_ids_batch[i][pos] for pos in ranked_pos if pos < len(candidate_ids_batch[i])]
                else:
                    recommended_items = candidate_ids_batch[i][:top_k]
            env_actions.append({'action_type': action_type, 'items': recommended_items})
        return env_actions
    def _update_policy_from_rollout(self, traj_states: List[torch.Tensor], traj_actions: List[torch.Tensor],
                                    traj_log_probs: List[torch.Tensor], traj_values: List[torch.Tensor],
                                    traj_rewards: List[torch.Tensor], traj_dones: List[torch.Tensor]) -> Dict:
        """Update PPO policy from one rollout with GAE returns/advantages."""
        states_t = torch.stack(traj_states).to(self.device)
        actions_t = torch.stack(traj_actions).to(self.device)
        log_probs_t = torch.stack(traj_log_probs).to(self.device)
        values_t = torch.stack(traj_values).to(self.device)
        rewards_t = torch.stack(traj_rewards).to(self.device)
        dones_t = torch.stack(traj_dones).to(self.device)
        rewards_seq = rewards_t.permute(1, 0, 2)
        values_seq = values_t.permute(1, 0, 2)
        dones_seq = dones_t.permute(1, 0)
        advantages, returns = self.ppo_agent.compute_gae(rewards_seq, values_seq, dones_seq)
        if self.config['training'].get('rl', {}).get('normalize_advantages', True):
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std
        states_flat = states_t.permute(1, 0, 2).reshape(-1, states_t.shape[-1])
        actions_flat = actions_t.permute(1, 0).reshape(-1).long()
        log_probs_flat = log_probs_t.permute(1, 0).reshape(-1)
        returns_flat = returns.reshape(-1, returns.shape[-1])
        advantages_flat = advantages.reshape(-1, advantages.shape[-1])
        env_cfg = self.config.get('environment', {})
        div_scale = float(env_cfg.get('reward_diversity_factor', 1.0))
        fair_scale = float(env_cfg.get('reward_fairness_factor', 1.0))
        diversity_signal = rewards_seq[:, :, 1] / max(div_scale, 1e-6)
        fairness_signal = rewards_seq[:, :, 2] / max(fair_scale, 1e-6)
        df_cfg = self.config.get('model', {}).get('diversity_fairness', {})
        min_div = float(df_cfg.get('min_diversity', 0.3))
        min_fair = float(df_cfg.get('min_fairness', 0.7))
        div_penalty = torch.relu(min_div - diversity_signal)
        fair_penalty = torch.relu(min_fair - fairness_signal)
        constraint_penalty_flat = (div_penalty + fair_penalty).reshape(-1)
        rl_cfg = self.config['training'].get('rl', {})
        mb_size = int(min(rl_cfg.get('ppo_minibatch_size', 256), states_flat.shape[0]))
        updates = int(max(1, rl_cfg.get('updates_per_rollout', 4)))
        stats = []
        for _ in range(updates):
            indices = torch.randperm(states_flat.shape[0], device=self.device)[:mb_size]
            stat = self.ppo_agent.update(states_flat[indices], actions_flat[indices], log_probs_flat[indices], returns_flat[indices], advantages_flat[indices], constraint_penalty=constraint_penalty_flat[indices])
            stats.append(stat)
        return {key: float(np.mean([s[key] for s in stats])) for key in stats[0].keys()}
    def _update_policy(self, buffer: 'ExperienceBuffer') -> Dict:
        """Update policy using PPO"""
        # Sample batch
        batch = buffer.sample(self.config['training']['batch_size'])
        
        # Prepare data
        states = torch.FloatTensor(np.array([exp['state'] for exp in batch])).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        old_log_probs = torch.FloatTensor([exp['log_prob'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor(np.array([exp['reward'] for exp in batch])).to(self.device)
        values = torch.FloatTensor(np.array([exp['value'] for exp in batch])).to(self.device)
        
        # Compute returns and advantages
        returns = rewards  # Simplified, should compute proper returns
        advantages = returns - values
        
        # PPO update
        update_stats = self.ppo_agent.update(
            states, actions, old_log_probs, returns, advantages
        )
        
        return update_stats
    
    def _create_batch_from_state(self, states: torch.Tensor) -> Dict:
        """Create model batch from environment states"""
        batch_size = states.shape[0]
        static_dim = self.config['model']['personalization']['static_features_dim']
        static_features = torch.zeros(batch_size, static_dim, device=self.device)
        # Use available environment state as weak profile signal instead of random features.
        copy_dim = min(states.shape[1], static_dim)
        static_features[:, :copy_dim] = states[:, :copy_dim]
        batch = {
            'utterances': ['Current utterance'] * batch_size,
            'dialogue_history': None,
            'static_features': static_features,
            'candidate_items': None
        }
        
        return batch
    
    def _batch_to_device(self, batch) -> Dict:
        """Preprocess and move batch to device."""
        from data_utils import ConversationBatch
        
        if isinstance(batch, ConversationBatch):
            conversations = batch.conversations
        else:
            conversations = batch.get('conversations', batch) if isinstance(batch, dict) else [batch]
        
        # Extract utterances from conversations
        utterances = []
        intents = []
        static_features_list = []
        sentiment_labels = []
        preference_labels = []
        
        intent_map = {'general': 0, 'recommend': 1, 'provide_preference': 2, 'ask_question': 3, 'provide_information': 4}
        
        for conv in conversations:
            turns = conv.get('turns', [])
            if turns:
                # Get the last utterance
                last_turn = turns[-1]
                utterance = last_turn.get('user_utterance', '')
                utterances.append(utterance)
                
                # Get intent
                intent = last_turn.get('intent', 'general')
                intents.append(intent_map.get(intent, 0))
                sentiment_labels.append(self._infer_sentiment_label(utterance))
            else:
                utterances.append('')
                intents.append(0)
                sentiment_labels.append(1)
            profile = conv.get('user_profile', {}) if isinstance(conv, dict) else {}
            static_features_list.append(
                self._build_static_features_from_profile(
                    profile,
                    self.config['model']['personalization']['static_features_dim']
                )
            )
            preference_labels.append(self._extract_preference_signal(conv))
        
        # Create batch dictionary
        batch_data = {
            'conversations': conversations,
            'batch_size': len(conversations),
            'utterances': utterances,  # Required by forward()
        }
        
        static_features = torch.as_tensor(np.asarray(static_features_list, dtype=np.float32), device=self.device)
        batch_data['static_features'] = static_features
        
        # Create intent labels for classification
        if intents:
            batch_data['intent_labels'] = torch.tensor(intents, dtype=torch.long).to(self.device)
        
        batch_data['sentiment_labels'] = torch.tensor(sentiment_labels, dtype=torch.long).to(self.device)
        
        pref_scores = torch.as_tensor(np.asarray(preference_labels, dtype=np.float32), device=self.device).unsqueeze(-1)
        batch_data['preference_labels'] = pref_scores
        
        # Create dialogue history (optional, but some methods might use it)
        # For batch processing, we don't use per-conversation history
        # The DST will handle utterances without history for simplicity
        
        batch_data['dialogue_history'] = None  # Use None for batch training
        
        return batch_data
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics to console and wandb"""
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['total_loss']:.4f}")
        
        if 'intent_accuracy' in val_metrics:
            print(f"  Val Intent Acc: {val_metrics['intent_accuracy']:.4f}")
        if 'sentiment_accuracy' in val_metrics:
            print(f"  Val Sentiment Acc: {val_metrics['sentiment_accuracy']:.4f}")
        
        if self.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()}
            })
        self.train_stats.setdefault('supervised_history', []).append({
            'epoch': int(epoch + 1),
            'train_total_loss': float(train_metrics.get('total_loss', 0.0)),
            'train_intent_loss': float(train_metrics.get('intent_loss', 0.0)),
            'train_sentiment_loss': float(train_metrics.get('sentiment_loss', 0.0)),
            'train_preference_loss': float(train_metrics.get('preference_loss', 0.0)),
            'val_total_loss': float(val_metrics.get('total_loss', 0.0)),
            'val_intent_accuracy': float(val_metrics.get('intent_accuracy', 0.0)),
            'val_sentiment_accuracy': float(val_metrics.get('sentiment_accuracy', 0.0)),
        })
    def _get_dataset_checkpoint_dir(self) -> str:
        """Return dataset-specific checkpoint directory under logging.save_dir."""
        checkpoint_root = self.config['logging'].get('save_dir', './checkpoints')
        dataset_name = str(self.config.get('data', {}).get('dataset_name', 'default')).strip() or 'default'
        checkpoint_dir = os.path.join(checkpoint_root, dataset_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        return checkpoint_dir
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_dir = self._get_dataset_checkpoint_dir()
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_stats': self.train_stats,
            'config': self.config
        }, checkpoint_path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        if os.path.isabs(filename) or os.path.exists(filename):
            checkpoint_path = filename
        else:
            dataset_dir = self._get_dataset_checkpoint_dir()
            dataset_checkpoint = os.path.join(dataset_dir, filename)
            legacy_checkpoint = os.path.join(self.config['logging'].get('save_dir', './checkpoints'), filename)
            if os.path.exists(dataset_checkpoint):
                checkpoint_path = dataset_checkpoint
            elif os.path.exists(legacy_checkpoint):
                checkpoint_path = legacy_checkpoint
            else:
                checkpoint_path = dataset_checkpoint
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'train_stats' in checkpoint:
            self.train_stats = checkpoint['train_stats']
        self.train_stats.setdefault('supervised_history', [])
        self.train_stats.setdefault('rl_history', [])
        self.train_stats.setdefault('rl_eval_history', [])
        print(f"Loaded checkpoint: {checkpoint_path}")
class ExperienceBuffer:
    """
    Experience replay buffer for RL training
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, experience: Dict):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample batch from buffer"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)
def main():
    parser = argparse.ArgumentParser(description='Train MO-CRS')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, default='pretrain',
                       choices=['pretrain', 'rl', 'both', 'test'],
                       help='Training mode')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Load checkpoint path, or checkpoint filename from dataset checkpoint folder')
    parser.add_argument('--refresh_splits', action='store_true',
                       help='Regenerate stronger train/val/test splits before training')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset to use (e.g., ReDial, GoRecDial, INSPIRED, MovieLens_1M, Yelp, DuRecDial, LastFM, OpenDialKG)')
    parser.add_argument('--eval_output', type=str, default=None,
                       help='Optional path to save evaluation metrics as JSON')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Resolve data paths relative to config file location
    config_dir = os.path.dirname(os.path.abspath(args.config))
    for key in ['catalog_file', 'train_file', 'val_file', 'test_file', 'full_data_file', 'data_dir', 'processed_dir', 'data_root']:
        if key in config.get('data', {}):
            path = config['data'][key]
            if isinstance(path, str) and path and not os.path.isabs(path):
                config['data'][key] = os.path.normpath(os.path.join(config_dir, path))
    for key in ['save_dir', 'log_dir']:
        if key in config.get('logging', {}):
            path = config['logging'][key]
            if isinstance(path, str) and path and not os.path.isabs(path):
                config['logging'][key] = os.path.normpath(os.path.join(config_dir, path))
    if args.eval_output and not os.path.isabs(args.eval_output):
        args.eval_output = os.path.normpath(os.path.join(config_dir, args.eval_output))
    apply_dataset_paths(config, args.dataset)
    print(f"Using dataset: {config['data']['dataset_name']}")
    print(f"  data_dir: {config['data']['data_dir']}")
    print(f"  train_file: {config['data']['train_file']}")
    print(f"  val_file: {config['data']['val_file']}")
    print(f"  test_file: {config['data']['test_file']}")
    stronger_val_enabled = bool(config.get('data', {}).get('stronger_validation', {}).get('enabled', False))
    if args.refresh_splits and stronger_val_enabled:
        split_stats = create_stronger_validation_splits(config)
        print(f"Regenerated splits: train={split_stats['train']}, val={split_stats['val']}, test={split_stats['test']}")
    elif args.refresh_splits and not stronger_val_enabled:
        print("[INFO] --refresh_splits ignored because data.stronger_validation.enabled is false; using fixed existing splits.")
    
    # Create trainer
    trainer = MOCRSTrainer(config, use_wandb=args.wandb)
    eval_report = {
        'mode': args.mode,
        'dataset': config.get('data', {}).get('dataset_name'),
        'config': os.path.abspath(args.config),
        'checkpoint': args.checkpoint,
        'ope': {}
    }
    def _save_eval_report() -> None:
        if not args.eval_output:
            return
        out_dir = os.path.dirname(args.eval_output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.eval_output, 'w', encoding='utf-8') as f:
            json.dump(eval_report, f, indent=2, ensure_ascii=True)
        print(f"Saved evaluation report to: {args.eval_output}")
    def _print_ope(split_name: str, split_file: str):
        if not split_file or not os.path.exists(split_file):
            print(f"Skipping {split_name} OPE: file not found")
            return None
        ope_metrics = off_policy_evaluate(trainer.model, split_file, trainer.item_catalog, config, trainer.device)
        print(f"Off-policy evaluation ({split_name}):")
        ci_level = float(ope_metrics.get('ci_level', 0.95))
        print(f"  ips: {float(ope_metrics.get('ips', 0.0)):.3e}")
        print(f"  snips: {float(ope_metrics.get('snips', 0.0)):.3e}")
        print(f"  dr: {float(ope_metrics.get('dr', 0.0)):.6f}")
        print(
            f"  dr_ci_{int(ci_level * 100)}: "
            f"[{float(ope_metrics.get('dr_ci_low', ope_metrics.get('dr', 0.0))):.6f}, "
            f"{float(ope_metrics.get('dr_ci_high', ope_metrics.get('dr', 0.0))):.6f}]"
        )
        print(f"  dm: {float(ope_metrics.get('dm', 0.0)):.6f}")
        print(
            f"  dm_ci_{int(ci_level * 100)}: "
            f"[{float(ope_metrics.get('dm_ci_low', ope_metrics.get('dm', 0.0))):.6f}, "
            f"{float(ope_metrics.get('dm_ci_high', ope_metrics.get('dm', 0.0))):.6f}]"
        )
        print(f"  logged_reward_mean: {float(ope_metrics.get('logged_reward_mean', 0.0)):.6f}")
        print(f"  logged_reward_std: {float(ope_metrics.get('logged_reward_std', 0.0)):.6f}")
        print(f"  behavior_recommend_rate: {float(ope_metrics.get('behavior_recommend_rate', 0.0)):.6f}")
        print(f"  num_samples: {float(ope_metrics.get('num_samples', 0.0)):.0f}")
        print(f"  bootstrap_samples: {int(float(ope_metrics.get('bootstrap_samples', 0.0)))}")
        return ope_metrics
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    if args.mode == 'test':
        if not args.checkpoint:
            raise ValueError("--mode test requires --checkpoint")
        test_file = config.get('data', {}).get('test_file')
        test_metrics = _print_ope('test', test_file)
        if test_metrics is not None:
            eval_report['ope']['test'] = test_metrics
        full_test_eval = evaluate_full_test_suite(
            trainer=trainer,
            config=config,
            episodes=int(config.get('evaluation', {}).get('num_eval_episodes', 80)),
            fairness_k_values=[5, 10, 20],
        )
        eval_report['test_evaluation'] = full_test_eval
        rec = full_test_eval.get('recommendation_results', {})
        conv = full_test_eval.get('conversation_results', {})
        div = full_test_eval.get('diversity', {})
        fair = full_test_eval.get('fairness', {})
        transp = full_test_eval.get('transparency', {})
        print("Full test evaluation metrics:")
        print(
            f"  Rec: R@10={float(rec.get('Recall@10', 0.0)):.4f}, "
            f"R@50={float(rec.get('Recall@50', 0.0)):.4f}, "
            f"MRR@10={float(rec.get('MRR@10', 0.0)):.4f}, "
            f"MRR@50={float(rec.get('MRR@50', 0.0)):.4f}, "
            f"NDCG@10={float(rec.get('NDCG@10', 0.0)):.4f}, "
            f"NDCG@50={float(rec.get('NDCG@50', 0.0)):.4f}"
        )
        print(
            f"  Conv: Dist-2={float(conv.get('Dist-2', 0.0)):.4f}, Dist-3={float(conv.get('Dist-3', 0.0)):.4f}, "
            f"BLEU-2={float(conv.get('BLEU-2', 0.0)):.4f}, BLEU-3={float(conv.get('BLEU-3', 0.0)):.4f}, "
            f"SR@5={float(conv.get('SR@5', 0.0)):.4f}, SR@10={float(conv.get('SR@10', 0.0)):.4f}, "
            f"SR@20={float(conv.get('SR@20', 0.0)):.4f}, AT={float(conv.get('AT', 0.0)):.2f}"
        )
        print(
            f"  Diversity@10: ILD={float(div.get('ILD@10', 0.0)):.4f}, "
            f"GenreCov={float(div.get('GenreCoverage@10', 0.0)):.4f}, "
            f"CategoryCov={float(div.get('CategoryCoverage@10', 0.0)):.4f}, "
            f"CalErr={float(div.get('CalibrationError@10', 0.0)):.4f}"
        )
        print(
            f"  Fairness@10: A={float(fair.get('A@10', 0.0)):.2f}, "
            f"G={float(fair.get('G@10', 0.0)):.4f}, L={float(fair.get('L@10', 0.0)):.4f}, "
            f"D={float(fair.get('D@10', 0.0)):.4f}, Entropy={float(fair.get('Entropy@10', 0.0)):.4f}"
        )
        print(
            f"  Transparency: grounded={float(transp.get('groundedness_factual_consistency', 0.0)):.4f}, "
            f"halluc={float(transp.get('deception_hallucination_rate', 0.0)):.4f}, "
            f"persuasive={float(transp.get('persuasiveness_score', 0.0)):.4f}, "
            f"transparency={float(transp.get('transparency_score', 0.0)):.4f}, "
            f"trust={float(transp.get('trust_score', 0.0)):.4f}, "
            f"useful={float(transp.get('usefulness_score', 0.0)):.4f}"
        )
        _save_eval_report()
        print("\n" + "="*60)
        print("Test-Only Evaluation Complete!")
        print("="*60)
        return
    
    # Training
    if args.mode in ['pretrain', 'both']:
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader, test_loader = create_dataloaders(config)
        
        # Pre-training
        trainer.supervised_pretrain(
            train_loader,
            val_loader,
            num_epochs=config['training']['num_epochs']
        )
    
    if args.mode in ['rl', 'both']:
        bc_cfg = config.get('training', {}).get('behavioral_cloning', {})
        if bc_cfg.get('enabled', False):
            bc_conversations = trainer._load_rl_conversations()
            trainer.behavioral_cloning_warmstart(
                bc_conversations,
                epochs=int(bc_cfg.get('epochs', 5)),
                learning_rate=float(bc_cfg.get('learning_rate', config['training']['learning_rate']))
            )
        # RL fine-tuning
        online_metrics = trainer.rl_finetune(
            num_episodes=config['training']['num_episodes']
        )
        if online_metrics:
            print("Online simulator metrics (aggregate):")
            print(f"  success_rate: {float(online_metrics.get('success_rate', 0.0)):.6f}")
            print(f"  avg_turns: {float(online_metrics.get('avg_turns', 0.0)):.3f}")
            print(f"  diversity: {float(online_metrics.get('diversity', 0.0)):.6f}")
            print(f"  fairness: {float(online_metrics.get('fairness', 0.0)):.6f}")
            eval_report['online_simulator'] = online_metrics
        # Off-policy evaluation on held-out validation and test conversations.
        val_file = config.get('data', {}).get('val_file')
        test_file = config.get('data', {}).get('test_file')
        val_metrics = _print_ope('validation', val_file)
        test_metrics = _print_ope('test', test_file)
        if val_metrics is not None:
            eval_report['ope']['validation'] = val_metrics
        if test_metrics is not None:
            eval_report['ope']['test'] = test_metrics
    _save_eval_report()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
if __name__ == "__main__":
    main()
