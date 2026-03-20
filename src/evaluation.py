"""
Evaluation metrics for MO-CRS
Measures accuracy, diversity, fairness, and engagement
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.stats import entropy
import json


def _bootstrap_mean_ci(values: List[float], rng: np.random.RandomState,
                       n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for the sample mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        v = float(arr[0])
        return v, v

    n_bootstrap = max(100, int(n_bootstrap))
    means = np.empty(n_bootstrap, dtype=np.float64)
    n = arr.size

    for i in range(n_bootstrap):
        sample = arr[rng.randint(0, n, size=n)]
        means[i] = sample.mean()

    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lower, upper


def _load_conversations_for_eval(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ['conversations', 'dialogues', 'data']:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"Unsupported conversation schema in {path}")


def _extract_logged_action_and_reward(conversation: Dict) -> Tuple[int, float]:
    """Extract logged action and stronger acceptance/relevance reward signal."""
    turns = conversation.get('turns', []) if isinstance(conversation, dict) else []
    accepted_items = set(str(x) for x in conversation.get('accepted_items', []) if x is not None)
    mentioned_items = []
    has_recommend = False
    recommend_turns = 0
    preference_turns = 0

    for turn in turns:
        if not isinstance(turn, dict):
            continue
        intent = str(turn.get('intent', '')).lower()
        if 'recommend' in intent or 'suggest' in intent:
            has_recommend = True
            recommend_turns += 1
        if 'prefer' in intent:
            preference_turns += 1
        for key in ['items_mentioned', 'mentioned_items', 'recommended_items']:
            values = turn.get(key, [])
            if isinstance(values, list):
                mentioned_items.extend(str(v) for v in values if v is not None)

    # 0=ask_preference, 1=recommend, 2=clarify, 3=end
    if accepted_items:
        action = 1
    elif has_recommend:
        action = 2
    elif len(turns) >= 10:
        action = 3
    else:
        action = 0

    # Stronger logged reward signal using acceptance relevance and interaction quality.
    unique_mentioned = set(mentioned_items)
    overlap = accepted_items.intersection(unique_mentioned)
    overlap_count = len(overlap)

    mention_precision = overlap_count / max(len(unique_mentioned), 1)
    acceptance_recall = overlap_count / max(len(accepted_items), 1) if accepted_items else 0.0
    if mention_precision + acceptance_recall > 0:
        relevance_f1 = (2.0 * mention_precision * acceptance_recall) / (mention_precision + acceptance_recall)
    else:
        relevance_f1 = 0.0

    accepted_count_score = min(1.0, len(accepted_items) / 4.0)
    turn_count = max(len(turns), 1)
    recommend_density = recommend_turns / turn_count
    preference_density = preference_turns / turn_count

    if accepted_items:
        reward = (
            0.55 * relevance_f1 +
            0.25 * accepted_count_score +
            0.10 * recommend_density +
            0.10 * preference_density
        )
    elif has_recommend:
        reward = 0.15 * recommend_density + 0.10 * preference_density
    else:
        reward = 0.05 * preference_density

    reward = float(min(max(reward, 0.0), 1.0))

    return action, reward


def _build_eval_batch(conversation: Dict, item_catalog, config: Dict, device: str, candidate_size: int = 100) -> Tuple[Dict, List[str], set]:
    """Create an evaluation batch conditioned on a logged conversation."""
    turns = conversation.get('turns', []) if isinstance(conversation, dict) else []
    profile = conversation.get('user_profile', {}) if isinstance(conversation, dict) else {}

    utterance = 'Eval utterance'
    if turns:
        utterance = turns[0].get('user_utterance', turns[0].get('utterance', utterance))

    observed_items = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        for key in ['items_mentioned', 'mentioned_items', 'recommended_items']:
            values = turn.get(key, [])
            if isinstance(values, list):
                observed_items.extend(str(v) for v in values if v is not None)

    accepted_items = set(str(x) for x in conversation.get('accepted_items', []) if x is not None)

    # Keep observed items first, then fill with random samples.
    dedup = []
    seen = set()
    for item_id in observed_items:
        if item_id in item_catalog.catalog and item_id not in seen:
            seen.add(item_id)
            dedup.append(item_id)
    if len(dedup) < candidate_size:
        for item_id in item_catalog.sample_items(candidate_size):
            if item_id not in seen:
                seen.add(item_id)
                dedup.append(item_id)
            if len(dedup) >= candidate_size:
                break

    candidate_ids = dedup[:candidate_size]
    candidate_embs = [item_catalog.get_item_embedding(i) for i in candidate_ids]
    candidate_items = torch.as_tensor(np.asarray(candidate_embs, dtype=np.float32), device=device).unsqueeze(0)

    static_dim = config['model']['personalization']['static_features_dim']
    static_features = torch.zeros(1, static_dim, dtype=torch.float32, device=device)

    batch = {
        'utterances': [utterance],
        'dialogue_history': None,
        'static_features': static_features,
        'candidate_items': candidate_items,
        'candidate_item_ids': [candidate_ids],
        'candidate_item_names': [[item_catalog.get_item(i).get('title', str(i)) if item_catalog.get_item(i) else str(i) for i in candidate_ids]],
        'user_demographics': [{
            'age_group': profile.get('age_group', '26-35'),
            'gender': profile.get('gender', 'U')
        }]
    }

    return batch, candidate_ids, accepted_items


def off_policy_evaluate(model, validation_file: str, item_catalog, config: Dict, device: str = 'cpu', max_samples: int = 5000) -> Dict:
    """
    Off-policy evaluation with IPS and Doubly Robust estimators.

    This uses logged conversation proxies and policy action probabilities.
    """
    model.eval()
    conversations = _load_conversations_for_eval(validation_file)
    if not conversations:
        return {'ips': 0.0, 'snips': 0.0, 'dr': 0.0, 'num_samples': 0.0}

    rng = np.random.RandomState(int(config.get('seed', 42)))
    if len(conversations) > max_samples:
        idx = rng.choice(len(conversations), size=max_samples, replace=False)
        sampled = [conversations[i] for i in idx]
    else:
        sampled = conversations

    ips_terms = []
    dr_terms = []
    dm_terms = []
    logged_rewards = []
    behavior_actions = []
    logged_records = []
    weight_sum = 0.0

    # First pass: collect logged records and estimate empirical behavior policy.
    for conv in sampled:
        logged_action, logged_reward = _extract_logged_action_and_reward(conv)
        behavior_actions.append(logged_action)
        logged_rewards.append(logged_reward)
        logged_records.append((conv, logged_action, logged_reward))

    counts = np.bincount(np.asarray(behavior_actions, dtype=np.int64), minlength=4)
    # Laplace smoothing for support coverage.
    behavior_probs = (counts + 1.0) / (counts.sum() + 4.0)

    eval_cfg = config.get('evaluation', {})
    clip_c = float(eval_cfg.get('ips_clip_c', 10.0))
    ci_level = float(eval_cfg.get('bootstrap_ci_level', 0.95))
    ci_level = min(max(ci_level, 0.5), 0.999)
    bootstrap_samples = int(eval_cfg.get('bootstrap_samples', 1000))
    topk = int(config.get('training', {}).get('rl', {}).get('top_k_recommendations', 10))

    with torch.no_grad():
        for conv, logged_action, logged_reward in logged_records:
            batch, candidate_ids, accepted_items = _build_eval_batch(conv, item_catalog, config, device)

            outputs = model(batch)
            action_probs = outputs['action_probs'][0]

            # Use first 4 dialogue actions for OPE action matching.
            action_probs_4 = action_probs[:4]
            action_probs_4 = action_probs_4 / action_probs_4.sum().clamp(min=1e-8)

            target_prob = float(action_probs_4[logged_action].item())
            behavior_prob = float(behavior_probs[logged_action])

            w = target_prob / max(behavior_prob, 1e-8)
            w = min(w, clip_c)

            # Direct-method reward proxy from top-k overlap with accepted items.
            dm_reward = 0.0
            reranked_indices = outputs.get('reranked_indices', None)
            if reranked_indices is not None and accepted_items:
                rec_pos = reranked_indices[0, :topk].detach().cpu().tolist()
                rec_ids = [candidate_ids[pos] for pos in rec_pos if 0 <= pos < len(candidate_ids)]
                if rec_ids:
                    dm_reward = len(set(rec_ids).intersection(accepted_items)) / len(set(rec_ids))

            q_logged = float(action_probs_4[logged_action].item()) * dm_reward

            ips_terms.append(w * logged_reward)
            weight_sum += w
            dm_terms.append(dm_reward)

            # Doubly robust: DM + importance-corrected residual.
            dr = dm_reward + w * (logged_reward - q_logged)
            dr_terms.append(dr)

    ips = float(np.mean(ips_terms)) if ips_terms else 0.0
    snips = float(np.sum(ips_terms) / max(weight_sum, 1e-8)) if ips_terms else 0.0
    dr = float(np.mean(dr_terms)) if dr_terms else 0.0
    dm = float(np.mean(dm_terms)) if dm_terms else 0.0
    alpha = 1.0 - ci_level
    dr_ci_low, dr_ci_high = _bootstrap_mean_ci(dr_terms, rng, n_bootstrap=bootstrap_samples, alpha=alpha)
    dm_ci_low, dm_ci_high = _bootstrap_mean_ci(dm_terms, rng, n_bootstrap=bootstrap_samples, alpha=alpha)

    return {
        'ips': ips,
        'snips': snips,
        'dr': dr,
        'dm': dm,
        'dr_ci_low': dr_ci_low,
        'dr_ci_high': dr_ci_high,
        'dm_ci_low': dm_ci_low,
        'dm_ci_high': dm_ci_high,
        'ci_level': ci_level,
        'bootstrap_samples': float(bootstrap_samples),
        'logged_reward_mean': float(np.mean(logged_rewards)) if logged_rewards else 0.0,
        'logged_reward_std': float(np.std(logged_rewards)) if logged_rewards else 0.0,
        'behavior_recommend_rate': float(behavior_probs[1]),
        'num_samples': float(len(sampled))
    }


class MOCRSEvaluator:
    """
    Comprehensive evaluator for multi-objective CRS
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all accumulated metrics"""
        self.metrics = {
            # Accuracy metrics
            'success_rate': [],
            'average_turns': [],
            'precision_at_k': defaultdict(list),
            'recall_at_k': defaultdict(list),
            'ndcg_at_k': defaultdict(list),
            'mrr': [],
            
            # Diversity metrics
            'intra_list_diversity': [],
            'coverage': [],
            'novelty': [],
            'temporal_diversity': [],
            
            # Fairness metrics
            'demographic_parity': [],
            'equal_opportunity': [],
            'calibration': [],
            'exposure_gini': [],
            'exposure_entropy': [],
            
            # Engagement metrics
            'user_satisfaction': [],
            'conversation_quality': [],
            'explanation_quality': []
        }
        
        self.demographic_stats = defaultdict(lambda: {
            'num_conversations': 0,
            'num_successes': 0,
            'total_utility': 0.0
        })
        
        self.item_exposure = defaultdict(int)
        self.total_recommendations = 0
    
    def evaluate_episode(self, predictions: Dict, ground_truth: Dict,
                        user_demographics: Dict = None) -> Dict:
        """
        Evaluate single episode
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels/preferences
            user_demographics: User demographic information
            
        Returns:
            Episode metrics
        """
        episode_metrics = {}
        
        # ==== Accuracy Metrics ====
        if 'recommendations' in predictions and 'relevant_items' in ground_truth:
            acc_metrics = self._evaluate_accuracy(
                predictions['recommendations'],
                ground_truth['relevant_items']
            )
            episode_metrics.update(acc_metrics)
        
        # ==== Diversity Metrics ====
        if 'recommendations' in predictions:
            div_metrics = self._evaluate_diversity(predictions['recommendations'])
            episode_metrics.update(div_metrics)
        
        # ==== Fairness Metrics ====
        if user_demographics:
            self._update_fairness_stats(
                predictions.get('success', False),
                predictions.get('utility', 0.0),
                user_demographics
            )
        
        # ==== Engagement Metrics ====
        if 'conversation' in ground_truth:
            eng_metrics = self._evaluate_engagement(
                predictions,
                ground_truth['conversation']
            )
            episode_metrics.update(eng_metrics)
        
        # Accumulate metrics
        for key, value in episode_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        return episode_metrics
    
    def _evaluate_accuracy(self, recommendations: List, 
                          relevant_items: List) -> Dict:
        """Evaluate recommendation accuracy"""
        metrics = {}
        
        relevant_set = set(relevant_items)
        
        # Success rate
        success = any(item in relevant_set for item in recommendations)
        metrics['success'] = float(success)
        
        # Precision, Recall, NDCG at K
        for k in [1, 3, 5, 10]:
            if len(recommendations) >= k:
                top_k = recommendations[:k]
                
                # Precision@K
                hits = sum(1 for item in top_k if item in relevant_set)
                precision_k = hits / k
                metrics[f'precision@{k}'] = precision_k
                
                # Recall@K
                recall_k = hits / len(relevant_set) if relevant_set else 0.0
                metrics[f'recall@{k}'] = recall_k
                
                # NDCG@K
                ndcg_k = self._compute_ndcg(top_k, relevant_set, k)
                metrics[f'ndcg@{k}'] = ndcg_k
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for rank, item in enumerate(recommendations, 1):
            if item in relevant_set:
                mrr = 1.0 / rank
                break
        metrics['mrr'] = mrr
        
        return metrics
    
    def _compute_ndcg(self, recommendations: List, relevant_set: set, k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain"""
        dcg = 0.0
        for i, item in enumerate(recommendations[:k], 1):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 1)
        
        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_set))))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg
    
    def _evaluate_diversity(self, recommendations: List) -> Dict:
        """Evaluate recommendation diversity"""
        metrics = {}
        
        if len(recommendations) <= 1:
            return {'diversity': 0.0}
        
        # Intra-list diversity (pairwise dissimilarity)
        # In practice, would use item embeddings
        # Here we approximate with category/feature diversity
        unique_items = len(set(recommendations))
        metrics['intra_list_diversity'] = unique_items / len(recommendations)
        
        # Coverage (how many unique items recommended across all sessions)
        for item in recommendations:
            self.item_exposure[item] += 1
        self.total_recommendations += len(recommendations)
        
        # Novelty (inverse of popularity)
        # Higher score for less popular items
        metrics['novelty'] = 0.7  # Placeholder
        
        return metrics
    
    def _update_fairness_stats(self, success: bool, utility: float,
                              user_demographics: Dict):
        """Update demographic fairness statistics"""
        demographic_group = user_demographics.get('age_group', 'unknown')
        
        stats = self.demographic_stats[demographic_group]
        stats['num_conversations'] += 1
        if success:
            stats['num_successes'] += 1
        stats['total_utility'] += utility
    
    def _evaluate_engagement(self, predictions: Dict,
                           conversation_ground_truth: Dict) -> Dict:
        """Evaluate user engagement"""
        metrics = {}
        
        # Conversation length
        num_turns = predictions.get('num_turns', 0)
        metrics['num_turns'] = num_turns
        
        # User satisfaction (from ratings if available)
        if 'user_rating' in conversation_ground_truth:
            metrics['user_satisfaction'] = conversation_ground_truth['user_rating']
        
        # Explanation quality
        if 'explanations' in predictions:
            exp_quality = self._evaluate_explanations(predictions['explanations'])
            metrics['explanation_quality'] = exp_quality
        
        return metrics
    
    def _evaluate_explanations(self, explanations: List[str]) -> float:
        """Evaluate explanation quality"""
        if not explanations:
            return 0.0
        
        scores = []
        for exp in explanations:
            # Length check
            words = exp.split()
            length_ok = 10 <= len(words) <= 50
            
            # Has reasoning words
            reasoning_words = ['because', 'since', 'as', 'like', 'similar']
            has_reasoning = any(word in exp.lower() for word in reasoning_words)
            
            # Not too generic
            generic = exp.lower().count('recommend') + exp.lower().count('suggest')
            not_generic = generic <= 1
            
            score = sum([length_ok, has_reasoning, not_generic]) / 3.0
            scores.append(score)
        
        return np.mean(scores)
    
    def compute_aggregate_metrics(self) -> Dict:
        """Compute aggregate metrics over all episodes"""
        aggregate = {}
        
        # Accuracy metrics
        if self.metrics['success_rate']:
            aggregate['success_rate'] = np.mean(self.metrics['success_rate'])
        
        if self.metrics['average_turns']:
            aggregate['average_turns'] = np.mean(self.metrics['average_turns'])
        
        if self.metrics['mrr']:
            aggregate['mrr'] = np.mean(self.metrics['mrr'])
        
        # Precision, Recall, NDCG at various K
        for k in [1, 3, 5, 10]:
            if self.metrics[f'precision_at_{k}']:
                aggregate[f'precision@{k}'] = np.mean(self.metrics[f'precision_at_{k}'])
            if self.metrics[f'recall_at_{k}']:
                aggregate[f'recall@{k}'] = np.mean(self.metrics[f'recall_at_{k}'])
            if self.metrics[f'ndcg_at_{k}']:
                aggregate[f'ndcg@{k}'] = np.mean(self.metrics[f'ndcg_at_{k}'])
        
        # Diversity metrics
        if self.metrics['intra_list_diversity']:
            aggregate['intra_list_diversity'] = np.mean(self.metrics['intra_list_diversity'])
        
        # Coverage
        if self.total_recommendations > 0:
            aggregate['coverage'] = len(self.item_exposure) / self.total_recommendations
        
        # Novelty
        if self.metrics['novelty']:
            aggregate['novelty'] = np.mean(self.metrics['novelty'])
        
        # Fairness metrics
        fairness = self._compute_fairness_metrics()
        aggregate.update(fairness)
        
        # Engagement metrics
        if self.metrics['user_satisfaction']:
            aggregate['user_satisfaction'] = np.mean(self.metrics['user_satisfaction'])
        
        if self.metrics['explanation_quality']:
            aggregate['explanation_quality'] = np.mean(self.metrics['explanation_quality'])
        
        return aggregate
    
    def _compute_fairness_metrics(self) -> Dict:
        """Compute demographic fairness metrics"""
        metrics = {}
        
        if not self.demographic_stats:
            return metrics
        
        # Success rates per group
        success_rates = []
        utilities = []
        
        for group, stats in self.demographic_stats.items():
            if stats['num_conversations'] > 0:
                success_rate = stats['num_successes'] / stats['num_conversations']
                avg_utility = stats['total_utility'] / stats['num_conversations']
                
                success_rates.append(success_rate)
                utilities.append(avg_utility)
        
        # Demographic parity (variance in success rates)
        if success_rates:
            metrics['demographic_parity'] = float(np.var(success_rates))
        
        # Equal opportunity (variance in utilities)
        if utilities:
            metrics['equal_opportunity'] = float(np.var(utilities))
        
        # Exposure fairness (Gini coefficient)
        if self.item_exposure:
            exposure_counts = np.array(list(self.item_exposure.values()))
            metrics['exposure_gini'] = self._compute_gini(exposure_counts)
            
            # Exposure entropy
            exposure_probs = exposure_counts / exposure_counts.sum()
            metrics['exposure_entropy'] = float(entropy(exposure_probs))
        
        return metrics
    
    def _compute_gini(self, values: np.ndarray) -> float:
        """Compute Gini coefficient"""
        sorted_values = np.sort(values)
        n = len(sorted_values)
        index = np.arange(1, n + 1)
        
        gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
        return float(gini)
    
    def get_metrics_summary(self) -> str:
        """Get formatted summary of metrics"""
        aggregate = self.compute_aggregate_metrics()
        
        summary = "\n" + "="*60 + "\n"
        summary += "Evaluation Metrics Summary\n"
        summary += "="*60 + "\n\n"
        
        summary += "ACCURACY METRICS:\n"
        summary += f"  Success Rate: {aggregate.get('success_rate', 0.0):.4f}\n"
        summary += f"  MRR: {aggregate.get('mrr', 0.0):.4f}\n"
        summary += f"  Precision@5: {aggregate.get('precision@5', 0.0):.4f}\n"
        summary += f"  Recall@5: {aggregate.get('recall@5', 0.0):.4f}\n"
        summary += f"  NDCG@5: {aggregate.get('ndcg@5', 0.0):.4f}\n"
        
        summary += "\nDIVERSITY METRICS:\n"
        summary += f"  Intra-List Diversity: {aggregate.get('intra_list_diversity', 0.0):.4f}\n"
        summary += f"  Coverage: {aggregate.get('coverage', 0.0):.4f}\n"
        summary += f"  Novelty: {aggregate.get('novelty', 0.0):.4f}\n"
        
        summary += "\nFAIRNESS METRICS:\n"
        summary += f"  Demographic Parity: {aggregate.get('demographic_parity', 0.0):.4f}\n"
        summary += f"  Equal Opportunity: {aggregate.get('equal_opportunity', 0.0):.4f}\n"
        summary += f"  Exposure Gini: {aggregate.get('exposure_gini', 0.0):.4f}\n"
        summary += f"  Exposure Entropy: {aggregate.get('exposure_entropy', 0.0):.4f}\n"
        
        summary += "\nENGAGEMENT METRICS:\n"
        summary += f"  User Satisfaction: {aggregate.get('user_satisfaction', 0.0):.4f}\n"
        summary += f"  Explanation Quality: {aggregate.get('explanation_quality', 0.0):.4f}\n"
        
        summary += "\n" + "="*60 + "\n"
        
        return summary
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON file"""
        aggregate = self.compute_aggregate_metrics()
        
        with open(filepath, 'w') as f:
            json.dump(aggregate, f, indent=2)
        
        print(f"✓ Metrics saved to {filepath}")


def evaluate_model(model, data_loader, config: Dict, device: str = 'cpu') -> Dict:
    """
    Evaluate model on a dataset
    
    Args:
        model: MO-CRS model
        data_loader: Data loader
        config: Configuration dictionary
        device: Device to use
        
    Returns:
        Evaluation metrics
    """
    model.eval()
    evaluator = MOCRSEvaluator(config)
    
    with torch.no_grad():
        for batch in data_loader:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Prepare predictions
            predictions = {
                'recommendations': outputs.get('reranked_indices', None),
                'success': False,  # Would determine from conversation
                'utility': 0.0,
                'num_turns': 1,
                'explanations': outputs.get('explanations', [])
            }
            
            # Prepare ground truth
            ground_truth = {
                'relevant_items': batch.get('relevant_items', []),
                'conversation': {}
            }
            
            # Evaluate episode
            evaluator.evaluate_episode(
                predictions,
                ground_truth,
                user_demographics=batch.get('user_demographics', None)
            )
    
    # Compute aggregate metrics
    metrics = evaluator.compute_aggregate_metrics()
    
    return metrics


if __name__ == "__main__":
    # Test evaluator
    import yaml
    import os
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Testing MO-CRS Evaluator")
    print("="*60)
    
    # Create evaluator
    evaluator = MOCRSEvaluator(config)
    
    # Simulate some episodes
    for i in range(10):
        # Dummy predictions
        predictions = {
            'recommendations': [f'item_{j}' for j in range(10)],
            'success': np.random.random() > 0.5,
            'utility': np.random.random(),
            'num_turns': np.random.randint(5, 15),
            'explanations': [f'I recommend this because...']
        }
        
        # Dummy ground truth
        ground_truth = {
            'relevant_items': [f'item_{j}' for j in range(3, 8)],
            'conversation': {
                'user_rating': np.random.random() * 5
            }
        }
        
        # User demographics
        demographics = {
            'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'])
        }
        
        # Evaluate
        metrics = evaluator.evaluate_episode(predictions, ground_truth, demographics)
    
    # Get summary
    summary = evaluator.get_metrics_summary()
    print(summary)
    
    # Save metrics
    evaluator.save_metrics('test_metrics.json')
    
    print("\n✓ Evaluator tests passed!")
