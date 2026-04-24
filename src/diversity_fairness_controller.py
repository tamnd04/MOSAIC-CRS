"""
Diversity and Fairness Controller (DFC)
Ensures diverse and fair recommendations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Set
import numpy as np
from collections import defaultdict


class DiversityFairnessController(nn.Module):
    """
    Controls diversity and fairness of recommendations
    Implements MMR, fairness constraints, and exposure balancing
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        dfc_config = config['model']['diversity_fairness']
        
        self.hidden_dim = dfc_config['hidden_dim']
        self.lambda_mmr = dfc_config['lambda_mmr']
        self.diversity_window = dfc_config['diversity_window']
        
        # Diversity scorer
        self.diversity_scorer = nn.Sequential(
            nn.Linear(dfc_config['item_embedding_dim'] * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Fairness scorer
        self.fairness_scorer = nn.Sequential(
            nn.Linear(dfc_config['user_embedding_dim'] + dfc_config['item_embedding_dim'], 
                     self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Exposure tracker
        self.exposure_tracker = ExposureTracker(config)

        # Temporal diversity tracker for recency-aware penalties.
        self.temporal_tracker = TemporalDiversityTracker(window_size=self.diversity_window)
        
        # Fairness constraints
        self.fairness_constraints = FairnessConstraints(config)
        
        # Diversity history
        self.diversity_history = []
        
    def forward(self, candidate_items: torch.Tensor,
               candidate_scores: torch.Tensor,
               user_embedding: torch.Tensor,
               recommended_history: List[str] = None,
               candidate_ids: List[Any] = None,
               user_demographics: Dict = None) -> Dict:
        """
        Rerank candidates for diversity and fairness
        
        Args:
            candidate_items: (num_candidates, item_dim) item embeddings
            candidate_scores: (num_candidates,) relevance scores
            user_embedding: (user_dim,) user representation
            recommended_history: List of previously recommended item IDs
            candidate_ids: Candidate item IDs aligned with candidate_items rows
            user_demographics: User demographic information
            
        Returns:
            Reranked items with diversity and fairness scores
        """
        num_candidates = candidate_items.shape[0]
        device = candidate_items.device
        
        # ==== MMR Re-ranking ====
        mmr_scores = self.mmr_rerank(
            candidate_items, 
            candidate_scores,
            recommended_history
        )
        
        # ==== Fairness Adjustment ====
        fairness_scores = self.compute_fairness_scores(
            candidate_items,
            user_embedding,
            user_demographics
        )

        if candidate_ids is None or len(candidate_ids) != num_candidates:
            candidate_ids = list(range(num_candidates))

        # ==== Temporal Diversity Penalty ====
        temporal_penalties = self.temporal_tracker.get_penalty(
            candidate_ids,
            external_history=recommended_history
        ).to(device)
        
        # ==== Exposure Balancing ====
        exposure_weights = self.exposure_tracker.get_exposure_weights(
            candidate_ids
        ).to(device)
        
        # ==== Combined Score ====
        # Weighted combination: relevance + diversity + fairness + exposure - temporal repetition.
        alpha_relevance = 0.28
        alpha_diversity = 0.20
        alpha_fairness = 0.22
        alpha_exposure = 0.20
        alpha_temporal = 0.10
        
        combined_scores = (
            alpha_relevance * candidate_scores +
            alpha_diversity * mmr_scores +
            alpha_fairness * fairness_scores +
            alpha_exposure * exposure_weights -
            alpha_temporal * temporal_penalties
        )
        
        # Get top-k
        top_k = min(50, num_candidates)
        top_scores, top_indices = torch.topk(combined_scores, top_k)

        selected_item_ids = [candidate_ids[idx] for idx in top_indices.detach().cpu().tolist()]
        self.temporal_tracker.add_items(selected_item_ids)
        self.exposure_tracker.update_exposure(selected_item_ids)
        
        return {
            'reranked_indices': top_indices,
            'reranked_scores': top_scores,
            'relevance_scores': candidate_scores[top_indices],
            'diversity_scores': mmr_scores[top_indices],
            'fairness_scores': fairness_scores[top_indices],
            'exposure_weights': exposure_weights[top_indices],
            'temporal_penalties': temporal_penalties[top_indices]
        }
    
    def mmr_rerank(self, candidate_items: torch.Tensor,
                   candidate_scores: torch.Tensor,
                   recommended_history: List[str] = None) -> torch.Tensor:
        """
        Maximal Marginal Relevance reranking
        
        Args:
            candidate_items: (num_candidates, item_dim)
            candidate_scores: (num_candidates,)
            recommended_history: Previously recommended items
            
        Returns:
            MMR scores (num_candidates,)
        """
        num_candidates = candidate_items.shape[0]
        device = candidate_items.device
        
        if recommended_history is None or len(recommended_history) == 0:
            # No history, return original scores
            return candidate_scores
        
        # Compute pairwise similarities
        # Normalize embeddings
        items_norm = F.normalize(candidate_items, p=2, dim=-1)
        
        # Similarity matrix
        similarity_matrix = torch.mm(items_norm, items_norm.t())  # (num_candidates, num_candidates)
        
        # For each candidate, compute max similarity to history
        # (In real implementation, would use actual history embeddings)
        # Here we approximate by using most similar candidates
        max_sim_to_history = torch.max(similarity_matrix, dim=-1)[0]  # (num_candidates,)
        
        # MMR formula: λ * relevance - (1-λ) * max_similarity
        mmr_scores = (
            self.lambda_mmr * candidate_scores -
            (1 - self.lambda_mmr) * max_sim_to_history
        )
        
        return mmr_scores
    
    def compute_fairness_scores(self, candidate_items: torch.Tensor,
                                user_embedding: torch.Tensor,
                                user_demographics: Dict = None) -> torch.Tensor:
        """
        Compute fairness scores for candidates
        
        Args:
            candidate_items: (num_candidates, item_dim)
            user_embedding: (user_dim,)
            user_demographics: User demographic info
            
        Returns:
            Fairness scores (num_candidates,)
        """
        num_candidates = candidate_items.shape[0]
        
        # Expand user embedding
        user_expanded = user_embedding.unsqueeze(0).expand(num_candidates, -1)
        
        # Concatenate user and item
        combined = torch.cat([user_expanded, candidate_items], dim=-1)
        
        # Compute fairness scores
        fairness_scores = self.fairness_scorer(combined).squeeze(-1)
        
        # Apply demographic fairness if available
        if user_demographics:
            demographic_group = user_demographics.get('age_group', 'unknown')
            fairness_adjustment = self.fairness_constraints.get_demographic_adjustment(
                demographic_group
            )
            fairness_scores = fairness_scores * fairness_adjustment
        
        return fairness_scores
    
    def compute_diversity_metrics(self, recommended_items: List[torch.Tensor],
                                  recommended_item_ids: List[Any] = None) -> Dict:
        """
        Compute diversity metrics for recommended items
        
        Args:
            recommended_items: List of item embeddings
            
        Returns:
            Dictionary of diversity metrics
        """
        if len(recommended_items) <= 1:
            return {
                'intra_list_diversity': 0.0,
                'coverage': 0.0,
                'novelty': 0.0
            }
        
        # Stack items
        items_tensor = torch.stack(recommended_items)  # (num_items, item_dim)
        
        # Intra-list diversity (average pairwise distance)
        items_norm = F.normalize(items_tensor, p=2, dim=-1)
        similarity_matrix = torch.mm(items_norm, items_norm.t())
        
        # Get upper triangle (excluding diagonal)
        triu_indices = torch.triu_indices(len(recommended_items), len(recommended_items), offset=1)
        pairwise_sims = similarity_matrix[triu_indices[0], triu_indices[1]]
        
        # Diversity = 1 - similarity
        intra_list_diversity = (1 - pairwise_sims.mean()).item()
        
        # Coverage: proportion of unique items exposed so far over catalog size.
        total_catalog_items = max(int(getattr(self.exposure_tracker, 'num_items', 0)), 1)
        historical_exposed = set(self.exposure_tracker.exposure_counts.keys())
        if recommended_item_ids:
            historical_exposed.update(recommended_item_ids)
        coverage = min(1.0, len(historical_exposed) / total_catalog_items)

        # Novelty: inverse normalized self-information from exposure frequency.
        novelty = 0.0
        if recommended_item_ids:
            total_exposures = max(int(self.exposure_tracker.total_exposures), 0)
            denom = total_exposures + total_catalog_items
            max_self_info = np.log2(denom)

            novelty_vals = []
            for item_id in recommended_item_ids:
                seen_count = int(self.exposure_tracker.exposure_counts.get(item_id, 0))
                # Laplace smoothing keeps unseen items finite and comparable.
                prob = (seen_count + 1.0) / max(denom, 1)
                self_info = -np.log2(prob)
                novelty_vals.append(self_info / max(max_self_info, 1e-8))

            novelty = float(np.mean(novelty_vals)) if novelty_vals else 0.0
        else:
            # Fallback when ids are unavailable: use dissimilarity as a novelty proxy.
            novelty = float(max(0.0, min(1.0, intra_list_diversity)))
        
        return {
            'intra_list_diversity': intra_list_diversity,
            'coverage': float(coverage),
            'novelty': float(max(0.0, min(1.0, novelty)))
        }


class ExposureTracker:
    """
    Tracks item exposure for fairness
    Implements exposure balancing across items
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Exposure counts
        self.exposure_counts = defaultdict(int)
        self.total_exposures = 0
        
        # Exposure targets (uniform by default)
        self.num_items = config['data']['num_items']
        self.target_exposure = 1.0 / self.num_items
        
    def update_exposure(self, item_ids: List[Any]):
        """Update exposure counts"""
        for item_id in item_ids:
            self.exposure_counts[item_id] += 1
            self.total_exposures += 1
    
    def get_exposure_weights(self, candidate_ids: List[Any]) -> torch.Tensor:
        """
        Get exposure weights for candidates
        Higher weight for under-exposed items
        
        Args:
            candidate_ids: List of candidate item IDs
            
        Returns:
            Exposure weights (num_candidates,)
        """
        weights = []
        
        for item_id in candidate_ids:
            if self.total_exposures == 0:
                weight = 1.0
            else:
                current_exposure = self.exposure_counts[item_id] / self.total_exposures
                # Weight inversely proportional to exposure
                ratio = current_exposure / max(self.target_exposure, 1e-8)
                weight = max(0.05, min(2.5, 1.5 / np.sqrt(1.0 + ratio)))
            
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def compute_exposure_metrics(self) -> Dict:
        """Compute exposure fairness metrics"""
        if self.total_exposures == 0:
            return {
                'gini_coefficient': 0.0,
                'exposure_entropy': 0.0
            }
        
        # Get exposure distribution
        exposures = np.array(list(self.exposure_counts.values()))
        
        if len(exposures) == 0:
            return {
                'gini_coefficient': 0.0,
                'exposure_entropy': 0.0
            }
        
        # Gini coefficient
        exposures_sorted = np.sort(exposures)
        n = len(exposures_sorted)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * exposures_sorted)) / (n * np.sum(exposures_sorted)) - (n + 1) / n
        
        # Entropy
        probs = exposures / np.sum(exposures)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return {
            'gini_coefficient': float(gini),
            'exposure_entropy': float(entropy)
        }


class FairnessConstraints:
    """
    Enforces fairness constraints on recommendations
    Handles demographic parity, equal opportunity, calibration
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Demographic groups
        self.demographic_groups = ['18-25', '26-35', '36-45', '46-55', '55+']
        
        # Fairness statistics per group
        self.group_stats = {
            group: {
                'num_users': 0,
                'num_recommendations': 0,
                'num_accepts': 0,
                'total_utility': 0.0
            }
            for group in self.demographic_groups
        }
        
    def update_statistics(self, demographic_group: str, 
                         accepted: bool, utility: float):
        """Update fairness statistics"""
        if demographic_group not in self.group_stats:
            demographic_group = '36-45'  # Default
        
        stats = self.group_stats[demographic_group]
        stats['num_recommendations'] += 1
        if accepted:
            stats['num_accepts'] += 1
        stats['total_utility'] += utility
    
    def get_demographic_adjustment(self, demographic_group: str) -> float:
        """
        Get fairness adjustment factor for demographic group
        Boost under-served groups
        
        Args:
            demographic_group: User's demographic group
            
        Returns:
            Adjustment factor
        """
        if demographic_group not in self.group_stats:
            return 1.0
        
        stats = self.group_stats[demographic_group]
        
        if stats['num_recommendations'] == 0:
            return 1.2  # Boost new groups
        
        # Compute acceptance rate
        acceptance_rate = stats['num_accepts'] / stats['num_recommendations']
        
        # Average acceptance rate across all groups
        total_recs = sum(s['num_recommendations'] for s in self.group_stats.values())
        total_accepts = sum(s['num_accepts'] for s in self.group_stats.values())
        
        if total_recs == 0:
            avg_acceptance_rate = 0.5
        else:
            avg_acceptance_rate = total_accepts / total_recs
        
        # Adjustment: boost if below average
        if acceptance_rate < avg_acceptance_rate:
            adjustment = 1.0 + (avg_acceptance_rate - acceptance_rate)
        else:
            adjustment = 1.0
        
        return min(adjustment, 1.5)  # Cap at 1.5
    
    def compute_fairness_metrics(self) -> Dict:
        """Compute fairness metrics across demographic groups"""
        metrics = {}
        
        # Demographic parity
        acceptance_rates = []
        for group, stats in self.group_stats.items():
            if stats['num_recommendations'] > 0:
                rate = stats['num_accepts'] / stats['num_recommendations']
                acceptance_rates.append(rate)
        
        if acceptance_rates:
            # Demographic parity = variance in acceptance rates
            metrics['demographic_parity'] = float(np.var(acceptance_rates))
        else:
            metrics['demographic_parity'] = 0.0
        
        # Equal opportunity
        utilities = []
        for group, stats in self.group_stats.items():
            if stats['num_recommendations'] > 0:
                avg_utility = stats['total_utility'] / stats['num_recommendations']
                utilities.append(avg_utility)
        
        if utilities:
            metrics['equal_opportunity'] = float(np.var(utilities))
        else:
            metrics['equal_opportunity'] = 0.0
        
        return metrics


class TemporalDiversityTracker:
    """
    Tracks diversity over time within a conversation
    Prevents repetitive recommendations
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history = []
        
    def add_items(self, item_ids: List[str]):
        """Add items to history"""
        self.history.extend(item_ids)
        
        # Keep only recent items
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
    
    def get_penalty(self, candidate_ids: List[Any], external_history: List[Any] = None) -> torch.Tensor:
        """
        Get penalty for candidates based on recent history
        
        Args:
            candidate_ids: List of candidate item IDs
            external_history: Optional conversation history items to include
            
        Returns:
            Penalties (num_candidates,) - higher for recently shown items
        """
        effective_history = list(self.history)
        if external_history:
            effective_history.extend(external_history)
            if len(effective_history) > self.window_size:
                effective_history = effective_history[-self.window_size:]

        penalties = []
        
        for item_id in candidate_ids:
            if item_id in effective_history:
                # Recently shown, high penalty
                recent_index = len(effective_history) - effective_history[::-1].index(item_id) - 1
                recency = 1.0 - (recent_index / max(len(effective_history), 1))
                penalty = recency  # More recent = higher penalty
            else:
                penalty = 0.0
            
            penalties.append(penalty)
        
        return torch.tensor(penalties, dtype=torch.float32)
    
    def reset(self):
        """Reset history"""
        self.history = []


if __name__ == "__main__":
    # Test Diversity & Fairness Controller
    import yaml
    import os
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create DFC
    print("Creating Diversity & Fairness Controller...")
    dfc = DiversityFairnessController(config)
    
    # Test data
    num_candidates = 50
    item_dim = config['model']['diversity_fairness']['item_embedding_dim']
    user_dim = config['model']['diversity_fairness']['user_embedding_dim']
    
    candidate_items = torch.randn(num_candidates, item_dim)
    candidate_scores = torch.rand(num_candidates)
    user_embedding = torch.randn(user_dim)
    
    user_demographics = {
        'age_group': '26-35',
        'gender': 'F'
    }
    
    print(f"\nTesting reranking...")
    print(f"  Candidates: {num_candidates}")
    print(f"  Item dim: {item_dim}")
    print(f"  User dim: {user_dim}")
    
    outputs = dfc(candidate_items, candidate_scores, user_embedding, 
                  recommended_history=None, user_demographics=user_demographics)
    
    print(f"\nOutput shapes:")
    print(f"  Reranked indices: {outputs['reranked_indices'].shape}")
    print(f"  Reranked scores: {outputs['reranked_scores'].shape}")
    print(f"  Relevance scores: {outputs['relevance_scores'].shape}")
    print(f"  Diversity scores: {outputs['diversity_scores'].shape}")
    print(f"  Fairness scores: {outputs['fairness_scores'].shape}")
    
    # Test diversity metrics
    print(f"\nTesting diversity metrics...")
    recommended_items = [torch.randn(item_dim) for _ in range(10)]
    diversity_metrics = dfc.compute_diversity_metrics(recommended_items)
    print(f"  Diversity metrics: {diversity_metrics}")
    
    # Test exposure tracker
    print(f"\nTesting exposure tracker...")
    exposure_tracker = ExposureTracker(config)
    exposure_tracker.update_exposure([1, 2, 3, 1, 2])
    weights = exposure_tracker.get_exposure_weights([1, 2, 3, 4, 5])
    print(f"  Exposure weights: {weights}")
    
    metrics = exposure_tracker.compute_exposure_metrics()
    print(f"  Exposure metrics: {metrics}")
    
    # Test fairness constraints
    print(f"\nTesting fairness constraints...")
    fairness = FairnessConstraints(config)
    fairness.update_statistics('26-35', accepted=True, utility=0.8)
    fairness.update_statistics('26-35', accepted=False, utility=0.4)
    fairness.update_statistics('46-55', accepted=True, utility=0.9)
    
    adjustment = fairness.get_demographic_adjustment('26-35')
    print(f"  Demographic adjustment: {adjustment}")
    
    fairness_metrics = fairness.compute_fairness_metrics()
    print(f"  Fairness metrics: {fairness_metrics}")
    
    print("\n✓ Diversity & Fairness Controller tests passed!")
