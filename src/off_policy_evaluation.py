"""
Off-policy evaluation utilities for MO-CRS.

This module contains the active evaluation path used by train.py and
run_experiments.py.
"""

import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


def _bootstrap_mean_ci(
    values: List[float],
    rng: np.random.RandomState,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
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
            0.55 * relevance_f1
            + 0.25 * accepted_count_score
            + 0.10 * recommend_density
            + 0.10 * preference_density
        )
    elif has_recommend:
        reward = 0.15 * recommend_density + 0.10 * preference_density
    else:
        reward = 0.05 * preference_density

    reward = float(min(max(reward, 0.0), 1.0))

    return action, reward


def _build_eval_batch(
    conversation: Dict,
    item_catalog,
    config: Dict,
    device: str,
    candidate_size: int = 100,
) -> Tuple[Dict, List[str], set]:
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


def off_policy_evaluate(
    model,
    validation_file: str,
    item_catalog,
    config: Dict,
    device: str = 'cpu',
    max_samples: int = 5000,
) -> Dict:
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
        progress = tqdm(logged_records, desc='OPE Evaluation', unit='conv')
        for conv, logged_action, logged_reward in progress:
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
        'num_samples': float(len(sampled)),
    }
