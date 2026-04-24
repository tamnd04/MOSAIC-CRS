"""
Comprehensive test-only evaluation suite for MO-CRS.

Outputs a unified metric payload containing:
- Recommendation metrics: Recall, MRR, NDCG
- Conversation metrics: Dist-n, BLEU-n, SR@K, AT
- Diversity metrics: ILD@K, genre/category coverage, calibration error
- Fairness metrics: A@K, G@K, L@K, D@K, and exposure entropy
- Transparency metrics: groundedness, hallucination, persuasiveness,
  transparency, trust, usefulness
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from environment import ConversationalRecommenderEnv
from explanation_generator import ExplanationEvaluator
from off_policy_evaluation import _build_eval_batch, _load_conversations_for_eval


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(str(text))]


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _extract_value_tokens(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(_extract_value_tokens(v))
        return out
    if not isinstance(value, str):
        value = str(value)

    parts = re.split(r"[|,/;]", value)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def _get_item_popularity(item: Optional[Dict]) -> float:
    if not item:
        return 0.0
    return float(item.get('mentions', item.get('popularity', 0.0)) or 0.0)


def _extract_item_categories(item: Optional[Dict]) -> Set[str]:
    if not item:
        return {'Unknown'}

    cats = set()
    for key in ['category', 'categories', 'genre', 'genres']:
        for token in _extract_value_tokens(item.get(key)):
            cats.add(token)

    return cats if cats else {'Unknown'}


def _extract_item_genres(item: Optional[Dict]) -> Set[str]:
    if not item:
        return {'Unknown'}

    genres = set()
    for key in ['genre', 'genres', 'category', 'categories']:
        for token in _extract_value_tokens(item.get(key)):
            genres.add(token)

    return genres if genres else {'Unknown'}


def _compute_bleu_n(candidate_tokens: List[str], reference_tokens: List[str], n: int) -> float:
    if not candidate_tokens or not reference_tokens:
        return 0.0

    # Sentence BLEU-n with modified precision and simple smoothing.
    precisions = []
    eps = 1e-9

    for k in range(1, n + 1):
        cand_ngrams = [tuple(candidate_tokens[i:i + k]) for i in range(max(0, len(candidate_tokens) - k + 1))]
        ref_ngrams = [tuple(reference_tokens[i:i + k]) for i in range(max(0, len(reference_tokens) - k + 1))]

        if not cand_ngrams:
            precisions.append(eps)
            continue

        cand_counts = Counter(cand_ngrams)
        ref_counts = Counter(ref_ngrams)

        clipped = 0
        total = 0
        for ng, cnt in cand_counts.items():
            clipped += min(cnt, ref_counts.get(ng, 0))
            total += cnt

        precisions.append((clipped + eps) / (total + eps))

    c = len(candidate_tokens)
    r = len(reference_tokens)
    bp = 1.0 if c > r else math.exp(1.0 - (r / max(c, 1)))

    score = bp * math.exp(sum(math.log(p) for p in precisions) / float(n))
    return float(score)


def _distinct_n(texts: List[str], n: int) -> float:
    all_ngrams = []
    for text in texts:
        toks = _tokenize(text)
        if len(toks) < n:
            continue
        all_ngrams.extend([tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)])

    if not all_ngrams:
        return 0.0
    return float(len(set(all_ngrams)) / max(len(all_ngrams), 1))


def _recall_at_k(ranked_ids: List[str], rel_set: Set[str], k: int) -> float:
    if not rel_set:
        return 0.0
    topk = ranked_ids[:k]
    hits = sum(1 for item in rel_set if item in topk)
    return float(hits / max(len(rel_set), 1))


def _mrr_at_k(ranked_ids: List[str], rel_set: Set[str], k: int) -> float:
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in rel_set:
            return float(1.0 / rank)
    return 0.0


def _ndcg_at_k(ranked_ids: List[str], rel_set: Set[str], k: int) -> float:
    dcg = 0.0
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in rel_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(rel_set), k)
    if ideal_hits <= 0:
        return 0.0

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _gini_from_counts(counts: np.ndarray) -> float:
    arr = np.asarray(counts, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if np.all(arr == 0):
        return 0.0

    arr = np.sort(arr)
    n = arr.size
    cum = np.sum((np.arange(1, n + 1)) * arr)
    total = arr.sum()
    return float((2.0 * cum) / (n * total) - (n + 1.0) / n)


def _kl_to_uniform(counts: np.ndarray, eps: float = 1e-12) -> float:
    arr = np.asarray(counts, dtype=np.float64)
    n = arr.size
    if n == 0:
        return 0.0

    p = (arr + eps) / (arr.sum() + eps * n)
    u = np.full(n, 1.0 / n, dtype=np.float64)
    return float(np.sum(p * np.log(np.maximum(p, eps) / np.maximum(u, eps))))


def _normalized_entropy(counts: np.ndarray, eps: float = 1e-12) -> float:
    arr = np.asarray(counts, dtype=np.float64)
    n = arr.size
    if n == 0:
        return 0.0

    p = (arr + eps) / (arr.sum() + eps * n)
    h = -float(np.sum(p * np.log(np.maximum(p, eps))))
    return float(h / math.log(max(n, 2)))


def _conversation_interest_distribution(conversation: Dict, item_catalog) -> Dict[str, float]:
    if not isinstance(conversation, dict):
        return {}

    item_ids = []

    profile = conversation.get('user_profile', {})
    if isinstance(profile, dict) and isinstance(profile.get('preferences', []), list):
        item_ids.extend(str(x) for x in profile.get('preferences', []) if x is not None)

    for item_id in conversation.get('accepted_items', []) or []:
        if item_id is not None:
            item_ids.append(str(item_id))

    turns = conversation.get('turns', [])
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        for key in ['mentioned_items', 'items_mentioned', 'recommended_items']:
            vals = turn.get(key, [])
            if isinstance(vals, list):
                item_ids.extend(str(v) for v in vals if v is not None)

    if not item_ids:
        return {}

    counts = Counter()
    for item_id in item_ids:
        item = item_catalog.get_item(item_id)
        for cat in _extract_item_categories(item):
            counts[cat] += 1

    total = sum(counts.values())
    if total <= 0:
        return {}

    return {k: float(v / total) for k, v in counts.items()}


def _list_category_distribution(item_ids: List[str], item_catalog) -> Dict[str, float]:
    counts = Counter()
    for item_id in item_ids:
        item = item_catalog.get_item(item_id)
        for cat in _extract_item_categories(item):
            counts[cat] += 1

    total = sum(counts.values())
    if total <= 0:
        return {}

    return {k: float(v / total) for k, v in counts.items()}


def _calibration_error(user_dist: Dict[str, float], rec_dist: Dict[str, float]) -> float:
    if not user_dist or not rec_dist:
        return 0.0

    keys = set(user_dist.keys()).union(rec_dist.keys())
    l1 = sum(abs(user_dist.get(k, 0.0) - rec_dist.get(k, 0.0)) for k in keys)
    return float(0.5 * l1)


def _avg_ild(item_ids: List[str], item_catalog) -> float:
    if len(item_ids) <= 1:
        return 0.0

    embs = []
    for item_id in item_ids:
        emb = item_catalog.get_item_embedding(item_id)
        emb = np.asarray(emb, dtype=np.float64)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        embs.append(emb)

    distances = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sim = float(np.dot(embs[i], embs[j]))
            distances.append(1.0 - sim)

    return float(np.mean(distances)) if distances else 0.0


def _extract_reference_utterance(conversation: Dict, turn_index: Optional[int] = None) -> str:
    if not isinstance(conversation, dict):
        return ''
    turns = conversation.get('turns', [])
    if not isinstance(turns, list) or not turns:
        return ''

    if turn_index is not None and 0 <= turn_index < len(turns):
        turn = turns[turn_index]
        if isinstance(turn, dict):
            for key in ['system_utterance', 'response', 'utterance']:
                text = turn.get(key, '')
                if isinstance(text, str) and text.strip():
                    return text

    for turn in reversed(turns):
        if not isinstance(turn, dict):
            continue
        for key in ['system_utterance', 'response', 'utterance']:
            text = turn.get(key, '')
            if isinstance(text, str) and text.strip():
                return text

    return ''


def _compute_ranking_metrics(trainer, config: Dict, split_file: str) -> Dict:
    conversations = _load_conversations_for_eval(split_file)

    recall10_vals = []
    recall50_vals = []
    mrr10_vals = []
    mrr50_vals = []
    ndcg10_vals = []
    ndcg50_vals = []

    used = 0

    trainer.model.eval()
    with torch.no_grad():
        for conversation in conversations:
            batch, candidate_ids, accepted_items = _build_eval_batch(
                conversation=conversation,
                item_catalog=trainer.item_catalog,
                config=config,
                device=trainer.device,
                candidate_size=100,
            )

            if not accepted_items:
                continue

            outputs = trainer.model(batch)
            reranked_indices = outputs.get('reranked_indices')
            if reranked_indices is None:
                continue

            idx = reranked_indices[0]
            if isinstance(idx, torch.Tensor):
                idx = idx.detach().cpu().numpy().tolist()
            else:
                idx = list(idx)

            ranked_ids = [str(candidate_ids[int(i)]) for i in idx if 0 <= int(i) < len(candidate_ids)]
            rel_set = set(str(x) for x in accepted_items)
            if not rel_set:
                continue

            recall10_vals.append(_recall_at_k(ranked_ids, rel_set, 10))
            recall50_vals.append(_recall_at_k(ranked_ids, rel_set, 50))
            mrr10_vals.append(_mrr_at_k(ranked_ids, rel_set, 10))
            mrr50_vals.append(_mrr_at_k(ranked_ids, rel_set, 50))
            ndcg10_vals.append(_ndcg_at_k(ranked_ids, rel_set, 10))
            ndcg50_vals.append(_ndcg_at_k(ranked_ids, rel_set, 50))
            used += 1

    return {
        'Recall@10': _safe_mean(recall10_vals),
        'Recall@50': _safe_mean(recall50_vals),
        'MRR@10': _safe_mean(mrr10_vals),
        'MRR@50': _safe_mean(mrr50_vals),
        'NDCG@10': _safe_mean(ndcg10_vals),
        'NDCG@50': _safe_mean(ndcg50_vals),
        'num_conversations_used': int(used),
        'num_conversations_total': int(len(conversations)),
    }


def _simulate_online_metrics(
    trainer,
    config: Dict,
    test_conversations: List[Dict],
    episodes: int,
    fairness_ks: List[int],
) -> Dict:
    env = ConversationalRecommenderEnv(config, trainer.item_catalog, mode='test')

    rl_cfg = config.get('training', {}).get('rl', {})
    rollout_horizon = int(rl_cfg.get('rollout_horizon', config['environment']['max_turns']))
    action_counts = np.zeros(4, dtype=np.int64)
    max_k = max(fairness_ks)

    success_flags = []
    turns_hist = []
    sr_by_k = {k: [] for k in fairness_ks}
    generated_texts = []
    reference_texts = []

    expl_eval = ExplanationEvaluator()
    grounded_vals = []
    halluc_vals = []
    persuasive_vals = []
    transparency_vals = []
    trust_vals = []
    useful_vals = []

    exposure_counts = {k: defaultdict(int) for k in fairness_ks}
    ild_values = {k: [] for k in fairness_ks}
    calib_values = {k: [] for k in fairness_ks}
    rec_genres = {k: set() for k in fairness_ks}
    rec_categories = {k: set() for k in fairness_ks}

    catalog_ids = list(trainer.item_catalog.catalog.keys())
    all_genres = set()
    all_categories = set()
    popularity_pairs = []
    for item_id in catalog_ids:
        item = trainer.item_catalog.get_item(item_id)
        all_genres.update(_extract_item_genres(item))
        all_categories.update(_extract_item_categories(item))
        popularity_pairs.append((str(item_id), _get_item_popularity(item)))

    popularity_pairs.sort(key=lambda x: x[1], reverse=True)
    head_cut = max(1, int(0.2 * len(popularity_pairs)))
    head_items = set(item_id for item_id, _ in popularity_pairs[:head_cut])

    ask_before_recommend_turns = int(rl_cfg.get('ask_before_recommend_turns', 1))
    min_turns_before_end = int(rl_cfg.get('min_turns_before_end', config.get('environment', {}).get('min_turns_before_end', 0)))
    min_recommendations_before_end = int(rl_cfg.get('min_recommendations_before_end', config.get('environment', {}).get('min_recommendations_before_end', 1)))
    fallback_action_before_end = int(rl_cfg.get('fallback_action_before_end', 0))

    trainer.model.eval()
    with torch.no_grad():
        for ep in range(max(1, episodes)):
            conv = test_conversations[ep % len(test_conversations)] if test_conversations else {}
            profile = trainer._conversation_to_user_profile(conv, ep)
            user_dist = _conversation_interest_distribution(conv, trainer.item_catalog)
            eval_batch, candidate_ids, _ = _build_eval_batch(conv, trainer.item_catalog, config, trainer.device, candidate_size=100)
            candidate_embeddings = eval_batch['candidate_items']
            candidate_ids_batch = eval_batch['candidate_item_ids']

            state_np = env.reset(user_profile=profile)
            done = False
            step = 0
            ep_success = False
            ep_success_turn = rollout_horizon + 1
            recommended_count = 0

            while (not done) and step < rollout_horizon:
                states = torch.FloatTensor(state_np).unsqueeze(0).to(trainer.device)
                batch = trainer._create_rl_batch(
                    states,
                    candidate_embeddings,
                    candidate_ids_batch,
                    user_profiles=[profile],
                    is_cold_start=(step == 0),
                    use_thompson_sampling=False,
                    generate_explanations=True,
                )
                outputs = trainer.model(batch)
                action_probs = outputs.get('action_probs')
                if action_probs is None:
                    break
                actions = torch.argmax(action_probs, dim=-1)
                reranked_indices = outputs.get('reranked_indices')
                env_action = trainer._build_env_actions(
                    actions.detach().cpu().numpy(),
                    candidate_ids_batch,
                    reranked_indices,
                    top_k=max_k,
                    current_turn=step + 1,
                    min_turns_before_end=min_turns_before_end,
                    recommended_counts=[recommended_count],
                    min_recommendations_before_end=min_recommendations_before_end,
                    fallback_action_type=fallback_action_before_end,
                    ask_before_recommend_turns=ask_before_recommend_turns,
                )[0]

                action_type = int(env_action.get('action_type', 0))
                if 0 <= action_type < 4:
                    action_counts[action_type] += 1
                recommended_items = [str(x) for x in env_action.get('items', [])]
                if action_type == 1 and recommended_items:
                    recommended_count += 1
                    explanation_text = ''
                    if outputs.get('explanations'):
                        explanation_text = str(outputs['explanations'][0])
                        generated_texts.append(explanation_text)
                        reference_texts.append(_extract_reference_utterance(conv, turn_index=step))

                    for k in fairness_ks:
                        topk = recommended_items[:k]
                        if not topk:
                            continue
                        for item_id in topk:
                            exposure_counts[k][item_id] += 1
                            item = trainer.item_catalog.get_item(item_id)
                            rec_genres[k].update(_extract_item_genres(item))
                            rec_categories[k].update(_extract_item_categories(item))
                        ild_values[k].append(_avg_ild(topk, trainer.item_catalog))
                        rec_dist = _list_category_distribution(topk, trainer.item_catalog)
                        calib_values[k].append(_calibration_error(user_dist, rec_dist))

                    top_item = trainer.item_catalog.get_item(recommended_items[0]) if recommended_items else None
                    item_name = ''
                    item_cats = set()
                    if top_item:
                        item_name = str(top_item.get('title', ''))
                        item_cats = _extract_item_categories(top_item)
                    if explanation_text:
                        eval_scores = expl_eval.evaluate(explanation_text, {'name': item_name})
                        mention_item = bool(item_name and item_name.lower() in explanation_text.lower())
                        mention_cat = any(cat.lower() in explanation_text.lower() for cat in item_cats)
                        grounded = 1.0 if (mention_item or mention_cat) else 0.0
                        placeholder_flag = ('item_' in explanation_text.lower()) or ('unknown' in explanation_text.lower())
                        halluc = 1.0 if (grounded < 0.5 or placeholder_flag) else 0.0
                        persuasive = float(eval_scores.get('overall', 0.0))
                        transparency = 0.5 * float(eval_scores.get('has_reasoning', False)) + 0.5 * float(mention_item)
                        trust = max(0.0, 1.0 - 0.7 * halluc + 0.3 * grounded)
                        usefulness = 0.4 * float(eval_scores.get('has_reasoning', False)) + 0.3 * float(eval_scores.get('not_generic', False)) + 0.3 * float(eval_scores.get('length_ok', False))
                        grounded_vals.append(grounded)
                        halluc_vals.append(halluc)
                        persuasive_vals.append(persuasive)
                        transparency_vals.append(transparency)
                        trust_vals.append(float(min(max(trust, 0.0), 1.0)))
                        useful_vals.append(float(min(max(usefulness, 0.0), 1.0)))

                next_state, rewards, done, info = env.step(env_action)
                step += 1
                state_np = next_state
                if done and bool(info.get('success', False)):
                    ep_success = True
                    ep_success_turn = int(info.get('turn', step))

            success_flags.append(1.0 if ep_success else 0.0)
            turns_hist.append(float(step))
            for k in fairness_ks:
                sr_by_k[k].append(1.0 if (ep_success and ep_success_turn <= k) else 0.0)

    total_actions = int(action_counts.sum())
    action_ratios = (action_counts / total_actions).tolist() if total_actions > 0 else [0.0, 0.0, 0.0, 0.0]

    bleu2, bleu3 = [], []
    for hyp, ref in zip(generated_texts, reference_texts):
        hyp_toks = _tokenize(hyp)
        ref_toks = _tokenize(ref)
        if hyp_toks and ref_toks:
            bleu2.append(_compute_bleu_n(hyp_toks, ref_toks, n=2))
            bleu3.append(_compute_bleu_n(hyp_toks, ref_toks, n=3))

    conversation_results = {
        'Dist-2': _distinct_n(generated_texts, 2),
        'Dist-3': _distinct_n(generated_texts, 3),
        'BLEU-2': _safe_mean(bleu2),
        'BLEU-3': _safe_mean(bleu3),
        'SR@5': _safe_mean(sr_by_k.get(5, [])),
        'SR@10': _safe_mean(sr_by_k.get(10, [])),
        'SR@20': _safe_mean(sr_by_k.get(20, [])),
        'AT': _safe_mean(turns_hist),
        'success_rate': _safe_mean(success_flags),
        'num_generated_explanations': int(len(generated_texts)),
    }

    diversity = {}
    fairness = {}
    id_to_pop = {item_id: pop for item_id, pop in popularity_pairs}
    for k in fairness_ks:
        counts_dict = exposure_counts[k]
        counts = np.array([float(counts_dict.get(item_id, 0.0)) for item_id in catalog_ids], dtype=np.float64)
        exposure_total = float(np.sum(counts))
        if exposure_total > 0:
            avg_pop = float(np.sum(np.array([id_to_pop[item_id] for item_id in catalog_ids]) * counts) / exposure_total)
            head_exposure = float(np.sum([counts_dict.get(item_id, 0.0) for item_id in head_items]))
            tail_exposure = max(exposure_total - head_exposure, 0.0)
            head_share = head_exposure / exposure_total
            tail_share = tail_exposure / exposure_total
            diff = tail_share - head_share
        else:
            avg_pop = 0.0
            head_share = 0.0
            tail_share = 0.0
            diff = 0.0
        diversity[f'ILD@{k}'] = _safe_mean(ild_values[k])
        diversity[f'GenreCoverage@{k}'] = float(len(rec_genres[k]) / max(len(all_genres), 1))
        diversity[f'CategoryCoverage@{k}'] = float(len(rec_categories[k]) / max(len(all_categories), 1))
        diversity[f'CalibrationError@{k}'] = _safe_mean(calib_values[k])
        fairness[f'A@{k}'] = avg_pop
        fairness[f'G@{k}'] = _gini_from_counts(counts)
        fairness[f'L@{k}'] = _kl_to_uniform(counts)
        fairness[f'D@{k}'] = diff
        fairness[f'HeadShare@{k}'] = head_share
        fairness[f'TailShare@{k}'] = tail_share
        fairness[f'Entropy@{k}'] = _normalized_entropy(counts)

    transparency = {
        'groundedness_factual_consistency': _safe_mean(grounded_vals),
        'deception_hallucination_rate': _safe_mean(halluc_vals),
        'persuasiveness_score': _safe_mean(persuasive_vals),
        'transparency_score': _safe_mean(transparency_vals),
        'trust_score': _safe_mean(trust_vals),
        'usefulness_score': _safe_mean(useful_vals),
    }

    return {
        'conversation_results': conversation_results,
        'diversity': diversity,
        'fairness': fairness,
        'transparency': transparency,
        'action_ratios': {
            'ask': float(action_ratios[0]),
            'recommend': float(action_ratios[1]),
            'clarify': float(action_ratios[2]),
            'end': float(action_ratios[3]),
        },
        'episodes': int(max(1, episodes)),
        'num_test_conversations': int(len(test_conversations)),
        'rollout_horizon': int(rollout_horizon),
        'num_catalog_items': int(len(catalog_ids)),
    }

def evaluate_full_test_suite(
    trainer,
    config: Dict,
    episodes: int = 80,
    fairness_k_values: Optional[List[int]] = None,
) -> Dict:
    """
    Run full test-only evaluation with recommendation, conversation,
    diversity, fairness, and transparency metrics.
    """
    fairness_ks = fairness_k_values or [5, 10, 20]
    fairness_ks = sorted(set(int(k) for k in fairness_ks if int(k) > 0))
    if not fairness_ks:
        fairness_ks = [5, 10, 20]

    test_file = config.get('data', {}).get('test_file')
    if not test_file:
        raise ValueError('Missing config data.test_file for test-only evaluation')

    test_conversations = _load_conversations_for_eval(test_file)

    recommendation_results = _compute_ranking_metrics(trainer, config, test_file)
    online_metrics = _simulate_online_metrics(
        trainer=trainer,
        config=config,
        test_conversations=test_conversations,
        episodes=episodes,
        fairness_ks=fairness_ks,
    )

    return {
        'split': 'test',
        'recommendation_results': recommendation_results,
        'conversation_results': online_metrics['conversation_results'],
        'diversity': online_metrics['diversity'],
        'fairness': online_metrics['fairness'],
        'transparency': online_metrics['transparency'],
        'online_rollout': {
            'episodes': online_metrics['episodes'],
            'num_test_conversations': online_metrics['num_test_conversations'],
            'rollout_horizon': online_metrics['rollout_horizon'],
            'action_ratios': online_metrics['action_ratios'],
            'num_catalog_items': online_metrics['num_catalog_items'],
        },
    }
