"""
Stable ablation runner for MO-CRS.

Key fixes vs the old run_experiments.py:
  1. Supports --dataset and uses apply_dataset_paths(), like train.py.
  2. Starts every variant/seed from the same supervised checkpoint by default.
  3. Uses a temporary checkpoint folder per variant/seed, then deletes it after evaluation.
  4. Evaluates the temporary best RL checkpoint for that variant/seed, not a potentially collapsed final in-memory policy.
  5. Does not persist effective configs or checkpoint artifacts; only the output JSON is saved.
  6. Records collapse diagnostics and both all-seed and valid-seed aggregates.

Example:
  cd src
  python run_experiments.py --config ../config_inspired.yaml --dataset INSPIRED \
      --checkpoint best_model.pt --episodes 1000 --seeds 42 43 44 \
      --output ../logs/INSPIRED_ablation_results_fixed.json
"""

import argparse
import copy
import json
import os
import random
import shutil
import tempfile
import types
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml

from data_utils import apply_dataset_paths
from train import MOCRSTrainer
from off_policy_evaluation import off_policy_evaluate
from test_evaluation_suite import evaluate_full_test_suite


# -----------------------------
# Reproducibility / paths
# -----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Keep cudnn benchmark off for less seed noise. Full deterministic mode can be slower
    # and may fail for some ops, so we do not force torch.use_deterministic_algorithms().
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def resolve_config_paths(config: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    """Resolve config paths relative to the config file directory."""
    config_dir = os.path.dirname(os.path.abspath(config_path))

    for key in [
        'catalog_file', 'train_file', 'val_file', 'test_file', 'full_data_file',
        'data_dir', 'processed_dir', 'data_root'
    ]:
        if key in config.get('data', {}):
            path = config['data'][key]
            if isinstance(path, str) and path and not os.path.isabs(path):
                config['data'][key] = os.path.normpath(os.path.join(config_dir, path))

    for key in ['save_dir', 'log_dir']:
        if key in config.get('logging', {}):
            path = config['logging'][key]
            if isinstance(path, str) and path and not os.path.isabs(path):
                config['logging'][key] = os.path.normpath(os.path.join(config_dir, path))

    return config


def set_if_exists(dct: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    """Set nested config value only when the path already exists."""
    cur = dct
    path = list(path)
    for key in path[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            return
        cur = cur[key]
    if isinstance(cur, dict) and path[-1] in cur:
        cur[path[-1]] = value


def ensure_path(dct: Dict[str, Any], path: Iterable[str], default: Any) -> Any:
    cur = dct
    path = list(path)
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    return cur.setdefault(path[-1], default)


def resolve_checkpoint_path(config: Dict[str, Any], checkpoint: Optional[str]) -> Optional[str]:
    """Resolve checkpoint filename using the dataset checkpoint folder."""
    if not checkpoint:
        return None
    if os.path.isabs(checkpoint) and os.path.exists(checkpoint):
        return checkpoint
    if os.path.exists(checkpoint):
        return os.path.abspath(checkpoint)

    save_dir = config.get('logging', {}).get('save_dir', './checkpoints')
    dataset = str(config.get('data', {}).get('dataset_name', 'default')).strip() or 'default'
    candidates = [
        os.path.join(save_dir, dataset, checkpoint),
        os.path.join(save_dir, checkpoint),
        checkpoint,
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    return os.path.abspath(candidates[0])


def load_model_weights_only(trainer: MOCRSTrainer, checkpoint_path: str) -> Tuple[List[str], List[str]]:
    """Load only model weights, not optimizer state, to keep each ablation run independent."""
    ckpt = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    incompat = trainer.model.load_state_dict(state, strict=False)
    missing = list(getattr(incompat, 'missing_keys', []))
    unexpected = list(getattr(incompat, 'unexpected_keys', []))
    print(f"Loaded model weights from: {checkpoint_path}")
    if missing:
        print(f"  [WARN] missing keys: {len(missing)}; first={missing[:3]}")
    if unexpected:
        print(f"  [WARN] unexpected keys: {len(unexpected)}; first={unexpected[:3]}")
    return missing, unexpected


# -----------------------------
# Variants
# -----------------------------


def _put(dct: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    """Set nested config value, creating intermediate dictionaries when needed."""
    cur = dct
    path = list(path)
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def _variant_meta(cfg: Dict[str, Any], mode: str, load_checkpoint: bool = True) -> Dict[str, Any]:
    cfg.setdefault('ablation', {})
    cfg['ablation']['mode'] = mode
    cfg['ablation']['load_checkpoint'] = bool(load_checkpoint)
    return cfg


def build_variants(base_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build ablation configs.

    The older runner only changed a few YAML values. In this project, several DFC
    weights are hard-coded inside Python forward methods, so those YAML edits may
    not change behavior. We still set the config values here for record keeping,
    but the real ablation is enforced later by apply_runtime_ablation().
    """
    variants: Dict[str, Dict[str, Any]] = {}

    full = copy.deepcopy(base_config)
    full.setdefault('logging', {})['experiment_name'] = 'full_model'
    _variant_meta(full, 'full_model', load_checkpoint=True)
    variants['full_model'] = full

    no_div = copy.deepcopy(base_config)
    no_div.setdefault('logging', {})['experiment_name'] = 'ablation_no_diversity'
    _put(no_div, ['training', 'rl', 'reward_weights', 'diversity'], 0.0)
    _put(no_div, ['environment', 'reward_diversity_factor'], 0.0)
    _put(no_div, ['model', 'diversity_fairness', 'lambda_mmr'], 1.0)
    _put(no_div, ['model', 'diversity_fairness', 'diversity_weight'], 0.0)
    _put(no_div, ['model', 'diversity_fairness', 'mmr_weight'], 0.0)
    _variant_meta(no_div, 'no_diversity', load_checkpoint=True)
    variants['ablation_no_diversity'] = no_div

    no_fair = copy.deepcopy(base_config)
    no_fair.setdefault('logging', {})['experiment_name'] = 'ablation_no_fairness'
    _put(no_fair, ['training', 'rl', 'reward_weights', 'fairness'], 0.0)
    _put(no_fair, ['training', 'rl', 'constraint_aware', 'enabled'], False)
    _put(no_fair, ['environment', 'reward_fairness_factor'], 0.0)
    _put(no_fair, ['model', 'diversity_fairness', 'fairness_weight'], 0.0)
    _put(no_fair, ['model', 'diversity_fairness', 'exposure_weight'], 0.0)
    _put(no_fair, ['model', 'diversity_fairness', 'popularity_weight'], 0.0)
    _put(no_fair, ['model', 'diversity_fairness', 'tail_bonus'], 0.0)
    _put(no_fair, ['model', 'diversity_fairness', 'head_penalty'], 0.0)
    _put(no_fair, ['model', 'diversity_fairness', 'target_exposure', 'head'], 1.0)
    _put(no_fair, ['model', 'diversity_fairness', 'target_exposure', 'mid'], 0.0)
    _put(no_fair, ['model', 'diversity_fairness', 'target_exposure', 'tail'], 0.0)
    _variant_meta(no_fair, 'no_fairness', load_checkpoint=True)
    variants['ablation_no_fairness'] = no_fair

    accuracy_only = copy.deepcopy(base_config)
    accuracy_only.setdefault('logging', {})['experiment_name'] = 'ablation_accuracy_only'
    _put(accuracy_only, ['training', 'rl', 'reward_weights'], {
        'accuracy': 1.0,
        'diversity': 0.0,
        'fairness': 0.0,
        'engagement': 0.0,
        'transparency': 0.0,
    })
    _put(accuracy_only, ['training', 'rl', 'constraint_aware', 'enabled'], False)
    _put(accuracy_only, ['environment', 'reward_diversity_factor'], 0.0)
    _put(accuracy_only, ['environment', 'reward_fairness_factor'], 0.0)
    _put(accuracy_only, ['environment', 'reward_engagement_factor'], 0.0)
    _put(accuracy_only, ['model', 'diversity_fairness', 'lambda_mmr'], 1.0)
    _put(accuracy_only, ['model', 'diversity_fairness', 'fairness_weight'], 0.0)
    _put(accuracy_only, ['model', 'diversity_fairness', 'exposure_weight'], 0.0)
    _put(accuracy_only, ['model', 'diversity_fairness', 'popularity_weight'], 0.0)
    _variant_meta(accuracy_only, 'accuracy_only', load_checkpoint=True)
    variants['ablation_accuracy_only'] = accuracy_only

    no_bc = copy.deepcopy(base_config)
    no_bc.setdefault('logging', {})['experiment_name'] = 'baseline_no_bc_warmstart'
    _put(no_bc, ['training', 'behavioral_cloning', 'enabled'], False)
    _variant_meta(no_bc, 'no_bc_warmstart', load_checkpoint=False)
    variants['baseline_no_bc_warmstart'] = no_bc

    return variants


# -----------------------------
# Runtime ablation enforcement
# -----------------------------

class _ZeroExposureTracker:
    """Replacement used by fairness ablations to disable exposure balancing."""
    total_exposures = 0
    exposure_counts: Dict[Any, int] = {}

    def get_exposure_weights(self, candidate_ids: List[Any]) -> torch.Tensor:
        return torch.zeros(len(candidate_ids), dtype=torch.float32)

    def update_exposure(self, item_ids: List[Any]) -> None:
        return None

    def compute_exposure_metrics(self) -> Dict[str, float]:
        return {'exposure_gini': 0.0, 'exposure_entropy': 0.0}


def _accuracy_only_dfc_forward(self, candidate_items: torch.Tensor,
                               candidate_scores: torch.Tensor,
                               user_embedding: torch.Tensor,
                               recommended_history: List[str] = None,
                               candidate_ids: List[Any] = None,
                               user_demographics: Dict = None) -> Dict[str, torch.Tensor]:
    """Bypass DFC and rank purely by preference/relevance score."""
    scores = candidate_scores.squeeze(-1) if candidate_scores.dim() > 1 else candidate_scores
    num_candidates = int(scores.shape[0])
    top_k = min(int(getattr(self, 'ablation_top_k', 50)), num_candidates)
    top_scores, top_indices = torch.topk(scores, top_k)
    zeros = torch.zeros_like(top_scores)
    return {
        'reranked_indices': top_indices,
        'reranked_scores': top_scores,
        'relevance_scores': scores[top_indices],
        'diversity_scores': zeros,
        'fairness_scores': zeros,
        'exposure_weights': zeros,
        'temporal_penalties': zeros,
        'popularity_bonus': zeros,
    }


def apply_runtime_ablation(trainer: MOCRSTrainer, variant_name: str, cfg: Dict[str, Any]) -> None:
    """Enforce ablations directly on the constructed modules.

    This is necessary because several scoring weights are hard-coded in DFC.forward(),
    so config-only ablations may otherwise produce identical results.
    """
    mode = str(cfg.get('ablation', {}).get('mode', variant_name))
    dfc = getattr(getattr(trainer, 'model', None), 'dfc', None)
    if dfc is None:
        print(f"  [WARN] no DFC module found; runtime ablation {mode!r} skipped")
        return

    setattr(dfc, 'ablation_top_k', int(cfg.get('training', {}).get('rl', {}).get('rerank_top_k', 50)))

    if mode == 'full_model':
        print("  runtime ablation: full model")
        return

    if mode == 'no_diversity':
        print("  runtime ablation: diversity disabled (MMR -> relevance only)")
        if hasattr(dfc, 'lambda_mmr'):
            dfc.lambda_mmr = 1.0

        def no_div_mmr(self, candidate_items, candidate_scores, recommended_history=None):
            return candidate_scores.squeeze(-1) if candidate_scores.dim() > 1 else candidate_scores

        dfc.mmr_rerank = types.MethodType(no_div_mmr, dfc)
        for attr in ['diversity_weight', 'mmr_weight', 'alpha_diversity']:
            if hasattr(dfc, attr):
                setattr(dfc, attr, 0.0)
        return

    if mode == 'no_fairness':
        print("  runtime ablation: fairness/exposure disabled")

        def zero_fairness(self, candidate_items, user_embedding, user_demographics=None):
            return torch.zeros(candidate_items.shape[0], dtype=candidate_items.dtype, device=candidate_items.device)

        dfc.compute_fairness_scores = types.MethodType(zero_fairness, dfc)
        dfc.exposure_tracker = _ZeroExposureTracker()

        if hasattr(dfc, '_candidate_popularity_bonus'):
            def zero_popularity_bonus(self, candidate_ids, device=None):
                device = device or next(self.parameters()).device
                return torch.zeros(len(candidate_ids), dtype=torch.float32, device=device)
            dfc._candidate_popularity_bonus = types.MethodType(zero_popularity_bonus, dfc)

        if hasattr(dfc, '_apply_head_concentration_guard'):
            def no_guard(self, combined_scores, candidate_ids, top_k):
                _, idx = torch.topk(combined_scores, min(top_k, combined_scores.shape[0]))
                return idx
            dfc._apply_head_concentration_guard = types.MethodType(no_guard, dfc)

        for attr in ['fairness_weight', 'exposure_weight', 'popularity_weight', 'tail_bonus',
                     'head_penalty', 'alpha_fairness', 'alpha_exposure', 'alpha_popularity']:
            if hasattr(dfc, attr):
                setattr(dfc, attr, 0.0)
        return

    if mode == 'accuracy_only':
        print("  runtime ablation: accuracy-only reranking (DFC bypassed)")
        dfc.forward = types.MethodType(_accuracy_only_dfc_forward, dfc)
        return

    if mode == 'no_bc_warmstart':
        print("  runtime ablation: no checkpoint/BC warmstart baseline")
        return

    print(f"  [WARN] unknown runtime ablation mode {mode!r}; no module patch applied")


# -----------------------------
# Metrics / collapse detection
# -----------------------------

def _get_nested(d: Dict[str, Any], path: Iterable[str], default: Any = 0.0) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def compute_action_entropy(action_ratios: Dict[str, Any]) -> float:
    vals = np.asarray([_float(v) for v in action_ratios.values()], dtype=np.float64)
    vals = vals[vals > 1e-12]
    if vals.size <= 1:
        return 0.0
    entropy = -float(np.sum(vals * np.log(vals)))
    return entropy / float(np.log(len(action_ratios)))


def diagnose_run(test_eval: Dict[str, Any]) -> Dict[str, Any]:
    conv = test_eval.get('conversation_results', {})
    rollout = test_eval.get('online_rollout', {})
    ratios = rollout.get('action_ratios', {}) if isinstance(rollout, dict) else {}
    success = _float(conv.get('success_rate', conv.get('SR@10', 0.0)))
    at = _float(conv.get('AT', 0.0))
    rec_rate = _float(ratios.get('recommend', 0.0))
    ask_rate = _float(ratios.get('ask', 0.0))
    clar_rate = _float(ratios.get('clarify', 0.0))
    end_rate = _float(ratios.get('end', 0.0))
    action_entropy = compute_action_entropy(ratios)

    collapsed = False
    reasons: List[str] = []
    if success <= 0.01:
        collapsed = True
        reasons.append('zero_success')
    if rec_rate <= 0.01:
        collapsed = True
        reasons.append('no_recommendations')
    if max(ask_rate, rec_rate, clar_rate, end_rate) >= 0.985:
        collapsed = True
        reasons.append('single_action_policy')
    if at >= _float(rollout.get('rollout_horizon', 20.0)) and success <= 0.05:
        collapsed = True
        reasons.append('horizon_timeout')

    return {
        'collapsed': collapsed,
        'collapse_reasons': reasons,
        'action_entropy': action_entropy,
        'success_rate': success,
        'AT': at,
        'ask_rate': ask_rate,
        'recommend_rate': rec_rate,
        'clarify_rate': clar_rate,
        'end_rate': end_rate,
    }


def flatten_metrics(seed_result: Dict[str, Any]) -> Dict[str, float]:
    ope = seed_result.get('ope_test', {})
    test_eval = seed_result.get('test_evaluation', {})
    rec = test_eval.get('recommendation_results', {})
    conv = test_eval.get('conversation_results', {})
    div = test_eval.get('diversity', {})
    fair = test_eval.get('fairness', {})
    trans = test_eval.get('transparency', {})
    diag = seed_result.get('diagnostics', {})

    return {
        'ips': _float(ope.get('ips')),
        'snips': _float(ope.get('snips')),
        'dr': _float(ope.get('dr')),
        'dm': _float(ope.get('dm')),
        'Recall@10': _float(rec.get('Recall@10')),
        'Recall@50': _float(rec.get('Recall@50')),
        'MRR@10': _float(rec.get('MRR@10')),
        'NDCG@10': _float(rec.get('NDCG@10')),
        'NDCG@50': _float(rec.get('NDCG@50')),
        'Success': _float(conv.get('success_rate')),
        'SR@5': _float(conv.get('SR@5')),
        'SR@10': _float(conv.get('SR@10')),
        'SR@20': _float(conv.get('SR@20')),
        'AT': _float(conv.get('AT')),
        'Dist-2': _float(conv.get('Dist-2')),
        'Dist-3': _float(conv.get('Dist-3')),
        'ILD@10': _float(div.get('ILD@10')),
        'CategoryCoverage@10': _float(div.get('CategoryCoverage@10')),
        'CalibrationError@10': _float(div.get('CalibrationError@10')),
        'G@10': _float(fair.get('G@10')),
        'L@10': _float(fair.get('L@10')),
        'D@10': _float(fair.get('D@10')),
        'HeadShare@10': _float(fair.get('HeadShare@10')),
        'TailShare@10': _float(fair.get('TailShare@10')),
        'Entropy@10': _float(fair.get('Entropy@10')),
        'Groundedness': _float(trans.get('groundedness_factual_consistency')),
        'Hallucination': _float(trans.get('deception_hallucination_rate')),
        'Trust': _float(trans.get('trust_score')),
        'Usefulness': _float(trans.get('usefulness_score')),
        'ActionEntropy': _float(diag.get('action_entropy')),
        'AskRate': _float(diag.get('ask_rate')),
        'RecommendRate': _float(diag.get('recommend_rate')),
        'ClarifyRate': _float(diag.get('clarify_rate')),
        'EndRate': _float(diag.get('end_rate')),
    }


def aggregate_results(per_seed: List[Dict[str, Any]], valid_only: bool = False) -> Dict[str, float]:
    rows = [x for x in per_seed if (not valid_only or not x.get('diagnostics', {}).get('collapsed', False))]
    if not rows:
        return {'num_runs': 0.0}

    flat_rows = [flatten_metrics(x) for x in rows]
    keys = sorted(flat_rows[0].keys())
    agg: Dict[str, float] = {'num_runs': float(len(rows))}
    for key in keys:
        vals = np.asarray([r.get(key, 0.0) for r in flat_rows], dtype=np.float64)
        agg[f'{key}_mean'] = float(vals.mean())
        agg[f'{key}_std'] = float(vals.std(ddof=0))
    return agg


# -----------------------------
# Main run logic
# -----------------------------

def run_single_seed(
    variant_name: str,
    config: Dict[str, Any],
    episodes: int,
    seed: int,
    checkpoint_path: Optional[str],
    output_dir: str,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg['seed'] = seed
    cfg['training']['num_episodes'] = int(episodes)

    dataset = str(cfg.get('data', {}).get('dataset_name', 'dataset'))

    # The trainer expects save_dir/log_dir and may write best_rl_model.pt during RL.
    # To avoid persistent ablation artifacts, we write them into a temporary folder,
    # load the temporary best checkpoint for evaluation, then delete the folder.
    tmp_root = tempfile.mkdtemp(prefix=f'mocrs_ablation_{dataset}_{variant_name}_seed_{seed}_')
    run_root = os.path.join(tmp_root, 'checkpoints')
    log_root = os.path.join(tmp_root, 'logs')
    os.makedirs(run_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)
    cfg.setdefault('logging', {})['save_dir'] = run_root
    cfg.setdefault('logging', {})['log_dir'] = log_root
    cfg['logging']['experiment_name'] = f'{variant_name}_seed_{seed}'

    print(f"\n[{variant_name} | seed={seed}] starting")
    print("  using temporary checkpoint/log folders; artifacts will be deleted after evaluation")

    best_rl_loaded = False
    try:
        set_global_seed(seed)
        trainer = MOCRSTrainer(cfg, use_wandb=False)

        missing: List[str] = []
        unexpected: List[str] = []
        load_ckpt_for_variant = bool(cfg.get('ablation', {}).get('load_checkpoint', True))
        effective_checkpoint_path = checkpoint_path if load_ckpt_for_variant else None
        if effective_checkpoint_path:
            missing, unexpected = load_model_weights_only(trainer, effective_checkpoint_path)
        else:
            print("  [INFO] no supervised checkpoint loaded for this variant; starting from random initialization.")

        apply_runtime_ablation(trainer, variant_name, cfg)

        bc_cfg = cfg.get('training', {}).get('behavioral_cloning', {})
        if bc_cfg.get('enabled', False):
            convs = trainer._load_rl_conversations()
            trainer.behavioral_cloning_warmstart(
                convs,
                epochs=int(bc_cfg.get('epochs', 5)),
                learning_rate=float(bc_cfg.get('learning_rate', cfg['training']['learning_rate'])),
            )

        trainer.rl_finetune(num_episodes=int(episodes))

        # Evaluate the selected temporary best RL checkpoint when it exists.
        best_rl_path = os.path.join(run_root, dataset, 'best_rl_model.pt')
        if os.path.exists(best_rl_path):
            print("  loading temporary per-run best RL checkpoint for evaluation")
            load_model_weights_only(trainer, best_rl_path)
            best_rl_loaded = True
            apply_runtime_ablation(trainer, variant_name, cfg)
        else:
            print("  [WARN] temporary best_rl_model.pt not found; evaluating final in-memory model")

        test_file = cfg.get('data', {}).get('test_file')
        if test_file and os.path.exists(test_file):
            ope = off_policy_evaluate(trainer.model, test_file, trainer.item_catalog, cfg, trainer.device)
        else:
            ope = {'ips': 0.0, 'snips': 0.0, 'dr': 0.0, 'num_samples': 0.0}

        full_test_eval = evaluate_full_test_suite(
            trainer=trainer,
            config=cfg,
            episodes=int(cfg.get('evaluation', {}).get('num_eval_episodes', 80)),
            fairness_k_values=[5, 10, 20],
        )

        diagnostics = diagnose_run(full_test_eval)
        if diagnostics['collapsed']:
            print(f"  [WARN] collapsed policy detected: {diagnostics['collapse_reasons']}")
        else:
            print("  valid policy")

        return {
            'seed': seed,
            'ablation_mode': cfg.get('ablation', {}).get('mode', variant_name),
            'checkpoint_loaded': effective_checkpoint_path,
            'best_rl_checkpoint_loaded': bool(best_rl_loaded),
            'checkpoint_artifacts_saved': False,
            'checkpoint_missing_keys': missing[:20],
            'checkpoint_unexpected_keys': unexpected[:20],
            'ope_test': ope,
            'test_evaluation': full_test_eval,
            'diagnostics': diagnostics,
        }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

def run_variant(
    name: str,
    config: Dict[str, Any],
    episodes: int,
    seeds: List[int],
    checkpoint_path: Optional[str],
    output_dir: str,
) -> Dict[str, Any]:
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        result = run_single_seed(name, config, episodes, int(seed), checkpoint_path, output_dir)
        per_seed.append(result)
        # Release CUDA memory between seeds/variants.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    collapsed = [x for x in per_seed if x.get('diagnostics', {}).get('collapsed', False)]
    return {
        'variant': name,
        'seeds': seeds,
        'num_collapsed': len(collapsed),
        'collapsed_seeds': [x['seed'] for x in collapsed],
        'results': per_seed,
        'aggregate_all': aggregate_results(per_seed, valid_only=False),
        'aggregate_valid_only': aggregate_results(per_seed, valid_only=True),
        # Backward-compatible key used by older plotting/table scripts.
        'aggregate': aggregate_results(per_seed, valid_only=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run stable reproducible baselines and ablations')
    parser.add_argument('--config', type=str, default='../config.yaml')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset override, e.g. ReDial or INSPIRED')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt', help='Supervised checkpoint to start every ablation run from')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--output', type=str, default='../logs/ablation_results.json')
    parser.add_argument('--variants', type=str, nargs='*', default=None, help='Optional subset of variants to run')
    parser.add_argument('--allow_random_init', action='store_true', help='Allow running without a checkpoint if it is missing')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    base_config = resolve_config_paths(base_config, args.config)
    base_config = apply_dataset_paths(base_config, args.dataset)

    dataset = str(base_config.get('data', {}).get('dataset_name', 'dataset'))
    print(f"Dataset: {dataset}")
    print(f"  train={base_config['data'].get('train_file')}")
    print(f"  val={base_config['data'].get('val_file')}")
    print(f"  test={base_config['data'].get('test_file')}")

    checkpoint_path = resolve_checkpoint_path(base_config, args.checkpoint)
    if checkpoint_path and not os.path.exists(checkpoint_path):
        msg = f"Checkpoint not found: {checkpoint_path}. Run train.py --mode both first or pass --allow_random_init."
        if not args.allow_random_init:
            raise FileNotFoundError(msg)
        print(f"[WARN] {msg}")
        checkpoint_path = None
    print(f"Starting checkpoint for all variants/seeds: {checkpoint_path}")

    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path) or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    variants = build_variants(base_config)
    if args.variants:
        requested = set(args.variants)
        variants = {k: v for k, v in variants.items() if k in requested}
        missing = requested.difference(variants.keys())
        if missing:
            raise ValueError(f"Unknown variants requested: {sorted(missing)}")

    all_results = []
    for name, cfg in variants.items():
        print(f"\n========== Running variant: {name} ==========")
        result = run_variant(
            name=name,
            config=cfg,
            episodes=int(args.episodes),
            seeds=[int(s) for s in args.seeds],
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )
        all_results.append(result)
        agg = result['aggregate_all']
        valid = result['aggregate_valid_only']
        print(
            f"  all: DR={agg.get('dr_mean', 0.0):.4f}, R@10={agg.get('Recall@10_mean', 0.0):.4f}, "
            f"SR@10={agg.get('SR@10_mean', 0.0):.4f}, AT={agg.get('AT_mean', 0.0):.2f}, "
            f"collapsed={result['num_collapsed']}/{len(args.seeds)}"
        )
        print(
            f"  valid-only: n={int(valid.get('num_runs', 0))}, DR={valid.get('dr_mean', 0.0):.4f}, "
            f"R@10={valid.get('Recall@10_mean', 0.0):.4f}, SR@10={valid.get('SR@10_mean', 0.0):.4f}"
        )

        # Incremental save in case a long ablation run is interrupted.
        payload = {
            'timestamp': datetime.utcnow().isoformat(),
            'config': os.path.abspath(args.config),
            'dataset': dataset,
            'episodes': int(args.episodes),
            'seeds': [int(s) for s in args.seeds],
            'checkpoint': checkpoint_path,
            'results': all_results,
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)

    print(f"\nSaved ablation results to {output_path}")


if __name__ == '__main__':
    main()
