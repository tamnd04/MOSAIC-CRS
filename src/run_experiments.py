"""
Reproducible baseline and ablation runner for MO-CRS.

Usage:
  cd src
  python run_experiments.py --config ../config.yaml --episodes 1500 --seeds 42 43 44
"""

import argparse
import copy
import json
import os
import random
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import yaml

from train import MOCRSTrainer
from off_policy_evaluation import off_policy_evaluate


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_variants(base_config: Dict) -> Dict[str, Dict]:
    variants = {}

    full = copy.deepcopy(base_config)
    full['logging']['experiment_name'] = 'full_model'
    variants['full_model'] = full

    no_div = copy.deepcopy(base_config)
    no_div['logging']['experiment_name'] = 'ablation_no_diversity'
    no_div['training']['rl']['reward_weights']['diversity'] = 0.0
    no_div['model']['diversity_fairness']['lambda_mmr'] = 0.0
    variants['ablation_no_diversity'] = no_div

    no_fair = copy.deepcopy(base_config)
    no_fair['logging']['experiment_name'] = 'ablation_no_fairness'
    no_fair['training']['rl']['reward_weights']['fairness'] = 0.0
    no_fair['training']['rl']['constraint_aware']['enabled'] = False
    variants['ablation_no_fairness'] = no_fair

    single_obj = copy.deepcopy(base_config)
    single_obj['logging']['experiment_name'] = 'ablation_accuracy_only'
    single_obj['training']['rl']['reward_weights'] = {
        'accuracy': 1.0,
        'diversity': 0.0,
        'fairness': 0.0,
        'engagement': 0.0,
    }
    single_obj['training']['rl']['constraint_aware']['enabled'] = False
    variants['ablation_accuracy_only'] = single_obj

    no_bc = copy.deepcopy(base_config)
    no_bc['logging']['experiment_name'] = 'baseline_no_bc_warmstart'
    no_bc['training']['behavioral_cloning']['enabled'] = False
    variants['baseline_no_bc_warmstart'] = no_bc

    return variants


def resolve_config_paths(config: Dict, config_path: str) -> Dict:
    """Resolve config file paths relative to config directory."""
    config_dir = os.path.dirname(os.path.abspath(config_path))

    for key in ['catalog_file', 'train_file', 'val_file', 'test_file', 'full_data_file', 'data_dir', 'processed_dir']:
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


def run_variant(name: str, config: Dict, episodes: int, seeds: List[int]) -> Dict:
    per_seed = []
    for seed in seeds:
        cfg = copy.deepcopy(config)
        cfg['seed'] = seed
        cfg['training']['num_episodes'] = episodes

        set_global_seed(seed)
        trainer = MOCRSTrainer(cfg, use_wandb=False)

        bc_cfg = cfg.get('training', {}).get('behavioral_cloning', {})
        if bc_cfg.get('enabled', False):
            convs = trainer._load_rl_conversations()
            trainer.behavioral_cloning_warmstart(
                convs,
                epochs=int(bc_cfg.get('epochs', 5)),
                learning_rate=float(bc_cfg.get('learning_rate', cfg['training']['learning_rate']))
            )

        trainer.rl_finetune(num_episodes=episodes)

        val_file = cfg.get('data', {}).get('val_file')
        if val_file and os.path.exists(val_file):
            ope = off_policy_evaluate(trainer.model, val_file, trainer.item_catalog, cfg, trainer.device)
        else:
            ope = {'ips': 0.0, 'snips': 0.0, 'dr': 0.0, 'num_samples': 0.0}

        per_seed.append({'seed': seed, 'ope': ope})

    ips_vals = [x['ope']['ips'] for x in per_seed]
    snips_vals = [x['ope']['snips'] for x in per_seed]
    dr_vals = [x['ope']['dr'] for x in per_seed]

    return {
        'variant': name,
        'seeds': seeds,
        'results': per_seed,
        'aggregate': {
            'ips_mean': float(np.mean(ips_vals)) if ips_vals else 0.0,
            'ips_std': float(np.std(ips_vals)) if ips_vals else 0.0,
            'snips_mean': float(np.mean(snips_vals)) if snips_vals else 0.0,
            'snips_std': float(np.std(snips_vals)) if snips_vals else 0.0,
            'dr_mean': float(np.mean(dr_vals)) if dr_vals else 0.0,
            'dr_std': float(np.std(dr_vals)) if dr_vals else 0.0,
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Run reproducible baselines and ablations')
    parser.add_argument('--config', type=str, default='../config.yaml')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--output', type=str, default='../logs/ablation_results.json')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    base_config = resolve_config_paths(base_config, args.config)

    variants = build_variants(base_config)
    all_results = []

    for name, cfg in variants.items():
        print(f"\nRunning variant: {name}")
        result = run_variant(name, cfg, episodes=args.episodes, seeds=args.seeds)
        all_results.append(result)
        print(f"  DR mean: {result['aggregate']['dr_mean']:.6f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'timestamp': datetime.utcnow().isoformat(),
                'episodes': args.episodes,
                'results': all_results
            },
            f,
            indent=2,
            ensure_ascii=True
        )

    print(f"\nSaved ablation results to {args.output}")


if __name__ == '__main__':
    main()
