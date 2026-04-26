"""
Plot MO-CRS training/validation/testing curves from checkpoints and eval JSON.

Usage examples:
  python plot_training_eval.py --config config_thesis.yaml
  python plot_training_eval.py --config config_thesis.yaml --checkpoint checkpoints/ReDial/best_model.pt --eval_json logs/ReDial_best_model_test_eval.json
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def _safe_series(values: List[float], window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    if window <= 1 or arr.size < window:
        return arr
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(arr, kernel, mode='same')


def _resolve_checkpoint(config: Dict, checkpoint_arg: str = None) -> str:
    if checkpoint_arg:
        return checkpoint_arg

    dataset = str(config.get('data', {}).get('dataset_name', 'ReDial'))
    save_dir = str(config.get('logging', {}).get('save_dir', './checkpoints'))

    candidates = [
        os.path.join(save_dir, dataset, 'best_rl_model.pt'),
        os.path.join(save_dir, dataset, 'best_model.pt'),
        os.path.join(save_dir, 'best_rl_model.pt'),
        os.path.join(save_dir, 'best_model.pt'),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return candidates[0]


def _resolve_eval_json(config: Dict, eval_arg: str = None) -> str:
    if eval_arg:
        return eval_arg

    log_dir = str(config.get('logging', {}).get('log_dir', './logs'))
    dataset = str(config.get('data', {}).get('dataset_name', 'ReDial'))

    candidates = [
        os.path.join(log_dir, f'{dataset}_best_model_test_eval.json'),
        os.path.join(log_dir, 'ReDial_best_model_test_eval.json'),
        os.path.join(log_dir, f'{dataset}_train_eval.json'),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return candidates[0]


def _load_train_stats(checkpoint_path: str) -> Dict:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return checkpoint.get('train_stats', {}) if isinstance(checkpoint, dict) else {}


def _load_eval_payload(eval_json_path: str) -> Dict:
    if not os.path.exists(eval_json_path):
        return {}
    with open(eval_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _plot_supervised_history(ax, supervised_history: List[Dict]) -> None:
    if not supervised_history:
        ax.set_title('Supervised Training/Validation')
        ax.text(0.5, 0.5, 'No supervised_history in checkpoint', ha='center', va='center')
        ax.axis('off')
        return

    epochs = [int(x.get('epoch', i + 1)) for i, x in enumerate(supervised_history)]
    train_loss = [float(x.get('train_total_loss', 0.0)) for x in supervised_history]
    val_loss = [float(x.get('val_total_loss', 0.0)) for x in supervised_history]
    val_intent_acc = [float(x.get('val_intent_accuracy', 0.0)) for x in supervised_history]
    val_sent_acc = [float(x.get('val_sentiment_accuracy', 0.0)) for x in supervised_history]

    ax.plot(epochs, train_loss, label='Train Loss', linewidth=2)
    ax.plot(epochs, val_loss, label='Val Loss', linewidth=2)
    ax.plot(epochs, val_intent_acc, label='Val Intent Acc', linestyle='--')
    ax.plot(epochs, val_sent_acc, label='Val Sentiment Acc', linestyle='--')
    ax.set_title('Supervised Training + Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric')
    ax.grid(alpha=0.3)
    ax.legend(loc='best')


def _plot_rl_history(ax, rl_history: List[Dict]) -> None:
    if not rl_history:
        ax.set_title('RL Fine-tuning')
        ax.text(0.5, 0.5, 'No rl_history in checkpoint', ha='center', va='center')
        ax.axis('off')
        return

    episodes = [int(x.get('episode', i + 1)) for i, x in enumerate(rl_history)]
    avg_reward = [float(x.get('avg_reward', 0.0)) for x in rl_history]
    online_success = [float(x.get('online_success', 0.0)) for x in rl_history]
    online_div = [float(x.get('online_diversity', 0.0)) for x in rl_history]
    online_fair = [float(x.get('online_fairness', 0.0)) for x in rl_history]

    ax.plot(episodes, _safe_series(avg_reward, 21), label='Avg Reward (smoothed)', linewidth=2)
    ax.plot(episodes, _safe_series(online_success, 21), label='Online Success (smoothed)', linewidth=2)
    ax.plot(episodes, _safe_series(online_div, 21), label='Online Diversity (smoothed)', linestyle='--')
    ax.plot(episodes, _safe_series(online_fair, 21), label='Online Fairness (smoothed)', linestyle='--')
    ax.set_title('RL Training Curves')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Metric')
    ax.grid(alpha=0.3)
    ax.legend(loc='best')


def _plot_rl_eval(ax, rl_eval_history: List[Dict]) -> None:
    if not rl_eval_history:
        ax.set_title('RL Validation OPE')
        ax.text(0.5, 0.5, 'No rl_eval_history in checkpoint', ha='center', va='center')
        ax.axis('off')
        return

    episodes = [int(x.get('episode', i + 1)) for i, x in enumerate(rl_eval_history)]
    dr = [float(x.get('dr', 0.0)) for x in rl_eval_history]
    dm = [float(x.get('dm', 0.0)) for x in rl_eval_history]
    ips = [float(x.get('ips', 0.0)) for x in rl_eval_history]

    ax.plot(episodes, dr, label='DR', linewidth=2)
    ax.plot(episodes, dm, label='DM', linewidth=2)
    ax.plot(episodes, ips, label='IPS', linestyle='--')
    ax.set_title('RL Validation Evaluation (OPE)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.grid(alpha=0.3)
    ax.legend(loc='best')


def _plot_test_eval(ax, eval_payload: Dict) -> None:
    test_eval = eval_payload.get('test_evaluation', eval_payload.get('test', {}))
    if not isinstance(test_eval, dict) or not test_eval:
        ax.set_title('Test Evaluation Snapshot')
        ax.text(0.5, 0.5, 'No test evaluation JSON found', ha='center', va='center')
        ax.axis('off')
        return

    rec = test_eval.get('recommendation_results', {})
    conv = test_eval.get('conversation_results', {})
    div = test_eval.get('diversity', {})
    fair = test_eval.get('fairness', {})
    transp = test_eval.get('transparency', {})

    labels = [
        'R@10', 'R@50', 'MRR@10', 'NDCG@10',
        'Dist-2', 'BLEU-2', 'SR@10',
        'ILD@10', 'CatCov@10', 'CalErr@10',
        'G@10', 'L@10', 'D@10',
        'Grounded', '1-Halluc', 'Trust'
    ]
    values = [
        float(rec.get('Recall@10', 0.0)),
        float(rec.get('Recall@50', 0.0)),
        float(rec.get('MRR@10', 0.0)),
        float(rec.get('NDCG@10', 0.0)),
        float(conv.get('Dist-2', 0.0)),
        float(conv.get('BLEU-2', 0.0)),
        float(conv.get('SR@10', 0.0)),
        float(div.get('ILD@10', 0.0)),
        float(div.get('CategoryCoverage@10', 0.0)),
        float(div.get('CalibrationError@10', 0.0)),
        float(fair.get('G@10', 0.0)),
        float(fair.get('L@10', 0.0)),
        float(fair.get('D@10', 0.0)),
        float(transp.get('groundedness_factual_consistency', 0.0)),
        1.0 - float(transp.get('deception_hallucination_rate', 0.0)),
        float(transp.get('trust_score', 0.0)),
    ]

    x = np.arange(len(labels))
    ax.bar(x, values, color='steelblue', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title('Testing / Evaluation Metrics Snapshot')
    ax.set_ylabel('Value')
    ax.grid(axis='y', alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description='Plot MO-CRS train/val/test curves from checkpoint + eval JSON')
    parser.add_argument('--config', type=str, default='config_thesis.yaml')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to .pt checkpoint containing train_stats')
    parser.add_argument('--eval_json', type=str, default=None, help='Path to evaluation JSON (test output)')
    parser.add_argument('--output', type=str, default='logs/training_validation_testing_plots.png')
    parser.add_argument('--dpi', type=int, default=160)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    checkpoint_path = _resolve_checkpoint(config, args.checkpoint)
    eval_json_path = _resolve_eval_json(config, args.eval_json)

    train_stats = _load_train_stats(checkpoint_path)
    eval_payload = _load_eval_payload(eval_json_path)

    supervised_history = train_stats.get('supervised_history', []) if isinstance(train_stats, dict) else []
    rl_history = train_stats.get('rl_history', []) if isinstance(train_stats, dict) else []
    rl_eval_history = train_stats.get('rl_eval_history', []) if isinstance(train_stats, dict) else []

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    _plot_supervised_history(axes[0, 0], supervised_history)
    _plot_rl_history(axes[0, 1], rl_history)
    _plot_rl_eval(axes[1, 0], rl_eval_history)
    _plot_test_eval(axes[1, 1], eval_payload)

    title_dataset = str(config.get('data', {}).get('dataset_name', 'Unknown'))
    fig.suptitle(f'MO-CRS Training / Validation / Testing Dashboard ({title_dataset})', fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=int(args.dpi))
    print(f'Saved plot: {os.path.abspath(args.output)}')
    print(f'Checkpoint used: {checkpoint_path}')
    print(f'Eval JSON used: {eval_json_path}')


if __name__ == '__main__':
    main()
