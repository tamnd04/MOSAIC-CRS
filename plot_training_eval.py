"""
Thesis-ready plotting utilities for MO-CRS training, validation, testing, and ablations.

This script improves the original 2x2 dashboard by:
  - prioritizing best_rl_model / best_rl test JSON resolution
  - separating metrics with incompatible scales
  - producing thesis-ready individual figures and summary tables
  - working for ReDial, INSPIRED, or any dataset with the same eval JSON schema

Examples:
  python plot_training_eval_thesis.py --config config_thesis.yaml \
    --checkpoint checkpoints/ReDial/best_rl_model.pt \
    --eval_json logs/ReDial_best_rl_test_eval.json \
    --output_dir logs/figures/ReDial

  python plot_training_eval_thesis.py --config config_inspired.yaml \
    --checkpoint checkpoints/INSPIRED/best_rl_model.pt \
    --eval_json logs/INSPIRED_best_rl_test_eval.json \
    --output_dir logs/figures/INSPIRED
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


# -----------------------------
# Loading / resolving utilities
# -----------------------------


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_series(values: Sequence[Any], window: int = 5) -> np.ndarray:
    arr = np.asarray([_as_float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return arr
    if window <= 1 or arr.size < window:
        return arr
    window = min(int(window), int(arr.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    # Edge padding avoids artificial drops at the beginning/end that happen with mode='same'.
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config is not a dictionary: {path}")
    return payload


def _load_json(path: str | Path) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _resolve_relative(config_path: str | Path, maybe_path: Optional[str]) -> Optional[str]:
    if not maybe_path:
        return maybe_path
    if os.path.isabs(maybe_path):
        return os.path.normpath(maybe_path)
    base = os.path.dirname(os.path.abspath(str(config_path)))
    return os.path.normpath(os.path.join(base, maybe_path))


def _resolve_checkpoint(config: Dict[str, Any], config_path: str | Path, checkpoint_arg: Optional[str]) -> str:
    if checkpoint_arg:
        return _resolve_relative(config_path, checkpoint_arg) or checkpoint_arg

    dataset = str(config.get("data", {}).get("dataset_name", "ReDial"))
    save_dir = _resolve_relative(config_path, str(config.get("logging", {}).get("save_dir", "./checkpoints")))
    assert save_dir is not None

    # Prefer final RL checkpoint for thesis plots.
    candidates = [
        os.path.join(save_dir, dataset, "best_rl_model.pt"),
        os.path.join(save_dir, dataset, "best_model.pt"),
        os.path.join(save_dir, "best_rl_model.pt"),
        os.path.join(save_dir, "best_model.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _resolve_eval_json(config: Dict[str, Any], config_path: str | Path, eval_arg: Optional[str]) -> str:
    if eval_arg:
        return _resolve_relative(config_path, eval_arg) or eval_arg

    dataset = str(config.get("data", {}).get("dataset_name", "ReDial"))
    log_dir = _resolve_relative(config_path, str(config.get("logging", {}).get("log_dir", "./logs")))
    assert log_dir is not None

    # Prefer final RL test evaluation before train-only JSONs.
    candidates = [
        os.path.join(log_dir, f"{dataset}_best_rl_test_eval.json"),
        os.path.join(log_dir, f"{dataset}_best_rl_test_eval_expl_quality.json"),
        os.path.join(log_dir, f"{dataset}_best_rl_test_eval_rank_action.json"),
        os.path.join(log_dir, f"{dataset}_best_model_test_eval.json"),
        os.path.join(log_dir, f"{dataset}_train_eval.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _load_train_stats(checkpoint_path: str) -> Dict[str, Any]:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        train_stats = checkpoint.get("train_stats", {})
        return train_stats if isinstance(train_stats, dict) else {}
    return {}


def _get_test_eval(eval_payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(eval_payload, dict):
        return {}
    if isinstance(eval_payload.get("test_evaluation"), dict):
        return eval_payload["test_evaluation"]
    if isinstance(eval_payload.get("test"), dict):
        return eval_payload["test"]
    return {}


def _get_ope_test(eval_payload: Dict[str, Any]) -> Dict[str, Any]:
    ope = eval_payload.get("ope", {}) if isinstance(eval_payload, dict) else {}
    if isinstance(ope, dict) and isinstance(ope.get("test"), dict):
        return ope["test"]
    if isinstance(eval_payload.get("ope_test"), dict):
        return eval_payload["ope_test"]
    return {}


# -----------------------------
# Plot helpers
# -----------------------------


def _finish_figure(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _empty_plot(message: str, title: str, output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.axis("off")
    _finish_figure(fig, output_path, dpi)


def _bar_plot(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
    ylim: Optional[Tuple[float, float]] = None,
    rotate: int = 30,
    annotate: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(max(7.5, len(labels) * 0.75), 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    if annotate:
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    _finish_figure(fig, output_path, dpi)


# -----------------------------
# Training curve plots
# -----------------------------


def plot_supervised_history(supervised_history: List[Dict[str, Any]], output_path: Path, dpi: int) -> None:
    if not supervised_history:
        _empty_plot("No supervised_history found in checkpoint.", "Supervised Pretraining", output_path, dpi)
        return

    epochs = [int(x.get("epoch", i + 1)) for i, x in enumerate(supervised_history)]
    train_loss = [_as_float(x.get("train_total_loss")) for x in supervised_history]
    val_loss = [_as_float(x.get("val_total_loss")) for x in supervised_history]
    intent_acc = [_as_float(x.get("val_intent_accuracy")) for x in supervised_history]
    sentiment_acc = [_as_float(x.get("val_sentiment_accuracy")) for x in supervised_history]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_loss.plot(epochs, train_loss, marker="o", label="Train loss")
    ax_loss.plot(epochs, val_loss, marker="o", label="Validation loss")
    ax_loss.set_title("Supervised Pretraining Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.25)
    ax_loss.legend()

    ax_acc.plot(epochs, intent_acc, marker="o", label="Intent accuracy")
    ax_acc.plot(epochs, sentiment_acc, marker="o", label="Sentiment accuracy")
    ax_acc.set_title("Validation Classification Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.grid(alpha=0.25)
    ax_acc.legend()
    _finish_figure(fig, output_path, dpi)


def plot_rl_history(rl_history: List[Dict[str, Any]], output_path: Path, dpi: int) -> None:
    if not rl_history:
        _empty_plot("No rl_history found in checkpoint.", "RL Fine-tuning", output_path, dpi)
        return

    episodes = [int(x.get("episode", i + 1)) for i, x in enumerate(rl_history)]
    avg_reward = [_as_float(x.get("avg_reward")) for x in rl_history]
    online_success = [_as_float(x.get("online_success")) for x in rl_history]
    avg_turns = [_as_float(x.get("avg_turns", x.get("turns", 0.0))) for x in rl_history]
    ask_rate = [_as_float(x.get("ask_rate", x.get("action_ask", 0.0))) for x in rl_history]
    rec_rate = [_as_float(x.get("recommend_rate", x.get("action_recommend", 0.0))) for x in rl_history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(episodes, _safe_series(avg_reward, 11), label="Avg reward")
    axes[0].set_title("RL Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.25)

    axes[1].plot(episodes, _safe_series(online_success, 11), label="Online success")
    axes[1].set_title("Online Success During RL")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.25)

    if any(v > 0 for v in avg_turns):
        axes[2].plot(episodes, _safe_series(avg_turns, 11), label="Avg turns")
        axes[2].set_ylabel("Turns")
        axes[2].set_title("Dialogue Length During RL")
    elif any(v > 0 for v in ask_rate) or any(v > 0 for v in rec_rate):
        axes[2].plot(episodes, _safe_series(ask_rate, 11), label="Ask rate")
        axes[2].plot(episodes, _safe_series(rec_rate, 11), label="Recommend rate")
        axes[2].set_ylim(0.0, 1.05)
        axes[2].set_ylabel("Action rate")
        axes[2].set_title("Action Balance During RL")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "No turn/action series found", ha="center", va="center")
        axes[2].set_title("Dialogue / Action Diagnostics")
    axes[2].set_xlabel("Episode")
    axes[2].grid(alpha=0.25)

    _finish_figure(fig, output_path, dpi)


def plot_rl_ope_history(rl_eval_history: List[Dict[str, Any]], output_path: Path, dpi: int) -> None:
    if not rl_eval_history:
        _empty_plot("No rl_eval_history found in checkpoint.", "Validation OPE During RL", output_path, dpi)
        return

    episodes = [int(x.get("episode", i + 1)) for i, x in enumerate(rl_eval_history)]
    dr = [_as_float(x.get("dr")) for x in rl_eval_history]
    dm = [_as_float(x.get("dm")) for x in rl_eval_history]
    ips = [_as_float(x.get("ips")) for x in rl_eval_history]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(episodes, dr, marker="o", label="DR")
    ax.plot(episodes, dm, marker="o", label="DM")
    ax.plot(episodes, ips, marker="o", linestyle="--", label="IPS")
    ax.set_title("Validation Off-Policy Evaluation During RL")
    ax.set_xlabel("Episode")
    ax.set_ylabel("OPE score")
    ax.grid(alpha=0.25)
    ax.legend()
    _finish_figure(fig, output_path, dpi)


# -----------------------------
# Final evaluation plots
# -----------------------------


def plot_recommendation_metrics(test_eval: Dict[str, Any], output_path: Path, dpi: int) -> None:
    rec = test_eval.get("recommendation_results", {}) if isinstance(test_eval, dict) else {}
    if not rec:
        _empty_plot("No recommendation_results found.", "Recommendation Metrics", output_path, dpi)
        return
    labels = ["Recall@10", "Recall@50", "MRR@10", "MRR@50", "NDCG@10", "NDCG@50"]
    values = [_as_float(rec.get(k)) for k in labels]
    _bar_plot(labels, values, "Final Recommendation Quality", "Score", output_path, dpi, ylim=(0, 1.05))


def plot_conversation_metrics(test_eval: Dict[str, Any], output_path: Path, dpi: int) -> None:
    conv = test_eval.get("conversation_results", {}) if isinstance(test_eval, dict) else {}
    rollout = test_eval.get("online_rollout", {}) if isinstance(test_eval, dict) else {}
    if not conv and not rollout:
        _empty_plot("No conversation_results or online_rollout found.", "Conversation Metrics", output_path, dpi)
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    sr_labels = ["Success", "SR@5", "SR@10", "SR@20"]
    sr_values = [
        _as_float(conv.get("success_rate")),
        _as_float(conv.get("SR@5")),
        _as_float(conv.get("SR@10")),
        _as_float(conv.get("SR@20")),
    ]
    x = np.arange(len(sr_labels))
    axes[0].bar(x, sr_values)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sr_labels, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Conversation Success")
    axes[0].set_ylabel("Rate")
    axes[0].grid(axis="y", alpha=0.25)

    at = _as_float(conv.get("AT"))
    axes[1].bar(["Avg turns"], [at])
    axes[1].set_title("Dialogue Efficiency")
    axes[1].set_ylabel("Turns")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].text(0, at, f"{at:.2f}", ha="center", va="bottom", fontsize=9)

    action_ratios = rollout.get("action_ratios", {}) if isinstance(rollout, dict) else {}
    action_labels = ["ask", "recommend", "clarify", "end"]
    action_values = [_as_float(action_ratios.get(k)) for k in action_labels]
    x2 = np.arange(len(action_labels))
    axes[2].bar(x2, action_values)
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels([x.title() for x in action_labels], rotation=25, ha="right")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Policy Action Balance")
    axes[2].set_ylabel("Action ratio")
    axes[2].grid(axis="y", alpha=0.25)

    _finish_figure(fig, output_path, dpi)


def plot_diversity_fairness(test_eval: Dict[str, Any], output_path: Path, dpi: int) -> None:
    div = test_eval.get("diversity", {}) if isinstance(test_eval, dict) else {}
    fair = test_eval.get("fairness", {}) if isinstance(test_eval, dict) else {}
    if not div and not fair:
        _empty_plot("No diversity/fairness metrics found.", "Diversity and Fairness", output_path, dpi)
        return

    ks = [5, 10, 20]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    axes[0, 0].plot(ks, [_as_float(div.get(f"ILD@{k}")) for k in ks], marker="o", label="ILD")
    axes[0, 0].plot(ks, [_as_float(div.get(f"GenreCoverage@{k}")) for k in ks], marker="o", label="Genre coverage")
    axes[0, 0].plot(ks, [_as_float(div.get(f"CategoryCoverage@{k}")) for k in ks], marker="o", label="Category coverage")
    axes[0, 0].set_title("Diversity / Coverage by Cutoff")
    axes[0, 0].set_xlabel("K")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(ks, [_as_float(div.get(f"CalibrationError@{k}")) for k in ks], marker="o")
    axes[0, 1].set_title("Calibration Error by Cutoff")
    axes[0, 1].set_xlabel("K")
    axes[0, 1].set_ylabel("Error (lower is better)")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(ks, [_as_float(fair.get(f"HeadShare@{k}")) for k in ks], marker="o", label="Head share")
    axes[1, 0].plot(ks, [_as_float(fair.get(f"TailShare@{k}")) for k in ks], marker="o", label="Tail share")
    axes[1, 0].plot(ks, [_as_float(fair.get(f"Entropy@{k}")) for k in ks], marker="o", label="Exposure entropy")
    axes[1, 0].set_title("Exposure Shares / Entropy")
    axes[1, 0].set_xlabel("K")
    axes[1, 0].set_ylabel("Share / entropy")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(ks, [_as_float(fair.get(f"G@{k}")) for k in ks], marker="o", label="Gini G")
    axes[1, 1].plot(ks, [_as_float(fair.get(f"L@{k}")) for k in ks], marker="o", label="KL L")
    axes[1, 1].plot(ks, [_as_float(fair.get(f"D@{k}")) for k in ks], marker="o", label="Tail-head D")
    axes[1, 1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1, 1].set_title("Fairness Concentration Metrics")
    axes[1, 1].set_xlabel("K")
    axes[1, 1].set_ylabel("Metric value")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    _finish_figure(fig, output_path, dpi)


def plot_transparency_metrics(test_eval: Dict[str, Any], output_path: Path, dpi: int) -> None:
    trans = test_eval.get("transparency", {}) if isinstance(test_eval, dict) else {}
    if not trans:
        _empty_plot("No transparency metrics found.", "Explanation / Transparency Metrics", output_path, dpi)
        return

    labels = [
        "Grounded", "1-Hallucination", "Persuasive", "Transparent", "Trust", "Useful",
    ]
    values = [
        _as_float(trans.get("groundedness_factual_consistency")),
        1.0 - _as_float(trans.get("deception_hallucination_rate")),
        _as_float(trans.get("persuasiveness_score")),
        _as_float(trans.get("transparency_score")),
        _as_float(trans.get("trust_score")),
        _as_float(trans.get("usefulness_score")),
    ]
    _bar_plot(labels, values, "Explanation and Transparency Quality", "Score", output_path, dpi, ylim=(0, 1.05))


def plot_ope_test(eval_payload: Dict[str, Any], output_path: Path, dpi: int) -> None:
    ope = _get_ope_test(eval_payload)
    if not ope:
        _empty_plot("No OPE test metrics found.", "Test Off-Policy Evaluation", output_path, dpi)
        return

    labels = ["IPS", "SNIPS", "DR", "DM", "Logged reward"]
    values = [
        _as_float(ope.get("ips")),
        _as_float(ope.get("snips")),
        _as_float(ope.get("dr")),
        _as_float(ope.get("dm")),
        _as_float(ope.get("logged_reward_mean")),
    ]
    _bar_plot(labels, values, "Test Off-Policy Evaluation", "Score", output_path, dpi, ylim=(0, max(1.05, max(values) * 1.2)))


# -----------------------------
# Tables
# -----------------------------


def _collect_summary_rows(eval_payload: Dict[str, Any]) -> List[Tuple[str, str, float]]:
    test_eval = _get_test_eval(eval_payload)
    ope = _get_ope_test(eval_payload)
    rec = test_eval.get("recommendation_results", {}) if isinstance(test_eval, dict) else {}
    conv = test_eval.get("conversation_results", {}) if isinstance(test_eval, dict) else {}
    div = test_eval.get("diversity", {}) if isinstance(test_eval, dict) else {}
    fair = test_eval.get("fairness", {}) if isinstance(test_eval, dict) else {}
    trans = test_eval.get("transparency", {}) if isinstance(test_eval, dict) else {}
    rollout = test_eval.get("online_rollout", {}) if isinstance(test_eval, dict) else {}
    action = rollout.get("action_ratios", {}) if isinstance(rollout, dict) else {}

    rows: List[Tuple[str, str, float]] = []
    for k in ["ips", "snips", "dr", "dm", "logged_reward_mean"]:
        if k in ope:
            rows.append(("OPE", k, _as_float(ope.get(k))))
    for k in ["Recall@10", "Recall@50", "MRR@10", "MRR@50", "NDCG@10", "NDCG@50"]:
        if k in rec:
            rows.append(("Recommendation", k, _as_float(rec.get(k))))
    for k in ["success_rate", "SR@5", "SR@10", "SR@20", "AT", "Dist-2", "Dist-3", "BLEU-2", "BLEU-3"]:
        if k in conv:
            rows.append(("Conversation", k, _as_float(conv.get(k))))
    for k in ["ask", "recommend", "clarify", "end"]:
        if k in action:
            rows.append(("Action", f"{k}_rate", _as_float(action.get(k))))
    for k in ["ILD@10", "GenreCoverage@10", "CategoryCoverage@10", "CalibrationError@10"]:
        if k in div:
            rows.append(("Diversity", k, _as_float(div.get(k))))
    for k in ["A@10", "G@10", "L@10", "D@10", "HeadShare@10", "TailShare@10", "Entropy@10"]:
        if k in fair:
            rows.append(("Fairness", k, _as_float(fair.get(k))))
    for k in [
        "groundedness_factual_consistency",
        "deception_hallucination_rate",
        "persuasiveness_score",
        "transparency_score",
        "trust_score",
        "usefulness_score",
    ]:
        if k in trans:
            rows.append(("Transparency", k, _as_float(trans.get(k))))
    return rows


def write_summary_tables(eval_payload: Dict[str, Any], output_dir: Path) -> None:
    rows = _collect_summary_rows(eval_payload)
    if not rows:
        return

    csv_path = output_dir / "thesis_metrics_summary.csv"
    md_path = output_dir / "thesis_metrics_summary.md"
    output_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Group", "Metric", "Value"])
        for group, metric, value in rows:
            writer.writerow([group, metric, f"{value:.6f}"])

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Thesis Metrics Summary\n\n")
        f.write("| Group | Metric | Value |\n")
        f.write("|---|---:|---:|\n")
        for group, metric, value in rows:
            f.write(f"| {group} | {metric} | {value:.4f} |\n")


# -----------------------------
# Dashboard
# -----------------------------


def plot_dashboard(train_stats: Dict[str, Any], eval_payload: Dict[str, Any], dataset: str, output_path: Path, dpi: int) -> None:
    """Compact overview. Use for appendix/debug; individual figures are better for the main thesis."""
    supervised = train_stats.get("supervised_history", []) if isinstance(train_stats, dict) else []
    rl_hist = train_stats.get("rl_history", []) if isinstance(train_stats, dict) else []
    rl_eval = train_stats.get("rl_eval_history", []) if isinstance(train_stats, dict) else []
    test_eval = _get_test_eval(eval_payload)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    if supervised:
        epochs = [int(x.get("epoch", i + 1)) for i, x in enumerate(supervised)]
        axes[0, 0].plot(epochs, [_as_float(x.get("train_total_loss")) for x in supervised], label="Train")
        axes[0, 0].plot(epochs, [_as_float(x.get("val_total_loss")) for x in supervised], label="Val")
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, "No supervised history", ha="center", va="center")
    axes[0, 0].set_title("Pretraining Loss")
    axes[0, 0].grid(alpha=0.25)

    if rl_hist:
        episodes = [int(x.get("episode", i + 1)) for i, x in enumerate(rl_hist)]
        axes[0, 1].plot(episodes, _safe_series([x.get("avg_reward", 0.0) for x in rl_hist], 11), label="Reward")
        axes[0, 1].plot(episodes, _safe_series([x.get("online_success", 0.0) for x in rl_hist], 11), label="Success")
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, "No RL history", ha="center", va="center")
    axes[0, 1].set_title("RL Progress")
    axes[0, 1].grid(alpha=0.25)

    if rl_eval:
        episodes = [int(x.get("episode", i + 1)) for i, x in enumerate(rl_eval)]
        axes[0, 2].plot(episodes, [_as_float(x.get("dr")) for x in rl_eval], marker="o", label="DR")
        axes[0, 2].plot(episodes, [_as_float(x.get("dm")) for x in rl_eval], marker="o", label="DM")
        axes[0, 2].legend()
    else:
        axes[0, 2].text(0.5, 0.5, "No RL OPE history", ha="center", va="center")
    axes[0, 2].set_title("Validation OPE")
    axes[0, 2].grid(alpha=0.25)

    rec = test_eval.get("recommendation_results", {}) if isinstance(test_eval, dict) else {}
    conv = test_eval.get("conversation_results", {}) if isinstance(test_eval, dict) else {}
    fair = test_eval.get("fairness", {}) if isinstance(test_eval, dict) else {}
    trans = test_eval.get("transparency", {}) if isinstance(test_eval, dict) else {}

    labels = ["R@10", "R@50", "MRR@10", "NDCG@10"]
    values = [_as_float(rec.get("Recall@10")), _as_float(rec.get("Recall@50")), _as_float(rec.get("MRR@10")), _as_float(rec.get("NDCG@10"))]
    axes[1, 0].bar(labels, values)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("Recommendation")
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].grid(axis="y", alpha=0.25)

    labels = ["Success", "SR@10", "Ask", "Recommend"]
    rollout = test_eval.get("online_rollout", {}) if isinstance(test_eval, dict) else {}
    action = rollout.get("action_ratios", {}) if isinstance(rollout, dict) else {}
    values = [_as_float(conv.get("success_rate")), _as_float(conv.get("SR@10")), _as_float(action.get("ask")), _as_float(action.get("recommend"))]
    axes[1, 1].bar(labels, values)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Conversation / Policy")
    axes[1, 1].tick_params(axis="x", rotation=25)
    axes[1, 1].grid(axis="y", alpha=0.25)

    labels = ["Head@10", "Tail@10", "Entropy@10", "Useful"]
    values = [_as_float(fair.get("HeadShare@10")), _as_float(fair.get("TailShare@10")), _as_float(fair.get("Entropy@10")), _as_float(trans.get("usefulness_score"))]
    axes[1, 2].bar(labels, values)
    axes[1, 2].set_ylim(0, 1.05)
    axes[1, 2].set_title("Fairness / Explanation")
    axes[1, 2].tick_params(axis="x", rotation=25)
    axes[1, 2].grid(axis="y", alpha=0.25)

    fig.suptitle(f"MO-CRS Thesis Dashboard ({dataset})", fontsize=14)
    _finish_figure(fig, output_path, dpi)


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-ready MO-CRS plots and summary tables.")
    parser.add_argument("--config", type=str, default="config_thesis.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint containing train_stats")
    parser.add_argument("--eval_json", type=str, default=None, help="Path to full test evaluation JSON")
    parser.add_argument("--output_dir", type=str, default="logs/thesis_plots")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip_dashboard", action="store_true", help="Do not create compact dashboard figure")
    args = parser.parse_args()

    config = _load_yaml(args.config)
    dataset = str(config.get("data", {}).get("dataset_name", "Unknown"))
    checkpoint_path = _resolve_checkpoint(config, args.config, args.checkpoint)
    eval_json_path = _resolve_eval_json(config, args.config, args.eval_json)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_stats = _load_train_stats(checkpoint_path)
    eval_payload = _load_json(eval_json_path)
    test_eval = _get_test_eval(eval_payload)

    supervised_history = train_stats.get("supervised_history", []) if isinstance(train_stats, dict) else []
    rl_history = train_stats.get("rl_history", []) if isinstance(train_stats, dict) else []
    rl_eval_history = train_stats.get("rl_eval_history", []) if isinstance(train_stats, dict) else []

    plot_supervised_history(supervised_history, output_dir / "01_supervised_pretraining.png", args.dpi)
    plot_rl_history(rl_history, output_dir / "02_rl_training_curves.png", args.dpi)
    plot_rl_ope_history(rl_eval_history, output_dir / "03_validation_ope.png", args.dpi)
    plot_ope_test(eval_payload, output_dir / "04_test_ope.png", args.dpi)
    plot_recommendation_metrics(test_eval, output_dir / "05_recommendation_metrics.png", args.dpi)
    plot_conversation_metrics(test_eval, output_dir / "06_conversation_policy_metrics.png", args.dpi)
    plot_diversity_fairness(test_eval, output_dir / "07_diversity_fairness_metrics.png", args.dpi)
    plot_transparency_metrics(test_eval, output_dir / "08_transparency_metrics.png", args.dpi)
    write_summary_tables(eval_payload, output_dir)

    if not args.skip_dashboard:
        plot_dashboard(train_stats, eval_payload, dataset, output_dir / "00_dashboard_overview.png", args.dpi)

    print(f"Dataset: {dataset}")
    print(f"Checkpoint used: {checkpoint_path}")
    print(f"Eval JSON used: {eval_json_path}")
    print(f"Saved thesis plots to: {output_dir.resolve()}")
    print("Generated files:")
    for p in sorted(output_dir.glob("*")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
