"""
Generate dataset EDA tables and figures for MOSAIC-CRS thesis reporting.

Run from source_code/:

    python generate_dataset_eda.py --data_root data --datasets ReDial INSPIRED --out_dir logs/eda

Expected project layout:

    data/ReDial/train_data.json
    data/ReDial/val_data.json
    data/ReDial/test_data.json
    data/ReDial/item_catalog.json

    data/INSPIRED/train_data.json
    data/INSPIRED/val_data.json
    data/INSPIRED/test_data.json
    data/INSPIRED/item_catalog.json

Outputs:
    logs/eda/eda_summary_table.csv
    logs/eda/eda_summary_table.tex
    logs/eda/eda_latex_snippet.tex
    logs/eda/<dataset>_*.png
    logs/eda/<dataset>_eda.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_num(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def split_genres(item: Dict[str, Any]) -> List[str]:
    genres = item.get("genres")
    if isinstance(genres, list) and genres:
        return [str(g).strip() for g in genres if str(g).strip()]
    genre = item.get("genre", "")
    if isinstance(genre, str):
        return [g.strip() for g in genre.replace(",", "|").split("|") if g.strip()]
    return []


def get_turn_text(turn: Dict[str, Any]) -> str:
    return str(turn.get("user_utterance") or turn.get("text") or turn.get("utterance") or "")


def conversation_stats(split_name: str, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
    turn_counts = []
    user_turn_counts = []
    system_turn_counts = []
    mentioned_counts = []
    accepted_counts = []
    token_counts = []
    success_flags = []
    unique_mentioned = Counter()
    unique_accepted = Counter()

    for conv in conversations:
        turns = conv.get("turns", []) or []
        turn_counts.append(len(turns))
        user_turn = 0
        system_turn = 0
        conv_tokens = 0
        for turn in turns:
            speaker = str(turn.get("speaker", "")).lower()
            if speaker == "user":
                user_turn += 1
            elif speaker == "system":
                system_turn += 1
            conv_tokens += len(get_turn_text(turn).split())
        user_turn_counts.append(user_turn)
        system_turn_counts.append(system_turn)
        token_counts.append(conv_tokens)

        mentioned = conv.get("mentioned_items", []) or []
        accepted = conv.get("accepted_items", []) or []
        mentioned_counts.append(len(mentioned))
        accepted_counts.append(len(accepted))
        success_flags.append(bool(conv.get("success", False)))
        unique_mentioned.update(map(str, mentioned))
        unique_accepted.update(map(str, accepted))

    def mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def median(xs: List[float]) -> float:
        return float(np.median(xs)) if xs else 0.0

    return {
        "split": split_name,
        "num_conversations": len(conversations),
        "success_rate": mean(success_flags),
        "avg_turns": mean(turn_counts),
        "median_turns": median(turn_counts),
        "min_turns": int(min(turn_counts)) if turn_counts else 0,
        "max_turns": int(max(turn_counts)) if turn_counts else 0,
        "avg_user_turns": mean(user_turn_counts),
        "avg_system_turns": mean(system_turn_counts),
        "avg_tokens_per_dialogue": mean(token_counts),
        "avg_mentions_per_dialogue": mean(mentioned_counts),
        "avg_accepted_items_per_dialogue": mean(accepted_counts),
        "unique_mentioned_items": len(unique_mentioned),
        "unique_accepted_items": len(unique_accepted),
        "turn_counts": turn_counts,
        "mentioned_counts": mentioned_counts,
        "success_flags": success_flags,
    }


def catalog_stats(catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    genres = Counter()
    categories = Counter()
    mentions = []
    ratings = []
    years = []

    for item_id, item in catalog.items():
        gs = split_genres(item)
        if not gs:
            gs = ["Unknown"]
        genres.update(gs)
        categories.update([str(item.get("category") or "Unknown")])
        mentions.append(safe_num(item.get("mentions", 0), 0.0))
        rating = safe_num(item.get("rating", 0), 0.0)
        if rating > 0:
            ratings.append(rating)
        year = item.get("year")
        try:
            y = int(str(year)[:4])
            if 1800 <= y <= 2100:
                years.append(y)
        except Exception:
            pass

    mentions_arr = np.array(mentions, dtype=float) if mentions else np.array([], dtype=float)
    if len(mentions_arr) > 0:
        q80 = float(np.quantile(mentions_arr, 0.80))
        q50 = float(np.quantile(mentions_arr, 0.50))
        head = int(np.sum(mentions_arr >= q80))
        mid = int(np.sum((mentions_arr < q80) & (mentions_arr >= q50)))
        tail = int(np.sum(mentions_arr < q50))
    else:
        q80 = q50 = 0.0
        head = mid = tail = 0

    return {
        "num_items": len(catalog),
        "num_unique_genres": len(genres),
        "num_unique_categories": len(categories),
        "avg_mentions": float(np.mean(mentions_arr)) if len(mentions_arr) else 0.0,
        "median_mentions": float(np.median(mentions_arr)) if len(mentions_arr) else 0.0,
        "max_mentions": float(np.max(mentions_arr)) if len(mentions_arr) else 0.0,
        "head_items_top20pct": head,
        "mid_items_50_to_80pct": mid,
        "tail_items_bottom50pct": tail,
        "avg_rating_nonzero": float(np.mean(ratings)) if ratings else 0.0,
        "median_year": float(np.median(years)) if years else 0.0,
        "genre_counter": dict(genres.most_common()),
        "category_counter": dict(categories.most_common()),
        "mentions": mentions,
        "years": years,
    }


def save_bar(counter: Dict[str, int], title: str, xlabel: str, ylabel: str, out_path: Path, top_n: int = 12) -> None:
    items = Counter(counter).most_common(top_n)
    if not items:
        return
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def remove_unknown_labels(counter: Counter) -> Counter:
    """Remove Unknown/empty labels from EDA display counters.

    This is mainly useful for thesis plots: unknown genres can dominate the
    visual distribution, especially for INSPIRED, and make the actual genre
    pattern harder to read.
    """
    filtered = Counter()
    for key, value in counter.items():
        label = str(key).strip()
        if not label:
            continue
        if label.lower() in {"unknown", "unk", "nan", "none", "null", "n/a", "na"}:
            continue
        filtered[label] += value
    return filtered


def save_hist(values: List[float], title: str, xlabel: str, ylabel: str, out_path: Path, bins: int = 30, log_y: bool = False) -> None:
    if not values:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_split_bar(split_stats: List[Dict[str, Any]], dataset: str, out_path: Path) -> None:
    labels = [s["split"] for s in split_stats]
    vals = [s["num_conversations"] for s in split_stats]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals)
    ax.set_title(f"{dataset}: conversations by split")
    ax.set_ylabel("Number of conversations")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fmt(x: Any, nd: int = 2) -> str:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return "--"
        if abs(v - int(v)) < 1e-9:
            return str(int(v))
        return f"{v:.{nd}f}"
    except Exception:
        return str(x)


def analyze_dataset(data_root: Path, dataset: str, out_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ds_dir = data_root / dataset
    files = {
        "train": ds_dir / "train_data.json",
        "val": ds_dir / "val_data.json",
        "test": ds_dir / "test_data.json",
        "catalog": ds_dir / "item_catalog.json",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files for {dataset}: {missing}")

    splits = {name: load_json(path) for name, path in files.items() if name != "catalog"}
    catalog = load_json(files["catalog"])
    split_stats = [conversation_stats(name, convs) for name, convs in splits.items()]
    cat_stats = catalog_stats(catalog)

    all_convs = []
    for convs in splits.values():
        all_convs.extend(convs)
    all_stats = conversation_stats("all", all_convs)

    # For reporting plots, remove the Unknown genre label for INSPIRED.
    # The raw INSPIRED movie database can contain missing genre values, and an
    # Unknown bar can dominate the genre EDA figure. The raw counter is still
    # preserved in catalog_stats inside the JSON file.
    display_genre_counter = Counter(cat_stats["genre_counter"])
    if dataset.lower() == "inspired":
        display_genre_counter = remove_unknown_labels(display_genre_counter)

    summary = {
        "dataset": dataset,
        "split_stats": split_stats,
        "overall_conversation_stats": {k: v for k, v in all_stats.items() if not isinstance(v, list)},
        "catalog_stats": {k: v for k, v in cat_stats.items() if not isinstance(v, (list, dict))},
        "top_genres": display_genre_counter.most_common(10),
        "top_categories": Counter(cat_stats["category_counter"]).most_common(10),
        "note": "For INSPIRED genre EDA plots, Unknown genre labels are removed from the displayed top-genre distribution.",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{dataset}_eda.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    save_split_bar(split_stats, dataset, out_dir / f"{dataset}_conversation_splits.png")
    save_hist(all_stats["turn_counts"], f"{dataset}: dialogue length distribution", "Turns per conversation", "Conversations", out_dir / f"{dataset}_turn_distribution.png", bins=30)
    save_hist(all_stats["mentioned_counts"], f"{dataset}: mentioned items per dialogue", "Mentioned items", "Conversations", out_dir / f"{dataset}_mentioned_items_distribution.png", bins=30)
    save_bar(display_genre_counter, f"{dataset}: top genres", "Genre", "Items", out_dir / f"{dataset}_top_genres.png", top_n=12)
    save_bar(cat_stats["category_counter"], f"{dataset}: top categories", "Category", "Items", out_dir / f"{dataset}_top_categories.png", top_n=12)
    save_hist(cat_stats["mentions"], f"{dataset}: item popularity distribution", "Mention count", "Items", out_dir / f"{dataset}_popularity_distribution.png", bins=40, log_y=True)

    row = {
        "Dataset": dataset,
        "Train": next(s["num_conversations"] for s in split_stats if s["split"] == "train"),
        "Val": next(s["num_conversations"] for s in split_stats if s["split"] == "val"),
        "Test": next(s["num_conversations"] for s in split_stats if s["split"] == "test"),
        "Total Dialogues": all_stats["num_conversations"],
        "Avg. Turns": all_stats["avg_turns"],
        "Median Turns": all_stats["median_turns"],
        "Success Rate": all_stats["success_rate"],
        "Avg. Mentions": all_stats["avg_mentions_per_dialogue"],
        "Unique Mentioned Items": all_stats["unique_mentioned_items"],
        "Catalog Items": cat_stats["num_items"],
        "Genres": cat_stats["num_unique_genres"],
        "Categories": cat_stats["num_unique_categories"],
        "Avg. Item Mentions": cat_stats["avg_mentions"],
        "Median Item Mentions": cat_stats["median_mentions"],
        "Max Item Mentions": cat_stats["max_mentions"],
        "Head Items": cat_stats["head_items_top20pct"],
        "Mid Items": cat_stats["mid_items_50_to_80pct"],
        "Tail Items": cat_stats["tail_items_bottom50pct"],
    }
    return summary, row


def build_latex_summary(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    df = pd.DataFrame(rows)
    df_out = df.copy()
    for col in df_out.columns:
        if col != "Dataset":
            df_out[col] = df_out[col].apply(lambda x: fmt(x, 3 if isinstance(x, float) and abs(x) < 1 else 2))
    df_out.to_csv(out_dir / "eda_summary_table.csv", index=False)

    small_cols = ["Dataset", "Train", "Val", "Test", "Catalog Items", "Avg. Turns", "Success Rate", "Avg. Mentions", "Genres", "Categories"]
    tex = df_out[small_cols].to_latex(index=False, escape=False, caption="EDA summary of ReDial and INSPIRED after preprocessing.", label="tab:dataset-eda-summary")
    (out_dir / "eda_summary_table.tex").write_text(tex, encoding="utf-8")

    lines = [
        r"\section{Exploratory Data Analysis}",
        "",
        r"This section reports descriptive statistics of the two processed datasets used in this thesis. The analysis focuses on split sizes, dialogue length, item mentions, item metadata, and catalog popularity distribution. These statistics are important because they indicate the scale of each dataset and reveal potential popularity imbalance that motivates diversity- and fairness-aware recommendation.",
        "",
        r"\input{eda_summary_table.tex}",
        "",
    ]
    for row in rows:
        ds = row["Dataset"]
        lines.extend([
            rf"\subsection{{{ds}}}",
            rf"The processed {ds} data contains {fmt(row['Total Dialogues'])} conversations across train, validation, and test splits, with {fmt(row['Catalog Items'])} catalog items. The average dialogue length is {fmt(row['Avg. Turns'])} turns, and each dialogue mentions {fmt(row['Avg. Mentions'])} items on average.",
            "",
            r"\begin{figure}[htbp]",
            r"    \centering",
            rf"    \includegraphics[width=0.48\textwidth]{{eda/{ds}_turn_distribution.png}}",
            rf"    \includegraphics[width=0.48\textwidth]{{eda/{ds}_top_genres.png}}",
            rf"    \caption{{Dialogue length distribution and top genres in {ds}.}}",
            rf"    \label{{fig:{ds.lower()}-eda-turn-genre}}",
            r"\end{figure}",
            "",
            r"\begin{figure}[htbp]",
            r"    \centering",
            rf"    \includegraphics[width=0.48\textwidth]{{eda/{ds}_conversation_splits.png}}",
            rf"    \includegraphics[width=0.48\textwidth]{{eda/{ds}_popularity_distribution.png}}",
            rf"    \caption{{Conversation split sizes and item popularity distribution in {ds}.}}",
            rf"    \label{{fig:{ds.lower()}-eda-split-popularity}}",
            r"\end{figure}",
            "",
        ])
    (out_dir / "eda_latex_snippet.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data", help="Root folder containing dataset folders")
    parser.add_argument("--datasets", nargs="+", default=["ReDial", "INSPIRED"])
    parser.add_argument("--out_dir", default="logs/eda")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    rows = []
    for ds in args.datasets:
        print(f"Analyzing {ds}...")
        _, row = analyze_dataset(data_root, ds, out_dir)
        rows.append(row)
    build_latex_summary(rows, out_dir)
    print(f"EDA complete. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
