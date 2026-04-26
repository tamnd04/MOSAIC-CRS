"""Rebuild ReDial dataset artifacts for MO-CRS.

This converter reads:
- ReDial raw JSONL dialogues (train/test)
- ReDial movie CSV with enriched metadata columns (mentions, genre, category)

It writes:
- train_data.json
- test_data_full.json
- val_data.json
- test_data.json
- train_data_full.json
- item_catalog.json
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


MOVIE_ID_PATTERN = re.compile(r"@(\d+)")
YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")


def infer_intent(utterance: str) -> str:
    text = utterance.lower()
    if any(token in text for token in ["recommend", "suggest", "watch", "try"]):
        return "recommend"
    if any(token in text for token in ["like", "love", "prefer", "favorite"]):
        return "provide_preference"
    if any(token in text for token in ["?", "what", "which", "how", "why", "when", "where"]):
        return "ask_question"
    if any(token in text for token in ["thanks", "thank you", "bye", "goodbye"]):
        return "goodbye"
    return "general"


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def normalize_utterance(text: str) -> str:
    return MOVIE_ID_PATTERN.sub(r"\1", text).strip()


def extract_movie_mentions(text: str) -> List[str]:
    return MOVIE_ID_PATTERN.findall(text)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_liked_items(question_blob: Any) -> List[str]:
    liked: List[str] = []

    if isinstance(question_blob, dict):
        for item_id, flags in question_blob.items():
            if isinstance(flags, dict) and int(flags.get("liked", 0)) == 1:
                liked.append(str(item_id))
    elif isinstance(question_blob, list):
        for entry in question_blob:
            if not isinstance(entry, dict):
                continue
            item_id = entry.get("movieId") or entry.get("id")
            liked_flag = entry.get("liked", 0)
            if item_id is not None and int(liked_flag) == 1:
                liked.append(str(item_id))

    return liked


def convert_dialogue(raw: Dict[str, Any]) -> Dict[str, Any]:
    messages = raw.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 2:
        return {}

    initiator_id = str(raw.get("initiatorWorkerId", "unknown"))
    turns: List[Dict[str, Any]] = []
    conversation_mentions: List[str] = []

    for message in messages:
        if isinstance(message, dict):
            raw_text = str(message.get("text", ""))
            sender_id = str(message.get("senderWorkerId", ""))
        elif isinstance(message, (list, tuple)):
            raw_text = str(message[1]) if len(message) > 1 else ""
            sender_id = str(message[0]) if len(message) > 0 else ""
        else:
            continue

        raw_text = raw_text.strip()
        if not raw_text:
            continue

        mentioned = extract_movie_mentions(raw_text)
        cleaned = normalize_utterance(raw_text)
        speaker = "user" if sender_id == initiator_id else "system"

        turns.append({
            "user_utterance": cleaned,
            "speaker": speaker,
            "intent": infer_intent(cleaned),
            "items_mentioned": mentioned,
        })
        conversation_mentions.extend(mentioned)

    if len(turns) < 2:
        return {}

    accepted_items = extract_liked_items(raw.get("respondentQuestions", {}))
    accepted_items.extend(extract_liked_items(raw.get("initiatorQuestions", {})))

    accepted_unique = sorted(set(accepted_items))
    mention_unique = sorted(set(conversation_mentions))

    return {
        "dialogue_id": str(raw.get("conversationId", "")),
        "turns": turns,
        "user_id": initiator_id,
        "accepted_items": accepted_unique,
        "mentioned_items": mention_unique,
        "success": bool(accepted_unique),
    }


def split_validation_test(
    conversations: List[Dict[str, Any]], val_ratio: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not conversations:
        return [], []

    rng = random.Random(seed)
    shuffled = list(conversations)
    rng.shuffle(shuffled)

    val_size = int(round(len(shuffled) * val_ratio))
    val_size = max(1, min(len(shuffled) - 1, val_size)) if len(shuffled) > 1 else len(shuffled)
    return shuffled[:val_size], shuffled[val_size:]


def parse_year_from_title(title: str) -> Any:
    match = YEAR_PATTERN.search(title)
    if not match:
        return ""
    try:
        return int(match.group(1))
    except Exception:
        return ""


def split_genres(value: str) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def build_item_catalog(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie_id = str(row.get("movieId") or row.get("id") or "").strip()
            if not movie_id:
                continue

            title = str(row.get("movieName") or row.get("title") or f"Movie {movie_id}").strip()
            mentions = as_int(row.get("nbMentions", row.get("mentions", 0)), 0)

            genre_text = str(row.get("genre") or row.get("genres") or "").strip()
            genres = split_genres(genre_text)
            category = str(row.get("category") or "").strip() or (genres[0] if genres else "Unknown")

            catalog[movie_id] = {
                "title": title,
                "year": parse_year_from_title(title),
                "rating": float(row.get("rating") or 0.0),
                "genre": genre_text,
                "genres": genres,
                "category": category,
                "mentions": mentions,
            }

    return catalog


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild ReDial dataset artifacts for MO-CRS")
    parser.add_argument("--dataset_dir", default="data/ReDial", help="Path to ReDial dataset directory")
    parser.add_argument("--train_jsonl", default="train_data.jsonl", help="Training JSONL filename")
    parser.add_argument("--test_jsonl", default="test_data.jsonl", help="Test JSONL filename")
    parser.add_argument(
        "--catalog_csv",
        default="movies_with_mentions_with_genre_category_filled.csv",
        help="Movie metadata CSV filename",
    )
    parser.add_argument("--val_ratio", type=float, default=0.5, help="Validation split ratio from test_data_full")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for val/test split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)

    train_jsonl_path = dataset_dir / args.train_jsonl
    test_jsonl_path = dataset_dir / args.test_jsonl
    catalog_csv_path = dataset_dir / args.catalog_csv

    missing = [
        str(path)
        for path in [train_jsonl_path, test_jsonl_path, catalog_csv_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Required file(s) missing: {missing}")

    print("=" * 70)
    print("ReDial dataset rebuild")
    print("=" * 70)

    train_raw = load_jsonl(train_jsonl_path)
    test_raw = load_jsonl(test_jsonl_path)

    train_conversations = [conv for conv in (convert_dialogue(row) for row in train_raw) if conv]
    test_full_conversations = [conv for conv in (convert_dialogue(row) for row in test_raw) if conv]
    val_conversations, test_conversations = split_validation_test(
        test_full_conversations,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_data_full = train_conversations + test_full_conversations
    item_catalog = build_item_catalog(catalog_csv_path)

    save_json(dataset_dir / "train_data.json", train_conversations)
    save_json(dataset_dir / "test_data_full.json", test_full_conversations)
    save_json(dataset_dir / "val_data.json", val_conversations)
    save_json(dataset_dir / "test_data.json", test_conversations)
    save_json(dataset_dir / "train_data_full.json", train_data_full)
    save_json(dataset_dir / "item_catalog.json", item_catalog)

    category_counter = Counter(item.get("category", "Unknown") for item in item_catalog.values())
    unknown_count = category_counter.get("Unknown", 0)
    mention_nonzero = sum(1 for item in item_catalog.values() if int(item.get("mentions", 0)) > 0)

    print(f"Train conversations:      {len(train_conversations)}")
    print(f"Validation conversations: {len(val_conversations)}")
    print(f"Test conversations:       {len(test_conversations)}")
    print(f"Test full conversations:  {len(test_full_conversations)}")
    print(f"Train full conversations: {len(train_data_full)}")
    print(f"Catalog size:             {len(item_catalog)}")
    print(f"Catalog Unknown category: {unknown_count}")
    print(f"Catalog nonzero mentions: {mention_nonzero}")

    print("\nTop categories:")
    for category, count in category_counter.most_common(10):
        print(f"- {category}: {count}")

    print("\nGenerated files:")
    print(f"- {dataset_dir / 'train_data.json'}")
    print(f"- {dataset_dir / 'val_data.json'}")
    print(f"- {dataset_dir / 'test_data.json'}")
    print(f"- {dataset_dir / 'test_data_full.json'}")
    print(f"- {dataset_dir / 'train_data_full.json'}")
    print(f"- {dataset_dir / 'item_catalog.json'}")


if __name__ == "__main__":
    main()
