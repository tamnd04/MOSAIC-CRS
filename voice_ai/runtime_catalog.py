"""Build a lightweight ReDial item catalog from converted conversation JSON.

The original MOSAIC-CRS runtime expects an external ``item_catalog.json``.  A
checkpoint and converted conversations are sufficient to load the neural model,
but not sufficient to recover the original title/genre/embedding metadata.  This
module creates a deterministic fallback catalog so the realtime demo can run.

When the original catalog is available, use it instead; it will provide much
better titles, genres, and recommendation quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import numpy as np


def _conversation_items(conversation: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ("accepted_items", "mentioned_items"):
        raw = conversation.get(key, [])
        if isinstance(raw, list):
            values.extend(str(v) for v in raw if v is not None)

    for turn in conversation.get("turns", []) or []:
        if not isinstance(turn, dict):
            continue
        for key in ("items_mentioned", "mentioned_items", "recommended_items"):
            raw = turn.get(key, [])
            if isinstance(raw, list):
                values.extend(str(v) for v in raw if v is not None)
    return values


def _seed_for(item_id: str) -> int:
    digest = hashlib.sha256(item_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _unit_hash_vector(item_id: str, dim: int) -> np.ndarray:
    rng = np.random.default_rng(_seed_for(item_id))
    vector = rng.standard_normal(dim, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


def build_fallback_catalog(
    train_data_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    embedding_dim: int = 128,
) -> Dict[str, Any]:
    """Create deterministic popularity/co-occurrence embeddings from ReDial JSON."""
    train_path = Path(train_data_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    with train_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in ("conversations", "dialogues", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError("Training data must be a list of conversation objects.")

    popularity: Counter[str] = Counter()
    conversation_sets: List[List[str]] = []
    all_ids: Set[str] = set()

    for conversation in payload:
        if not isinstance(conversation, dict):
            continue
        raw_ids = _conversation_items(conversation)
        for item_id in raw_ids:
            popularity[item_id] += 1
            all_ids.add(item_id)
        unique_ids = list(dict.fromkeys(raw_ids))
        if unique_ids:
            conversation_sets.append(unique_ids)

    if not all_ids:
        raise ValueError("No movie IDs were found in the supplied ReDial training data.")

    item_ids = sorted(all_ids, key=lambda x: (len(x), x))
    base = {item_id: _unit_hash_vector(item_id, embedding_dim) for item_id in item_ids}
    accum = {item_id: np.zeros(embedding_dim, dtype=np.float32) for item_id in item_ids}
    weights = Counter()

    # Co-occurrence sketch: each movie receives the average hashed context of the
    # other movies mentioned in the same dialogue.  This is cheap, deterministic,
    # and gives the fallback catalog more structure than pure random embeddings.
    for ids in conversation_sets:
        limited = ids[:40]
        if not limited:
            continue
        total = np.sum([base[item_id] for item_id in limited], axis=0)
        for item_id in limited:
            count = len(limited) - 1
            if count <= 0:
                continue
            accum[item_id] += (total - base[item_id]) / float(count)
            weights[item_id] += 1

    catalog: Dict[str, Dict[str, Any]] = {}
    max_pop = max(popularity.values()) if popularity else 1
    for item_id in item_ids:
        vector = base[item_id].copy()
        if weights[item_id] > 0:
            context = accum[item_id] / float(weights[item_id])
            vector = 0.45 * vector + 0.55 * context
        norm = float(np.linalg.norm(vector))
        vector = vector / max(norm, 1e-8)
        count = int(popularity[item_id])
        catalog[item_id] = {
            "id": item_id,
            "title": f"Movie {item_id}",
            "category": "Unknown",
            "genres": [],
            "mentions": count,
            "popularity": count,
            "rating": "",
            "year": "",
            "provider": "ReDial",
            "embedding": [round(float(value), 7) for value in vector.tolist()],
            "generated_fallback": True,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(catalog, handle, ensure_ascii=False, separators=(",", ":"))

    return {
        "output": str(output),
        "items": len(catalog),
        "conversations": len(payload),
        "max_popularity": int(max_pop),
        "fallback": True,
    }


def ensure_catalog(
    train_data_path: str | os.PathLike[str],
    catalog_path: str | os.PathLike[str],
    embedding_dim: int = 128,
) -> Dict[str, Any]:
    catalog = Path(catalog_path).expanduser().resolve()
    if catalog.exists() and catalog.stat().st_size > 0:
        return {"output": str(catalog), "created": False, "fallback": False}
    result = build_fallback_catalog(train_data_path, catalog, embedding_dim)
    result["created"] = True
    return result
