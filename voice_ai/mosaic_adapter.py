"""Inference adapter that exposes MOSAIC-CRS as a safe, stateful recommendation service.

This module intentionally keeps UI/LangChain concerns outside the research model. It
maps model reranking positions back to real catalog IDs and maintains lightweight
per-session preference/history state for interactive use.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .runtime_catalog import ensure_catalog


_WORD_RE = re.compile(r"[a-z0-9]+")
_NEGATIVE_PREFIX_RE = re.compile(
    r"(?:\bnot\b|\bno\b|\bavoid\b|\bhate\b|\bdislike\b|\bdon t like\b|\bdon't like\b|\bdo not like\b)(?:\s+[a-z0-9]+){0,5}\s*$",
    re.IGNORECASE,
)

_RECOMMENDATION_REQUEST_RE = re.compile(
    r"\b(?:recommend|suggest|find|show|give|looking for|in the mood for|want)\b",
    re.IGNORECASE,
)
_REFINEMENT_RE = re.compile(
    r"\b(?:prefer|instead|rather|something|make it|also|more|lighter|funnier)\b",
    re.IGNORECASE,
)
_REPLACEMENT_RE = re.compile(
    r"\b(?:actually|instead|forget|rather than|change my mind|not that)\b",
    re.IGNORECASE,
)
_ANY_MOVIE_RE = re.compile(r"\b(?:any movie|anything|any genre|surprise me)\b", re.IGNORECASE)

_CATEGORY_ALIASES: Dict[str, Sequence[str]] = {
    "science fiction": ("science fiction", "sci fi", "scifi"),
    "comedy": ("comedy", "comedies", "funny", "humorous"),
    "drama": ("drama", "dramas", "dramatic"),
    "romance": ("romance", "romantic", "love story", "love stories"),
    "horror": ("horror", "scary"),
    "thriller": ("thriller", "thrillers", "suspense", "suspenseful"),
    "action": ("action",),
    "adventure": ("adventure", "adventures"),
    "animation": ("animation", "animated", "cartoon", "cartoons"),
    "fantasy": ("fantasy",),
    "musical": ("musical", "musicals"),
    "crime": ("crime", "criminal"),
    "mystery": ("mystery", "mysterious"),
    "documentary": ("documentary", "documentaries"),
    "western": ("western", "westerns"),
    "family": ("family", "family friendly"),
    "war": ("war",),
    "history": ("history", "historical"),
}

_CATEGORY_DISPLAY = {
    "science fiction": "Science Fiction",
    "comedy": "Comedy",
    "drama": "Drama",
    "romance": "Romance",
    "horror": "Horror",
    "thriller": "Thriller",
    "action": "Action",
    "adventure": "Adventure",
    "animation": "Animation",
    "fantasy": "Fantasy",
    "musical": "Musical",
    "crime": "Crime",
    "mystery": "Mystery",
    "documentary": "Documentary",
    "western": "Western",
    "family": "Family",
    "war": "War",
    "history": "History",
}

_CONTENT_AVOIDANCE_ALIASES: Dict[str, Sequence[str]] = {
    "gore": ("gore", "gory", "graphic violence"),
    "violence": ("violence", "violent"),
    "sad ending": ("sad ending", "tragic ending"),
}

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _normalise(value: Any) -> str:
    return " ".join(_WORD_RE.findall(str(value or "").lower()))


def _split_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: List[str] = []
        for item in value:
            result.extend(_split_values(item))
        return result
    return [part.strip() for part in re.split(r"[|,;/]", str(value)) if part.strip()]


def _canonical_category(value: Any) -> str:
    normalised = _normalise(value)
    for canonical, aliases in _CATEGORY_ALIASES.items():
        if normalised == canonical or normalised in aliases:
            return canonical
    return normalised


def _canonical_categories(value: Any) -> Set[str]:
    """Return every known genre represented by one catalog value.

    This handles compound labels such as ``Romantic Comedy`` or ``Historical Drama``
    without treating them as an unknown single genre.
    """
    normalised = _normalise(value)
    if not normalised:
        return set()
    matches: Set[str] = set()
    for canonical, aliases in _CATEGORY_ALIASES.items():
        for alias in set(aliases) | {canonical}:
            if re.search(rf"\b{re.escape(alias)}\b", normalised):
                matches.add(canonical)
                break
    return matches or {normalised}


def _display_category(value: str) -> str:
    return _CATEGORY_DISPLAY.get(value, value.replace("_", " ").title())


@dataclass
class ConversationSession:
    history: List[Dict[str, str]] = field(default_factory=list)
    preferred_item_ids: List[str] = field(default_factory=list)
    mentioned_item_ids: Set[str] = field(default_factory=set)
    disliked_item_ids: Set[str] = field(default_factory=set)
    preferred_categories: Set[str] = field(default_factory=set)
    disliked_categories: Set[str] = field(default_factory=set)
    active_required_categories: Set[str] = field(default_factory=set)
    content_avoidances: Set[str] = field(default_factory=set)
    turn: int = 0


class MosaicRecommendationAdapter:
    """Lazy-loading, thread-safe inference wrapper around the existing MOCRS model."""

    ACTION_NAMES = {
        0: "ask_preference",
        1: "recommend",
        2: "clarify",
        3: "end",
    }

    def __init__(
        self,
        project_root: str | os.PathLike[str],
        config_path: str | os.PathLike[str],
        checkpoint: str = "best_rl_model.pt",
        dataset: Optional[str] = None,
        train_data_path: Optional[str | os.PathLike[str]] = None,
        catalog_path: Optional[str | os.PathLike[str]] = None,
        auto_build_catalog: bool = True,
        candidate_pool_size: int = 160,
        top_k: int = 5,
        history_turns: int = 8,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.config_path = self._resolve_from_root(config_path)
        self.checkpoint_name = checkpoint
        self.dataset = dataset
        self.train_data_path = (
            self._resolve_from_root(train_data_path) if train_data_path else None
        )
        self.catalog_path = self._resolve_from_root(catalog_path) if catalog_path else None
        self.auto_build_catalog = bool(auto_build_catalog)
        self.catalog_bootstrap: Dict[str, Any] = {}
        self.candidate_pool_size = max(10, int(candidate_pool_size))
        self.top_k = max(1, int(top_k))
        self.history_turns = max(1, int(history_turns))

        self._load_lock = threading.RLock()
        self._inference_lock = threading.RLock()
        self._loaded = False
        self._load_error: Optional[str] = None
        self._sessions: Dict[str, ConversationSession] = {}

        self.config: Dict[str, Any] = {}
        self.model = None
        self.item_catalog = None
        self.device = None
        self.torch = None
        self.np = None

        self._item_ids: List[str] = []
        self._item_by_id: Dict[str, Dict[str, Any]] = {}
        self._title_entries: List[tuple[str, str]] = []
        self._category_to_ids: Dict[str, List[str]] = {}
        self._category_display: Dict[str, str] = {}
        self._popularity_by_id: Dict[str, float] = {}
        self._head_items: Set[str] = set()
        self._tail_items: Set[str] = set()

    def _resolve_from_root(self, path: str | os.PathLike[str]) -> Path:
        candidate = Path(path)
        return candidate.resolve() if candidate.is_absolute() else (self.project_root / candidate).resolve()

    def _prepare_import_path(self) -> None:
        # The original repository keeps model modules in src/. The uploaded snapshot
        # may also place them at the repository root, so support both layouts.
        for candidate in (self.project_root / "src", self.project_root):
            candidate_str = str(candidate)
            if candidate.exists() and candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

    def _resolve_config_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config_dir = self.config_path.parent
        for key in (
            "catalog_file",
            "train_file",
            "val_file",
            "test_file",
            "full_data_file",
            "data_dir",
            "processed_dir",
            "data_root",
        ):
            value = config.get("data", {}).get(key)
            if isinstance(value, str) and value and not os.path.isabs(value):
                config["data"][key] = str((config_dir / value).resolve())

        for key in ("save_dir", "log_dir"):
            value = config.get("logging", {}).get(key)
            if isinstance(value, str) and value and not os.path.isabs(value):
                config["logging"][key] = str((config_dir / value).resolve())
        return config

    def _resolve_checkpoint(self) -> Path:
        requested = Path(self.checkpoint_name)
        if requested.is_absolute() and requested.exists():
            return requested
        direct = (self.project_root / requested).resolve()
        if direct.exists():
            return direct

        save_dir = Path(self.config.get("logging", {}).get("save_dir", self.project_root / "checkpoints"))
        dataset_name = str(self.config.get("data", {}).get("dataset_name", "default"))
        candidates = [
            save_dir / dataset_name / requested.name,
            save_dir / requested.name,
            direct,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        expected = "\n  - ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "MOSAIC-CRS checkpoint was not found. Checked:\n  - " + expected
        )

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            if self._load_error:
                raise RuntimeError(self._load_error)
            try:
                self._prepare_import_path()
                import numpy as np
                import torch
                import yaml

                from data_utils import ItemCatalog, apply_dataset_paths
                from mocrs import MOCRS

                if not self.config_path.exists():
                    raise FileNotFoundError(f"Config file not found: {self.config_path}")

                with self.config_path.open("r", encoding="utf-8") as handle:
                    config = yaml.safe_load(handle)
                if not isinstance(config, dict):
                    raise ValueError(f"Invalid YAML configuration: {self.config_path}")

                config = self._resolve_config_paths(config)
                apply_dataset_paths(config, self.dataset)

                if self.train_data_path is not None:
                    if not self.train_data_path.exists():
                        raise FileNotFoundError(f"Training data not found: {self.train_data_path}")
                    config["data"]["train_file"] = str(self.train_data_path)
                    config["data"]["full_data_file"] = str(self.train_data_path)

                if self.catalog_path is not None:
                    config["data"]["catalog_file"] = str(self.catalog_path)

                catalog_path = Path(config["data"]["catalog_file"])
                if not catalog_path.exists() and self.auto_build_catalog:
                    train_path = Path(config["data"].get("train_file", ""))
                    if not train_path.exists():
                        raise FileNotFoundError(
                            f"Item catalog is missing ({catalog_path}) and training data was not found ({train_path})."
                        )
                    generated_path = self.catalog_path or (
                        self.project_root / "runtime_data" / "ReDial" / "item_catalog.generated.json"
                    )
                    self.catalog_bootstrap = ensure_catalog(
                        train_path,
                        generated_path,
                        int(config["model"]["personalization"].get("item_embedding_dim", 128)),
                    )
                    catalog_path = Path(self.catalog_bootstrap["output"])
                    config["data"]["catalog_file"] = str(catalog_path)

                if not catalog_path.exists():
                    raise FileNotFoundError(
                        f"Item catalog not found: {catalog_path}. Add the original ReDial item_catalog.json "
                        "or enable fallback catalog generation with --train-data."
                    )

                self.config = config

                self.torch = torch
                self.np = np
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.item_catalog = ItemCatalog(str(catalog_path))
                self.model = MOCRS(config).to(self.device)

                checkpoint_path = self._resolve_checkpoint()
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                incompat = self.model.load_state_dict(state, strict=False)
                missing = list(getattr(incompat, "missing_keys", []))
                unexpected = list(getattr(incompat, "unexpected_keys", []))
                if missing or unexpected:
                    print(
                        "[voice adapter] checkpoint loaded with non-strict matching: "
                        f"missing={len(missing)}, unexpected={len(unexpected)}"
                    )

                self.model.eval()
                self._build_catalog_indexes()
                if hasattr(self.model, "dfc") and hasattr(self.model.dfc, "set_catalog_popularity"):
                    self.model.dfc.set_catalog_popularity(self._popularity_by_id)

                self._loaded = True
                print(
                    f"[voice adapter] loaded dataset={config['data']['dataset_name']} "
                    f"items={len(self._item_ids)} device={self.device} checkpoint={checkpoint_path}"
                )
            except Exception as exc:
                self._load_error = f"Failed to initialise MOSAIC-CRS: {exc}"
                raise RuntimeError(self._load_error) from exc

    def _build_catalog_indexes(self) -> None:
        assert self.item_catalog is not None
        self._item_ids = [str(item_id) for item_id in self.item_catalog.item_ids]
        self._item_by_id = {}
        self._title_entries = []
        self._category_to_ids = {}
        self._category_display = {}
        self._popularity_by_id = {}

        for item_id in self._item_ids:
            item = self.item_catalog.get_item(item_id) or {}
            self._item_by_id[item_id] = item
            title = str(item.get("title", item.get("name", item_id))).strip()
            title_norm = _normalise(title)
            if len(title_norm) >= 3:
                self._title_entries.append((title_norm, item_id))

            category_values: List[str] = []
            for key in ("category", "categories", "genre", "genres"):
                category_values.extend(_split_values(item.get(key)))
            if not category_values:
                category_values = ["Unknown"]

            for category in category_values:
                for category_norm in _canonical_categories(category):
                    self._category_to_ids.setdefault(category_norm, []).append(item_id)
                    self._category_display.setdefault(category_norm, _display_category(category_norm))

            popularity = float(item.get("mentions", item.get("popularity", 0.0)) or 0.0)
            self._popularity_by_id[item_id] = popularity

        ordered = sorted(
            self._item_ids,
            key=lambda item_id: (self._popularity_by_id.get(item_id, 0.0), item_id),
            reverse=True,
        )
        n_items = max(1, len(ordered))
        head_cut = max(1, int(0.20 * n_items))
        tail_cut = max(1, int(0.50 * n_items))
        self._head_items = set(ordered[:head_cut])
        self._tail_items = set(ordered[-tail_cut:])

    def status(self) -> Dict[str, Any]:
        return {
            "loaded": self._loaded,
            "error": self._load_error,
            "config": str(self.config_path),
            "dataset": self.config.get("data", {}).get("dataset_name", self.dataset),
            "checkpoint": self.checkpoint_name,
            "device": str(self.device) if self.device is not None else None,
            "items": len(self._item_ids),
            "train_data": str(self.train_data_path) if self.train_data_path else None,
            "catalog": self.config.get("data", {}).get("catalog_file"),
            "catalog_bootstrap": self.catalog_bootstrap,
        }

    def _get_session(self, session_id: str) -> ConversationSession:
        return self._sessions.setdefault(session_id, ConversationSession())

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def record_assistant(self, session_id: str, text: str) -> None:
        session = self._get_session(session_id)
        session.history.append({"role": "assistant", "content": str(text)})
        session.history = session.history[-(self.history_turns * 2) :]

    @staticmethod
    def _is_negated(text: str, start_index: int) -> bool:
        prefix = text[max(0, start_index - 40) : start_index]
        return bool(_NEGATIVE_PREFIX_RE.search(prefix))

    @staticmethod
    def _extract_requested_count(text: str, default: int) -> int:
        patterns = (
            r"\b(?:recommend|suggest|give|show|find)(?:\s+me)?\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:movies?|recommendations?)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            raw = match.group(1)
            value = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw, default)
            return max(1, min(int(value), 10))
        return default

    def _extract_query_signals(self, query: str) -> Dict[str, Any]:
        text = _normalise(query)
        positive: Set[str] = set()
        negative: Set[str] = set()
        content_avoidances: Set[str] = set()

        for canonical, aliases in _CATEGORY_ALIASES.items():
            for alias in sorted(set(aliases) | {canonical}, key=len, reverse=True):
                match = re.search(rf"\b{re.escape(alias)}\b", text)
                if not match:
                    continue
                if self._is_negated(text, match.start()):
                    negative.add(canonical)
                else:
                    positive.add(canonical)
                break

        for canonical, aliases in _CONTENT_AVOIDANCE_ALIASES.items():
            for alias in sorted(set(aliases) | {canonical}, key=len, reverse=True):
                match = re.search(rf"\b{re.escape(alias)}\b", text)
                if match and self._is_negated(text, match.start()):
                    content_avoidances.add(canonical)
                    break

        return {
            "text": text,
            "positive_categories": positive,
            "negative_categories": negative,
            "content_avoidances": content_avoidances,
            "explicit_recommendation": bool(_RECOMMENDATION_REQUEST_RE.search(text)),
            "is_refinement": bool(_REFINEMENT_RE.search(text)),
            "is_replacement": bool(_REPLACEMENT_RE.search(text)),
            "requests_any_movie": bool(_ANY_MOVIE_RE.search(text)),
            "requested_count": self._extract_requested_count(text, self.top_k),
        }

    def _extract_preferences(self, session: ConversationSession, query: str) -> Dict[str, Any]:
        signals = self._extract_query_signals(query)
        text = signals["text"]
        if not text:
            return signals

        positive = set(signals["positive_categories"])
        negative = set(signals["negative_categories"])

        for category in negative:
            session.disliked_categories.add(category)
            session.preferred_categories.discard(category)
            session.active_required_categories.discard(category)

        for category in positive:
            if category not in negative:
                session.preferred_categories.add(category)
                session.disliked_categories.discard(category)

        if signals["explicit_recommendation"] and signals["requests_any_movie"]:
            session.active_required_categories.clear()
        elif signals["explicit_recommendation"] and positive:
            # A direct recommendation request starts a new hard-constraint set.
            session.active_required_categories = positive - negative
        elif positive and signals["is_refinement"] and session.active_required_categories:
            # Follow-ups such as "something light and funny" refine the current request.
            session.active_required_categories.update(positive - negative)
        elif positive and signals["is_replacement"]:
            session.active_required_categories = positive - negative

        session.content_avoidances.update(signals["content_avoidances"])

        # Exact title phrase matches. This is deliberately conservative to avoid
        # treating common words as movie titles.
        for title_norm, item_id in self._title_entries:
            if len(title_norm) < 4 or title_norm not in text:
                continue
            start = text.find(title_norm)
            session.mentioned_item_ids.add(item_id)
            if self._is_negated(text, start):
                session.disliked_item_ids.add(item_id)
                session.preferred_item_ids = [x for x in session.preferred_item_ids if x != item_id]
            elif item_id not in session.preferred_item_ids:
                session.preferred_item_ids.append(item_id)

        session.preferred_item_ids = session.preferred_item_ids[-30:]
        signals["active_required_categories"] = set(session.active_required_categories)
        return signals

    def _item_categories(self, item_id: str) -> Set[str]:
        item = self._item_by_id.get(item_id, {})
        categories: Set[str] = set()
        for key in ("category", "categories", "genre", "genres"):
            for value in _split_values(item.get(key)):
                categories.update(_canonical_categories(value))
        return categories

    def _item_metadata_text(self, item_id: str) -> str:
        item = self._item_by_id.get(item_id, {})
        values: List[str] = []
        for key in (
            "category", "categories", "genre", "genres", "keywords", "tags",
            "content_warnings", "description", "overview", "plot", "synopsis", "summary",
        ):
            values.extend(_split_values(item.get(key)))
        return _normalise(" ".join(values))

    def _contains_avoided_content(self, session: ConversationSession, item_id: str) -> bool:
        metadata_text = self._item_metadata_text(item_id)
        if not metadata_text:
            return False
        for avoidance in session.content_avoidances:
            aliases = set(_CONTENT_AVOIDANCE_ALIASES.get(avoidance, ())) | {avoidance}
            if any(re.search(rf"\b{re.escape(alias)}\b", metadata_text) for alias in aliases):
                return True
        return False

    def _matches_required_categories(self, session: ConversationSession, item_id: str) -> bool:
        if not session.active_required_categories:
            return True
        categories = self._item_categories(item_id)
        return session.active_required_categories.issubset(categories)

    def _allowed_item(self, session: ConversationSession, item_id: str) -> bool:
        if item_id in session.disliked_item_ids:
            return False
        categories = self._item_categories(item_id)
        if categories.intersection(session.disliked_categories):
            return False
        return not self._contains_avoided_content(session, item_id)

    def _seeded_rng(self, session_id: str, turn: int) -> random.Random:
        digest = hashlib.sha256(f"{session_id}:{turn}".encode("utf-8")).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _select_candidates(self, session_id: str, session: ConversationSession) -> List[str]:
        target = min(self.candidate_pool_size, len(self._item_ids))
        rng = self._seeded_rng(session_id, session.turn)
        selected: List[str] = []
        seen: Set[str] = set()

        def add(item_id: str) -> None:
            item_id = str(item_id)
            if (
                len(selected) < target
                and item_id in self._item_by_id
                and item_id not in seen
                and self._allowed_item(session, item_id)
            ):
                seen.add(item_id)
                selected.append(item_id)

        if session.active_required_categories:
            strict_pool = [
                item_id
                for item_id in self._item_ids
                if self._allowed_item(session, item_id)
                and self._matches_required_categories(session, item_id)
                and item_id not in session.mentioned_item_ids
            ]
            strict_pool.sort(
                key=lambda item_id: (
                    item_id in self._head_items,
                    -self._popularity_by_id.get(item_id, 0.0),
                    item_id,
                )
            )
            for item_id in strict_pool:
                add(item_id)
                if len(selected) >= target:
                    break

        # Soft-preference items are only used to fill the model candidate tensor.
        # Final results are filtered again, so non-matching items cannot leak into replies.
        for item_id in session.preferred_item_ids:
            add(item_id)

        category_matches: List[str] = []
        for category in session.preferred_categories:
            category_matches.extend(self._category_to_ids.get(category, []))
        category_matches = list(dict.fromkeys(category_matches))
        # Prefer non-head matches first, but retain some popular items for relevance.
        category_matches.sort(
            key=lambda item_id: (
                item_id in self._head_items,
                -self._popularity_by_id.get(item_id, 0.0),
                item_id,
            )
        )
        for item_id in category_matches[: max(20, target // 2)]:
            add(item_id)

        tail_pool = [item_id for item_id in self._tail_items if self._allowed_item(session, item_id)]
        rng.shuffle(tail_pool)
        for item_id in tail_pool[: max(10, target // 3)]:
            add(item_id)

        remaining = [item_id for item_id in self._item_ids if item_id not in seen and self._allowed_item(session, item_id)]
        rng.shuffle(remaining)
        for item_id in remaining:
            add(item_id)
            if len(selected) >= target:
                break

        if not selected:
            raise RuntimeError("No eligible candidate items were found in the catalog.")
        return selected

    def _build_static_features(self, session: ConversationSession):
        assert self.np is not None
        static_dim = int(self.config["model"]["personalization"]["static_features_dim"])
        values = self.np.zeros((1, static_dim), dtype=self.np.float32)
        values[0, 0] = 0.5  # unknown/neutral age bucket
        values[0, 1] = 0.0  # female indicator unknown
        values[0, 2] = 0.0  # male indicator unknown
        values[0, 3] = min(len(session.preferred_item_ids), 20) / 20.0
        return self.torch.as_tensor(values, dtype=self.torch.float32, device=self.device)

    def _candidate_metadata(self, candidate_ids: Sequence[str]) -> tuple[List[str], List[Dict[str, Any]]]:
        names: List[str] = []
        metadata: List[Dict[str, Any]] = []
        for item_id in candidate_ids:
            item = self._item_by_id.get(item_id, {})
            title = str(item.get("title", item.get("name", item_id)))
            genres = _split_values(item.get("genres", item.get("genre", item.get("category", []))))
            category = str(item.get("category", genres[0] if genres else "Unknown"))
            names.append(title)
            metadata.append(
                {
                    "id": item_id,
                    "name": title,
                    "title": title,
                    "genre": genres[0] if genres else category,
                    "genres": genres,
                    "category": category,
                    "year": item.get("year", ""),
                    "rating": item.get("rating", ""),
                    "actors": item.get("actors", item.get("people_names", [])),
                    "directors": item.get("directors", item.get("director", [])),
                    "people_names": item.get("people_names", []),
                }
            )
        return names, metadata

    def recommend(self, query: str, session_id: str) -> Dict[str, Any]:
        """Run one conversational recommendation turn and return JSON-serialisable data."""
        self.ensure_loaded()
        query = str(query or "").strip()
        if not query:
            raise ValueError("The recommendation query is empty.")

        session = self._get_session(session_id)
        previous_history = list(session.history)
        session.turn += 1
        query_signals = self._extract_preferences(session, query)
        candidate_ids = self._select_candidates(session_id, session)

        assert self.item_catalog is not None
        assert self.model is not None
        assert self.torch is not None
        assert self.np is not None

        embeddings = [self.item_catalog.get_item_embedding(item_id) for item_id in candidate_ids]
        candidate_tensor = self.torch.as_tensor(
            self.np.asarray(embeddings, dtype=self.np.float32),
            dtype=self.torch.float32,
            device=self.device,
        ).unsqueeze(0)

        names, metadata = self._candidate_metadata(candidate_ids)
        history_payload = [
            {"utterance": turn.get("content", "")}
            for turn in previous_history[-(self.history_turns * 2) :]
            if turn.get("content")
        ]
        preferred_titles = [
            str(self._item_by_id.get(item_id, {}).get("title", item_id))
            for item_id in session.preferred_item_ids[-5:]
        ]
        preferred_categories = [
            self._category_display.get(category, category)
            for category in sorted(session.preferred_categories)
        ]

        batch = {
            "utterances": [query],
            "dialogue_history": history_payload or None,
            "static_features": self._build_static_features(session),
            "candidate_items": candidate_tensor,
            "candidate_item_ids": [candidate_ids],
            "candidate_item_names": [names],
            "candidate_item_metadata": [metadata],
            "preferred_reference_titles": [preferred_titles],
            "preferred_categories": [preferred_categories[0] if preferred_categories else ""],
            "generate_explanations": True,
            "is_cold_start": self.torch.tensor(
                [session.turn == 1], dtype=self.torch.bool, device=self.device
            ),
            "use_thompson_sampling": False,
            "user_demographics": [{"age_group": "unknown", "gender": "U"}],
        }

        with self._inference_lock, self.torch.inference_mode():
            outputs = self.model(batch)

        action_probs = outputs.get("action_probs")
        if action_probs is None:
            action_id = 1
            action_probability = 1.0
        else:
            action_id = int(self.torch.argmax(action_probs, dim=-1)[0].item())
            action_probability = float(action_probs[0, action_id].item())

        reranked = outputs.get("reranked_indices")
        reranked_scores = outputs.get("reranked_scores")
        if reranked is None:
            positions = list(range(len(candidate_ids)))
        else:
            raw_positions = [int(index) for index in reranked[0].detach().cpu().tolist()]
            positions = []
            seen_positions: Set[int] = set()
            for position in raw_positions + list(range(len(candidate_ids))):
                if 0 <= position < len(candidate_ids) and position not in seen_positions:
                    seen_positions.add(position)
                    positions.append(position)

        requested_limit = min(int(query_signals.get("requested_count", self.top_k)), self.top_k)
        recommendations: List[Dict[str, Any]] = []
        filtered_out = 0
        for rerank_order, position in enumerate(positions):
            item_id = candidate_ids[position]
            if item_id in session.mentioned_item_ids:
                filtered_out += 1
                continue
            if not self._allowed_item(session, item_id):
                filtered_out += 1
                continue
            if not self._matches_required_categories(session, item_id):
                filtered_out += 1
                continue

            item = self._item_by_id.get(item_id, {})
            item_categories = self._item_categories(item_id)
            score = None
            if reranked_scores is not None and rerank_order < reranked_scores.shape[1]:
                score = float(reranked_scores[0, rerank_order].item())
            recommendations.append(
                {
                    "rank": len(recommendations) + 1,
                    "item_id": item_id,
                    "title": str(item.get("title", item.get("name", item_id))),
                    "category": str(item.get("category", "Unknown")),
                    "genres": _split_values(item.get("genres", item.get("genre", item.get("category", [])))),
                    "canonical_genres": sorted(item_categories),
                    "year": item.get("year", ""),
                    "rating": item.get("rating", ""),
                    "score": score,
                    "constraint_match": True,
                }
            )
            if len(recommendations) >= requested_limit:
                break

        explanations = outputs.get("explanations") or []
        explanation = str(explanations[0]).strip() if explanations else ""
        session.history.append({"role": "user", "content": query})
        session.history = session.history[-(self.history_turns * 2) :]

        strict_candidate_count = sum(
            1
            for item_id in candidate_ids
            if item_id not in session.mentioned_item_ids
            and self._allowed_item(session, item_id)
            and self._matches_required_categories(session, item_id)
        )
        required_display = [_display_category(value) for value in sorted(session.active_required_categories)]
        avoided_display = [_display_category(value) for value in sorted(session.disliked_categories)]
        effective_action = self.ACTION_NAMES.get(action_id, "unknown")
        if query_signals.get("explicit_recommendation"):
            effective_action = "recommend" if recommendations else "clarify"

        return {
            "request_type": "recommendation",
            "policy_action": effective_action,
            "policy_action_probability": action_probability,
            "recommendations": recommendations,
            # The model explanation is retained for diagnostics, but the voice layer
            # does not repeat it because it may overstate genre/content matches.
            "model_explanation": explanation,
            "constraints": {
                "required_genres": required_display,
                "excluded_genres": avoided_display,
                "unverified_content_avoidances": sorted(session.content_avoidances),
                "strict_filter_applied": bool(session.active_required_categories or session.disliked_categories),
                "requested_count": requested_limit,
                "matching_candidates_in_model_pool": strict_candidate_count,
                "returned_count": len(recommendations),
                "filtered_out_count": filtered_out,
            },
            "preferences_detected": {
                "titles": preferred_titles,
                "categories": preferred_categories,
                "avoided_categories": avoided_display,
            },
            "dataset": self.config.get("data", {}).get("dataset_name", self.dataset),
            "turn": session.turn,
        }

    def recommend_json(self, query: str, session_id: str) -> str:
        return json.dumps(self.recommend(query, session_id), ensure_ascii=False)
