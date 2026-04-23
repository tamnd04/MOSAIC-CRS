"""
Data Loading and Preprocessing Utilities
Handles ReDial dataset and creates conversation episodes
"""

import json
import os
import pickle
import random
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm


DATASET_FOLDER_ALIASES = {
    'redial': 'ReDial',
    'gorecdial': 'GoRecDial',
    'inspired': 'INSPIRED',
    'movielens1m': 'MovieLens_1M',
    'movielens': 'MovieLens_1M',
    'yelp': 'Yelp',
    'durecdial': 'DuRecDial',
    'lastfm': 'LastFM',
    'opendialkg': 'OpenDialKG',
}


def _normalize_dataset_name(name: str) -> str:
    """Normalize dataset names for alias matching."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def _load_json_or_jsonl(path: str):
    """Load JSON or JSONL content from a file path."""
    if path.lower().endswith('.jsonl'):
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _pick_existing_path(candidates: List[str]) -> str:
    """Pick the first existing path from candidates, fallback to first candidate."""
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0] if candidates else ''


def _resolve_dataset_folder_name(dataset_name: str, data_root: str) -> str:
    """Resolve canonical folder name for dataset, preferring existing folders."""
    normalized = _normalize_dataset_name(dataset_name)
    preferred = DATASET_FOLDER_ALIASES.get(normalized, dataset_name)

    if os.path.isdir(data_root):
        for entry in os.listdir(data_root):
            full_path = os.path.join(data_root, entry)
            if os.path.isdir(full_path) and _normalize_dataset_name(entry) == normalized:
                return entry

    return preferred


def apply_dataset_paths(config: Dict, dataset_name: str = None) -> Dict:
    """
    Resolve dataset-specific data paths under data_root.

    This enables selecting datasets like ReDial, GoRecDial, INSPIRED, MovieLens_1M,
    Yelp, DuRecDial, LastFM, OpenDialKG from the same train/test entrypoints.
    """
    data_cfg = config.setdefault('data', {})

    requested_name = dataset_name or data_cfg.get('dataset_name', 'ReDial')
    requested_name = str(requested_name)

    data_root = data_cfg.get('data_root')
    current_data_dir = data_cfg.get('data_dir', './data')
    if not data_root:
        base = os.path.basename(os.path.normpath(current_data_dir))
        if _normalize_dataset_name(base) in {
            _normalize_dataset_name(folder) for folder in DATASET_FOLDER_ALIASES.values()
        }:
            data_root = os.path.dirname(os.path.normpath(current_data_dir))
        else:
            data_root = current_data_dir

    folder_name = _resolve_dataset_folder_name(requested_name, data_root)
    dataset_dir = os.path.join(data_root, folder_name)
    processed_dir = os.path.join(dataset_dir, 'processed')

    train_file = _pick_existing_path([
        os.path.join(dataset_dir, 'train_data.json'),
        os.path.join(dataset_dir, 'train_data.jsonl'),
    ])
    val_file = _pick_existing_path([
        os.path.join(dataset_dir, 'val_data.json'),
        os.path.join(dataset_dir, 'val_data.jsonl'),
    ])
    test_file = _pick_existing_path([
        os.path.join(dataset_dir, 'test_data.json'),
        os.path.join(dataset_dir, 'test_data.jsonl'),
    ])

    full_data_file = _pick_existing_path([
        os.path.join(dataset_dir, 'train_data_full.json'),
        train_file,
    ])

    catalog_candidates = [
        os.path.join(dataset_dir, 'item_catalog.json'),
        os.path.join(data_root, 'item_catalog.json'),
        data_cfg.get('catalog_file', ''),
    ]
    catalog_file = _pick_existing_path(catalog_candidates)

    resolved_num_items = data_cfg.get('num_items', 0)
    if catalog_file and os.path.exists(catalog_file):
        try:
            with open(catalog_file, 'r', encoding='utf-8') as f:
                catalog_data = json.load(f)
            if isinstance(catalog_data, dict):
                resolved_num_items = len(catalog_data)
        except Exception:
            # Keep existing num_items when catalog is unreadable.
            pass

    data_cfg.update({
        'dataset_name': folder_name,
        'data_root': data_root,
        'data_dir': dataset_dir,
        'processed_dir': processed_dir,
        'train_file': train_file,
        'val_file': val_file,
        'test_file': test_file,
        'full_data_file': full_data_file,
        'catalog_file': catalog_file,
        'num_items': int(resolved_num_items) if resolved_num_items else data_cfg.get('num_items', 0),
    })

    return config


def _load_conversation_file(path: str) -> List[Dict]:
    """Load conversation list from a JSON file supporting common schemas."""
    data = _load_json_or_jsonl(path)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ['conversations', 'dialogues', 'data']:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"Unsupported conversation format in {path}")


def _save_conversation_file(path: str, conversations: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=True, indent=2)


def create_stronger_validation_splits(config: Dict, source_file: str = None) -> Dict[str, int]:
    """
    Create deterministic stratified train/val/test splits for stronger validation.

    Splits are stratified by demographic group and conversation success when available.
    """
    data_cfg = config.get('data', {})
    split_cfg = data_cfg.get('stronger_validation', {})

    if source_file is None:
        source_file = data_cfg.get('full_data_file', data_cfg.get('train_file'))

    if not source_file or not os.path.exists(source_file):
        raise FileNotFoundError(f"Split source file not found: {source_file}")

    conversations = _load_conversation_file(source_file)
    if not conversations:
        raise ValueError("No conversations available for split generation")

    rng = random.Random(int(config.get('seed', 42)))
    val_ratio = float(split_cfg.get('val_ratio', 0.15))
    test_ratio = float(split_cfg.get('test_ratio', 0.15))
    min_group_samples = int(split_cfg.get('min_group_samples', 5))
    stratify_by = split_cfg.get('stratify_by', 'demographic_success')

    groups = defaultdict(list)

    for idx, conv in enumerate(conversations):
        profile = conv.get('user_profile', {}) if isinstance(conv, dict) else {}
        age_group = str(profile.get('age_group', 'unknown'))
        success = bool(conv.get('success', False)) if isinstance(conv, dict) else False

        if stratify_by == 'demographic_success':
            group_key = f"{age_group}|{int(success)}"
        elif stratify_by == 'demographic':
            group_key = age_group
        else:
            group_key = 'all'

        groups[group_key].append(conv)

    train_split = []
    val_split = []
    test_split = []

    for _, group_convs in groups.items():
        rng.shuffle(group_convs)
        n = len(group_convs)

        if n < min_group_samples:
            train_split.extend(group_convs)
            continue

        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_test + n_val >= n:
            n_test = max(1, n // 5)
            n_val = max(1, n // 5)

        test_split.extend(group_convs[:n_test])
        val_split.extend(group_convs[n_test:n_test + n_val])
        train_split.extend(group_convs[n_test + n_val:])

    rng.shuffle(train_split)
    rng.shuffle(val_split)
    rng.shuffle(test_split)

    train_file = data_cfg.get('train_file', './data/train_data.json')
    val_file = data_cfg.get('val_file', './data/val_data.json')
    test_file = data_cfg.get('test_file', './data/test_data.json')

    _save_conversation_file(train_file, train_split)
    _save_conversation_file(val_file, val_split)
    _save_conversation_file(test_file, test_split)

    return {
        'train': len(train_split),
        'val': len(val_split),
        'test': len(test_split)
    }


class ReDialDataset(Dataset):
    """ReDial Conversational Recommendation Dataset"""
    
    def __init__(self, data_path: str, item_catalog_path: str, split: str = 'train',
                 split_file: str = None,
                 max_dialogue_length: int = 20, max_utterance_length: int = 128):
        """
        Args:
            data_path: Path to ReDial dataset
            item_catalog_path: Path to item catalog JSON
            split: 'train', 'val', or 'test'
            max_dialogue_length: Maximum turns in dialogue
            max_utterance_length: Maximum tokens in utterance
        """
        self.data_path = data_path
        self.split = split
        self.split_file = split_file
        self.max_dialogue_length = max_dialogue_length
        self.max_utterance_length = max_utterance_length
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load item catalog
        self.item_catalog = self.load_item_catalog(item_catalog_path)
        
        # Load conversations
        self.conversations = self.load_conversations()
        
        print(f"Loaded {len(self.conversations)} conversations for {split} split")
    
    def load_item_catalog(self, catalog_path: str) -> Dict:
        """Load item catalog with features"""
        if os.path.exists(catalog_path):
            with open(catalog_path, 'r', encoding='utf-8') as f:
                catalog = json.load(f)
        else:
            # Create dummy catalog if not exists
            print(f"Creating dummy catalog at {catalog_path}")
            catalog = self.create_dummy_catalog()
            os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
            with open(catalog_path, 'w', encoding='utf-8') as f:
                json.dump(catalog, f)
        
        return catalog
    
    def create_dummy_catalog(self, num_items: int = 1000) -> Dict:
        """Create dummy item catalog for testing"""
        catalog = {}
        categories = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi']
        
        for i in range(num_items):
            catalog[str(i)] = {
                'id': str(i),
                'title': f'Item {i}',
                'category': random.choice(categories),
                'rating': round(random.uniform(3.0, 5.0), 1),
                'popularity': random.randint(1, 1000),
                'features': {
                    'category': random.choice(categories),
                    'year': random.randint(1990, 2024),
                    'rating': round(random.uniform(3.0, 5.0), 1)
                },
                'embedding': np.random.randn(128).tolist()
            }
        
        return catalog
    
    def load_conversations(self) -> List[Dict]:
        """Load and preprocess conversations"""
        data_file = self.split_file or os.path.join(self.data_path, f'{self.split}_data.json')
        
        if not os.path.exists(data_file):
            # Create dummy data if not exists
            print(f"Creating dummy conversation data at {data_file}")
            conversations = self.create_dummy_conversations()
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            _save_conversation_file(data_file, conversations)
        else:
            conversations = _load_conversation_file(data_file)
        
        # Preprocess conversations
        processed = []
        for conv in conversations:
            processed_conv = self.preprocess_conversation(conv)
            if processed_conv:
                processed.append(processed_conv)
        
        return processed
    
    def create_dummy_conversations(self, num_convs: int = 100) -> List[Dict]:
        """Create dummy conversations for testing"""
        conversations = []
        intents = ['request_recommendation', 'inform_preference', 'accept', 
                   'reject', 'request_info', 'goodbye']
        
        for conv_id in range(num_convs):
            num_turns = random.randint(5, 15)
            turns = []
            
            for turn_id in range(num_turns):
                user_utterance = f"User utterance {turn_id} in conversation {conv_id}"
                system_utterance = f"System response {turn_id}"
                
                # Random intent
                intent = random.choice(intents)
                
                # Random mentioned items
                mentioned_items = random.sample(
                    list(self.item_catalog.keys()), 
                    k=random.randint(0, 3)
                )
                
                turns.append({
                    'turn_id': turn_id,
                    'user_utterance': user_utterance,
                    'system_utterance': system_utterance,
                    'intent': intent,
                    'mentioned_items': mentioned_items,
                    'accepted': random.random() > 0.7
                })
            
            conversations.append({
                'conversation_id': conv_id,
                'turns': turns,
                'user_profile': {
                    'age_group': random.choice(['18-25', '26-40', '40+']),
                    'gender': random.choice(['M', 'F', 'Other']),
                    'preferences': random.sample(list(self.item_catalog.keys()), 5)
                },
                'success': turns[-1]['accepted'] if turns else False
            })
        
        return conversations
    
    def preprocess_conversation(self, conv: Dict) -> Dict:
        """Preprocess a single conversation"""
        if len(conv['turns']) == 0:
            return None
        
        # Truncate if too long
        if len(conv['turns']) > self.max_dialogue_length:
            conv['turns'] = conv['turns'][:self.max_dialogue_length]
        
        # Tokenize utterances
        for turn in conv['turns']:
            turn['user_tokens'] = self.tokenizer.encode(
                turn['user_utterance'],
                max_length=self.max_utterance_length,
                padding='max_length',
                truncation=True
            )
        
        return conv
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single conversation"""
        return self.conversations[idx]


class ConversationBatch:
    """Batch of conversations with padding"""
    
    def __init__(self, conversations: List[Dict], device: str = 'cpu'):
        self.conversations = conversations
        self.device = device
        self.batch_size = len(conversations)
        
    def to_device(self, device: str):
        """Move batch to device"""
        self.device = device
        return self


def collate_conversations(batch: List[Dict]) -> ConversationBatch:
    """Collate function for DataLoader"""
    return ConversationBatch(batch)


class ItemCatalog:
    """Item catalog manager"""
    
    def __init__(self, catalog_path: str):
        with open(catalog_path, 'r', encoding='utf-8') as f:
            self.catalog = json.load(f)
        
        self.item_ids = list(self.catalog.keys())
        self.num_items = len(self.item_ids)
        
        # Build category index
        self.category_index = defaultdict(list)
        for item_id, item_data in self.catalog.items():
            category = item_data.get('category', 'Unknown')
            self.category_index[category].append(item_id)
        
        print(f"Loaded catalog with {self.num_items} items")
        print(f"Categories: {list(self.category_index.keys())}")
    
    def get_item(self, item_id: str) -> Dict:
        """Get item by ID"""
        return self.catalog.get(item_id, None)
    
    def get_item_embedding(self, item_id: str) -> np.ndarray:
        """Get item embedding"""
        item = self.get_item(item_id)
        if item and 'embedding' in item:
            return np.array(item['embedding'])
        return np.random.randn(128)  # Random embedding if not found
    
    def get_items_by_category(self, category: str, limit: int = 10) -> List[str]:
        """Get items in a category"""
        items = self.category_index.get(category, [])
        return random.sample(items, min(limit, len(items)))
    
    def sample_items(self, n: int) -> List[str]:
        """Sample random items"""
        return random.sample(self.item_ids, min(n, self.num_items))
    
    def get_similar_items(self, item_id: str, n: int = 10) -> List[str]:
        """Get similar items (simple version based on category)"""
        item = self.get_item(item_id)
        if not item:
            return self.sample_items(n)
        
        category = item.get('category', 'Unknown')
        similar = self.get_items_by_category(category, limit=n+1)
        
        # Remove the query item and return
        similar = [i for i in similar if i != item_id]
        return similar[:n]


class UserSimulator:
    """Simulates user behavior for training"""
    
    def __init__(self, item_catalog: ItemCatalog, config: Dict):
        self.item_catalog = item_catalog
        self.config = config
        
        # User profile
        self.user_id = None
        self.preferences = []
        self.constraints = {}
        self.liked_items = []
        self.disliked_items = []
        
        # Demographics
        self.age_group = None
        self.gender = None
        
        # Behavioral traits
        self.patience = random.randint(10, 20)  # Max turns
        self.acceptance_threshold = random.uniform(0.6, 0.9)
        self.diversity_preference = random.uniform(0.3, 0.7)
    
    
    def reset(self, user_profile: Dict = None):
        """Reset user state with richer preference and taxonomy information."""
        if user_profile:
            self.user_id = user_profile.get('user_id', f'user_{random.randint(0, 10000)}')
            self.preferences = [str(x) for x in user_profile.get('preferences', [])]
            self.age_group = user_profile.get('age_group', random.choice(['18-25', '26-40', '40+']))
            self.gender = user_profile.get('gender', random.choice(['M', 'F', 'Other']))
        else:
            self.user_id = f'user_{random.randint(0, 10000)}'
            self.preferences = [str(x) for x in self.item_catalog.sample_items(5)]
            self.age_group = random.choice(['18-25', '26-40', '40+'])
            self.gender = random.choice(['M', 'F', 'Other'])

        accepted_items = [str(x) for x in (user_profile or {}).get('accepted_items', []) if x is not None]
        mentioned_items = [str(x) for x in (user_profile or {}).get('mentioned_items', []) if x is not None]
        for item_id in accepted_items + mentioned_items:
            if item_id not in self.preferences:
                self.preferences.append(item_id)

        self.accepted_items: Set[str] = set(accepted_items)
        self.mentioned_items: Set[str] = set(mentioned_items)
        self.preference_categories: Set[str] = set(str(x) for x in (user_profile or {}).get('preferred_categories', []) if x)
        self.preference_genres: Set[str] = set(str(x) for x in (user_profile or {}).get('preferred_genres', []) if x)

        # Recover taxonomy from preference items when caller did not provide it explicitly.
        for item_id in self.preferences:
            item = self.item_catalog.get_item(item_id)
            if not item:
                continue
            category = item.get('category')
            if category:
                self.preference_categories.add(str(category))
            genres = item.get('genres', item.get('genre', []))
            if isinstance(genres, str):
                genres = [g.strip() for g in genres.split('|') if g.strip()]
            for genre in genres if isinstance(genres, list) else []:
                self.preference_genres.add(str(genre))

        self.liked_items = []
        self.disliked_items = []
        self.patience = random.randint(12, 20)
        base_thresh = 0.5 if self.preferences else 0.58
        self.acceptance_threshold = random.uniform(base_thresh, min(base_thresh + 0.18, 0.74))

    
    def respond_to_recommendation(self, items: List[str], turn: int) -> Dict:
        """Respond to system recommendation using exact-match and taxonomy-aware relevance."""
        if not items:
            return {'action': 'reject', 'feedback': 'No items provided'}

        if turn >= self.patience:
            return {'action': 'leave', 'feedback': 'Too many turns'}

        items = [str(x) for x in items]

        # Strong positive signal if the system recommends an explicitly accepted item.
        accepted_overlap = [item_id for item_id in items if item_id in self.accepted_items]
        if accepted_overlap:
            chosen = accepted_overlap[0]
            self.liked_items.append(chosen)
            return {
                'action': 'accept',
                'item': chosen,
                'feedback': f'I would watch {chosen}',
                'score': 1.0,
            }

        scores = []
        for item_id in items:
            score = self.compute_relevance(item_id)
            scores.append((item_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_item, best_score = scores[0]
        avg_top = float(np.mean([s for _, s in scores[: min(3, len(scores))]]))

        if best_score >= self.acceptance_threshold or avg_top >= (self.acceptance_threshold + 0.05):
            self.liked_items.append(best_item)
            return {
                'action': 'accept',
                'item': best_item,
                'feedback': f'I like {best_item}!',
                'score': best_score,
            }
        if best_score >= 0.36 or avg_top >= 0.32:
            matched_item = self.item_catalog.get_item(best_item) or {}
            matched_category = matched_item.get('category', 'that kind of movie')
            return {
                'action': 'ask_more',
                'feedback': f'Tell me more about the {matched_category} option {best_item}',
                'score': best_score,
            }

        self.disliked_items.extend(items)
        return {
            'action': 'reject',
            'feedback': 'These do not match what I am looking for',
            'score': best_score,
        }

    
    def compute_relevance(self, item_id: str) -> float:
        """Compute a richer relevance score using exact matches and genre/category overlap."""
        item_id = str(item_id)
        item = self.item_catalog.get_item(item_id)
        if not item:
            return 0.0

        score = 0.0
        item_category = str(item.get('category', 'Unknown'))
        item_genres = item.get('genres', item.get('genre', []))
        if isinstance(item_genres, str):
            item_genres = [g.strip() for g in item_genres.split('|') if g.strip()]
        item_genres = set(str(g) for g in item_genres if g)

        if item_id in self.preferences:
            score += 0.72
        elif item_id in self.mentioned_items:
            score += 0.18

        if item_category in self.preference_categories:
            score += 0.16
        if item_genres:
            overlap = len(item_genres.intersection(self.preference_genres))
            score += min(0.18, 0.06 * overlap)

        liked_taxonomy = set()
        for liked_id in self.liked_items:
            liked_item = self.item_catalog.get_item(liked_id)
            if not liked_item:
                continue
            liked_taxonomy.add(str(liked_item.get('category', 'Unknown')))
            liked_genres = liked_item.get('genres', liked_item.get('genre', []))
            if isinstance(liked_genres, str):
                liked_genres = [g.strip() for g in liked_genres.split('|') if g.strip()]
            liked_taxonomy.update(str(g) for g in liked_genres if g)

        if item_category in liked_taxonomy:
            score += 0.08
        if item_genres.intersection(liked_taxonomy):
            score += 0.05

        disliked_categories = set()
        for disliked_id in self.disliked_items:
            disliked_item = self.item_catalog.get_item(disliked_id)
            if disliked_item:
                disliked_categories.add(str(disliked_item.get('category', 'Unknown')))
        if item_category in disliked_categories:
            score -= 0.15

        mentions = float(item.get('mentions', item.get('popularity', 0.0)) or 0.0)
        # Mild popularity prior keeps reasonable mainstream titles acceptable while not dominating.
        score += min(0.06, np.log1p(max(mentions, 0.0)) / 20.0)

        score += random.uniform(-0.04, 0.04)
        return max(0.0, min(1.0, score))

    
        def provide_preference(self) -> str:
            """Provide a more specific preference statement grounded in item taxonomy."""
            if not self.preferences:
                return "I'm looking for something good"

            item_id = random.choice(self.preferences)
            item = self.item_catalog.get_item(item_id)

            if item:
                category = str(item.get('category', 'movies'))
                title = str(item.get('title', item_id))
                genres = item.get('genres', item.get('genre', []))
                if isinstance(genres, str):
                    genres = [g.strip() for g in genres.split('|') if g.strip()]
                genre_text = str(genres[0]) if isinstance(genres, list) and genres else category
                return f"I usually enjoy {genre_text} or {category} movies like {title}"
            return "I'm looking for something interesting"


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders"""
    apply_dataset_paths(config)
    
    data_dir = config['data']['data_dir']
    catalog_path = config['data']['catalog_file']
    
    # Create datasets
    train_dataset = ReDialDataset(
        data_path=data_dir,
        item_catalog_path=catalog_path,
        split='train',
        split_file=config['data'].get('train_file'),
        max_dialogue_length=config['data']['max_dialogue_length'],
        max_utterance_length=config['data']['max_utterance_length']
    )
    
    val_dataset = ReDialDataset(
        data_path=data_dir,
        item_catalog_path=catalog_path,
        split='val',
        split_file=config['data'].get('val_file'),
        max_dialogue_length=config['data']['max_dialogue_length'],
        max_utterance_length=config['data']['max_utterance_length']
    )
    
    test_dataset = ReDialDataset(
        data_path=data_dir,
        item_catalog_path=catalog_path,
        split='test',
        split_file=config['data'].get('test_file'),
        max_dialogue_length=config['data']['max_dialogue_length'],
        max_utterance_length=config['data']['max_utterance_length']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_conversations,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_conversations,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_conversations,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    return train_loader, val_loader, test_loader


def create_dummy_data(config: Dict) -> None:
    """
    Create dummy data files for testing purposes
    Creates item catalog and sample dialogues
    
    Args:
        config: Configuration dictionary
    """
    import json
    
    # Create data directory
    data_dir = config['data']['data_dir']
    os.makedirs(data_dir, exist_ok=True)
    
    # Create dummy item catalog
    catalog_file = config['data']['catalog_file']
    dummy_catalog = {}
    
    categories = ['action', 'comedy', 'drama', 'sci-fi', 'romance', 'thriller', 'horror', 'documentary']
    
    for i in range(100):  # Create 100 dummy items
        item_id = f'item_{i}'
        item = {
            'title': f'Movie {i}',
            'category': categories[i % len(categories)],
            'rating': round(3.0 + random.random() * 2.0, 1),  # 3.0 - 5.0
            'year': 2000 + (i % 24),
            'features': [random.random() for _ in range(config['data']['item_features_dim'])]
        }
        dummy_catalog[item_id] = item
    
    # Save catalog
    with open(catalog_file, 'w', encoding='utf-8') as f:
        json.dump(dummy_catalog, f, indent=2)
    
    print(f"✓ Created dummy catalog with {len(dummy_catalog)} items at {catalog_file}")
    
    # Create dummy dialogue data
    processed_dir = config['data']['processed_dir']
    os.makedirs(processed_dir, exist_ok=True)
    
    dummy_dialogues = []
    item_ids = list(dummy_catalog.keys())
    for i in range(50):  # Create 50 dummy dialogues
        item_id = item_ids[i % len(item_ids)]
        item_title = dummy_catalog[item_id]['title']
        dialogue = {
            'dialogue_id': f'dialogue_{i}',
            'turns': [
                {
                    'utterance': f'I like {categories[i % len(categories)]} movies',
                    'intent': 'provide_preference',
                    'items_mentioned': []
                },
                {
                    'utterance': f'How about {item_title}?',
                    'intent': 'recommend',
                    'items_mentioned': [item_id]
                }
            ],
            'user_id': f'user_{i % 20}',
            'accepted_items': [item_id] if i % 3 == 0 else []
        }
        dummy_dialogues.append(dialogue)
    
    # Save dialogues
    dialogues_file = os.path.join(processed_dir, 'dummy_dialogues.json')
    with open(dialogues_file, 'w', encoding='utf-8') as f:
        json.dump({'dialogues': dummy_dialogues}, f, indent=2)
    
    print(f"✓ Created {len(dummy_dialogues)} dummy dialogues at {dialogues_file}")


if __name__ == "__main__":
    # Test data loading
    import yaml
    
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dummy data directory
    os.makedirs(config['data']['data_dir'], exist_ok=True)
    
    # Test dataset
    dataset = ReDialDataset(
        data_path=config['data']['data_dir'],
        item_catalog_path=config['data']['catalog_file'],
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample conversation: {dataset[0]}")
    
    # Test catalog
    catalog = ItemCatalog(config['data']['catalog_file'])
    print(f"Catalog size: {catalog.num_items}")
    
    # Test user simulator
    user = UserSimulator(catalog, config)
    user.reset()
    print(f"User profile: {user.age_group}, {user.gender}")
    
    # Test recommendation response
    items = catalog.sample_items(5)
    response = user.respond_to_recommendation(items, turn=1)
    print(f"User response: {response}")
