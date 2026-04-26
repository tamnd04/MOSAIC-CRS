
import ast
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def norm_title(s: str) -> str:
    s = str(s or '').strip().lower().replace('’', "'")
    s = s.replace('&amp;', 'and')
    s = re.sub(r'\(\d{4}\)\s*$', '', s).strip()
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def extract_year(s: str) -> str:
    m = re.search(r'\((\d{4})\)\s*$', str(s))
    return m.group(1) if m else ''


def try_parse(val):
    if val is None or val == '' or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return None


def parse_rating(x) -> float:
    s = str(x).strip()
    if not s:
        return 0.0
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)', s)
    return float(m.group(1)) if m else 0.0


def deterministic_embedding(seed_text: str, dim: int = 128) -> List[float]:
    seed = int(hashlib.md5(seed_text.encode('utf-8')).hexdigest()[:8], 16)
    rs = np.random.RandomState(seed)
    vec = rs.normal(size=dim).astype(float)
    vec /= max(np.linalg.norm(vec), 1e-8)
    return vec.tolist()


GOODBYE_RE = re.compile(r'\b(bye|goodbye|see you|have a good|have a great|thank(s| you))\b', re.I)
RECOMMEND_RE = re.compile(r'\b(recommend|suggest|watch|check out|try|looking for|any suggestions|would you recommend)\b', re.I)


def parse_title_list(val) -> List[str]:
    v = try_parse(val)
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, dict):
        return [str(k) for k in v.keys() if str(k).strip()]
    s = str(val).strip()
    return [s] if s else []


def age_map_inspired(age: str) -> str:
    mapping = {
        '18 - 24': '18-25',
        '25 - 34': '26-35',
        '35 - 44': '36-45',
        '45 - 54': '46-55',
        '55 - 64': '55+',
        '65+': '55+',
    }
    return mapping.get(str(age).strip(), '26-35')


def gender_map(g: str) -> str:
    s = str(g).strip().lower()
    if s.startswith('f'):
        return 'F'
    if s.startswith('m'):
        return 'M'
    return 'U'


def map_intent(row: Dict, speaker: str, mentioned_titles: List[str]) -> str:
    text = str(row.get('text', '') or '')
    label = str(row.get('expert_label', '') or '').lower()

    if GOODBYE_RE.search(text):
        return 'goodbye'

    if speaker == 'user':
        if RECOMMEND_RE.search(text):
            return 'recommend'
        if mentioned_titles or str(row.get('genres', '')).strip():
            return 'provide_preference'
        if '?' in text:
            return 'ask_question'
        return 'general'

    if mentioned_titles:
        return 'recommend'
    if 'inquiry' in label or '?' in text:
        return 'ask_question'
    if 'opinion' in label or 'preference' in label or 'personal' in label:
        return 'provide_preference'
    if RECOMMEND_RE.search(text):
        return 'recommend'
    return 'general'


def build_inspired_dataset(raw_root: str, out_root: str) -> None:
    raw_root = str(Path(raw_root))
    dialog_root = os.path.join(raw_root, 'dialog_data')
    survey_root = os.path.join(raw_root, 'survey_data')
    out_root = str(Path(out_root))
    os.makedirs(out_root, exist_ok=True)

    movies_df = pd.read_csv(os.path.join(raw_root, 'movie_database.tsv'), sep='\t').fillna('')
    list_ids = pd.read_csv(os.path.join(survey_root, 'list_of_dialog_ids_with_movie_id_all.tsv'), sep='\t').fillna('')
    seek_dem = pd.read_csv(os.path.join(survey_root, 'seeker_demographic.tsv'), sep='\t').fillna('')

    by_video: Dict[str, Dict] = {}
    by_norm: Dict[str, List[Dict]] = defaultdict(list)
    for _, row in movies_df.iterrows():
        d = row.to_dict()
        vid = str(d.get('video_id', '')).strip()
        if vid:
            by_video[vid] = d
        by_norm[norm_title(d.get('title', ''))].append(d)

    def best_db_match(title: str) -> Optional[Dict]:
        title = str(title or '').strip()
        if not title:
            return None
        cands = by_norm.get(norm_title(title), [])
        if not cands:
            return None

        year = extract_year(title)
        if year:
            for c in cands:
                if str(c.get('year', '')).strip() == year:
                    return c

        def score(c: Dict):
            vals = []
            for key in ['youtube_view', 'youtube_like', 'imdb_votes', 'trailer_duration']:
                try:
                    vals.append(float(c.get(key) or 0))
                except Exception:
                    vals.append(0.0)
            return tuple(vals)

        return max(cands, key=score)

    dialog_meta = {str(r['dialog_id']): r.to_dict() for _, r in list_ids.iterrows()}
    seek_by_survey = {str(r['seeker_survey_id']): r.to_dict() for _, r in seek_dem.iterrows()}

    # Include every explicitly recommended target movie ID in the catalog.
    # INSPIRED stores the target recommendation as a YouTube video id in movie_id.
    # Some target movies are not repeated in the utterance-level movies column, so
    # the original converter could create accepted_items that were not in the catalog.
    target_video_counts: Counter = Counter()
    for _, r in list_ids.iterrows():
        vid = str(r.get('movie_id', '')).strip()
        if 'SPLIT' in vid:
            vid = vid.split('SPLIT')[0]
        if vid:
            target_video_counts[vid] += 1

    mention_counts: Counter = Counter()
    all_dialog_titles = set()

    for split in ['train', 'dev', 'test']:
        df = pd.read_csv(os.path.join(dialog_root, f'{split}.tsv'), sep='\t').fillna('')
        for s in df['movie_dict']:
            d = try_parse(s)
            if isinstance(d, dict):
                for t in d.keys():
                    if t:
                        all_dialog_titles.add(t)
                        mention_counts[t] += 1
        for s in df['movies']:
            v = try_parse(s)
            if isinstance(v, list):
                for t in v:
                    t = str(t).strip()
                    if t:
                        all_dialog_titles.add(t)
                        mention_counts[t] += 1
            elif isinstance(v, str):
                t = v.strip()
                if t:
                    all_dialog_titles.add(t)
                    mention_counts[t] += 1

    def synthetic_id(title: str) -> str:
        return 'syn_' + hashlib.md5(title.encode('utf-8')).hexdigest()[:12]

    catalog: Dict[str, Dict] = {}
    title_to_id: Dict[str, str] = {}

    for title in sorted(all_dialog_titles):
        db = best_db_match(title)
        if db is not None and str(db.get('video_id', '')).strip():
            item_id = str(db['video_id']).strip()
            genres = [g.strip() for g in str(db.get('genre', '')).split(',') if g.strip()]
            category = genres[0] if genres else 'Unknown'
            popularity = int(mention_counts[title])
            try:
                popularity += int(float(db.get('youtube_view') or 0) // 100000)
            except Exception:
                pass

            catalog[item_id] = {
                'title': str(db.get('title') or title),
                'year': str(db.get('year', '')).strip(),
                'rating': parse_rating(db.get('rating', '')),
                'genre': '|'.join(genres),
                'genres': genres,
                'category': category,
                'mentions': popularity,
                'actors': [a.strip() for a in str(db.get('actors', '')).split(',') if a.strip()],
                'director': str(db.get('director', '')).strip(),
                'plot': str(db.get('short_plot') or db.get('long_plot') or '').strip(),
                'video_id': item_id,
                'embedding': deterministic_embedding(f"{title}|{db.get('year','')}|{'|'.join(genres)}"),
            }
        else:
            item_id = synthetic_id(title)
            catalog[item_id] = {
                'title': title,
                'year': extract_year(title),
                'rating': 0.0,
                'genre': '',
                'genres': [],
                'category': 'Unknown',
                'mentions': int(mention_counts[title]),
                'actors': [],
                'director': '',
                'plot': '',
                'video_id': '',
                'embedding': deterministic_embedding(title),
            }
        title_to_id[title] = item_id

    # Add target recommendation movies that did not appear in utterance-level movie mentions.
    # This improves ranking supervision because every accepted target item can be sampled,
    # scored, and evaluated as a real catalog item.
    for vid in sorted(target_video_counts.keys()):
        if not vid or vid in catalog or vid not in by_video:
            continue
        db = by_video[vid]
        title = str(db.get("title") or vid)
        genres = [g.strip() for g in str(db.get("genre", "")).split(",") if g.strip()]
        category = genres[0] if genres else "Unknown"
        popularity = int(target_video_counts[vid])
        try:
            popularity += int(float(db.get("youtube_view") or 0) // 100000)
        except Exception:
            pass
        catalog[vid] = {
            "title": title,
            "year": str(db.get("year", "")).strip(),
            "rating": parse_rating(db.get("rating", "")) ,
            "genre": "|".join(genres),
            "genres": genres,
            "category": category,
            "mentions": popularity,
            "actors": [a.strip() for a in str(db.get("actors", "")).split(",") if a.strip()],
            "director": str(db.get("director", "")).strip(),
            "plot": str(db.get("short_plot") or db.get("long_plot") or "").strip(),
            "video_id": vid,
            "embedding": deterministic_embedding(title),
        }
        title_to_id.setdefault(title, vid)

    def convert_split(split: str) -> List[Dict]:
        df = pd.read_csv(os.path.join(dialog_root, f'{split}.tsv'), sep='\t').fillna('')
        conversations: List[Dict] = []

        for dialog_id, g in df.groupby('dialog_id', sort=False):
            g = g.sort_values(['utt_id', 'turn_id'])
            meta = dialog_meta.get(str(dialog_id), {})
            fine = str(g['fine_label'].iloc[0] or meta.get('fine_label', '')).strip()
            coarse = str(g['coarse_label'].iloc[0] or meta.get('coarse_label', '')).strip()
            target_vid = str(g['movie_id'].iloc[0] or meta.get('movie_id', '')).strip()
            if 'SPLIT' in target_vid:
                target_vid = target_vid.split('SPLIT')[0]
            target_item = target_vid if target_vid in by_video or target_vid in catalog else ''
            success = fine != 'reject' and not coarse.startswith('reject')

            seek_row = seek_by_survey.get(str(meta.get('seek_survey_id', '')), {})
            seeker_movie_ids: List[str] = []
            turns: List[Dict] = []
            mentioned_all: List[str] = []

            for _, row in g.iterrows():
                speaker = 'system' if str(row.get('speaker', '')).upper() == 'RECOMMENDER' else 'user'
                titles: List[str] = []
                for title in parse_title_list(row.get('movies', '')):
                    item_id = title_to_id.get(title)
                    if item_id:
                        titles.append(item_id)
                        if item_id not in mentioned_all:
                            mentioned_all.append(item_id)
                        if speaker == 'user' and item_id not in seeker_movie_ids:
                            seeker_movie_ids.append(item_id)

                turns.append({
                    'turn_id': int(row.get('utt_id', 0) or 0),
                    'user_utterance': str(row.get('text', '')),
                    'speaker': speaker,
                    'intent': map_intent(row, 'user' if speaker == 'user' else 'system', titles),
                    'items_mentioned': titles,
                })

            if success and target_item and target_item not in seeker_movie_ids:
                seeker_movie_ids.append(target_item)

            conversations.append({
                'dialogue_id': str(dialog_id),
                'turns': turns,
                'user_id': str(seek_row.get('seeker_id', '') or f"seeker_{dialog_id}"),
                'user_profile': {
                    'user_id': str(seek_row.get('seeker_id', '') or f"seeker_{dialog_id}"),
                    'age_group': age_map_inspired(seek_row.get('age_group', '')),
                    'gender': gender_map(seek_row.get('gender', '')),
                    'preferences': seeker_movie_ids[:20],
                    'reason': str(seek_row.get('reason', '')),
                    'case': str(seek_row.get('case', fine)),
                },
                'accepted_items': [target_item] if success and target_item else [],
                'mentioned_items': mentioned_all,
                'success': bool(success),
                'dataset': 'INSPIRED',
                'metadata': {
                    'fine_label': fine,
                    'coarse_label': coarse,
                    'target_movie_id': target_item,
                    'target_video_id': target_vid,
                },
            })

        out_name = 'val_data.json' if split == 'dev' else f'{split}_data.json'
        with open(os.path.join(out_root, out_name), 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        return conversations

    train_convs = convert_split('train')
    dev_convs = convert_split('dev')
    test_convs = convert_split('test')

    with open(os.path.join(out_root, 'train_data_full.json'), 'w', encoding='utf-8') as f:
        json.dump(train_convs, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_root, 'item_catalog.json'), 'w', encoding='utf-8') as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    print(f"Converted INSPIRED -> {out_root}")
    print(f"Train: {len(train_convs)}, Val: {len(dev_convs)}, Test: {len(test_convs)}, Items: {len(catalog)}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert raw INSPIRED TSV files into MO-CRS JSON format')
    parser.add_argument('--raw_root', type=str, default='.', help='Folder containing the raw INSPIRED TSV files')
    parser.add_argument('--out_root', type=str, default='./INSPIRED', help='Output folder for converted JSON files')
    args = parser.parse_args()

    build_inspired_dataset(args.raw_root, args.out_root)
