"""
Convert ReDial dataset files to the format expected by MO-CRS
- Converts JSONL to JSON
- Converts movies CSV to item catalog JSON
- Splits test data into validation and test sets
"""

import json
import csv
import os
from collections import defaultdict

def convert_jsonl_to_json(jsonl_file, json_file):
    """Convert JSONL file to JSON list"""
    print(f"Converting {jsonl_file} to {json_file}...")
    
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Converted {len(data)} records")
    return data

def convert_movies_csv_to_catalog(csv_file, catalog_file):
    """Convert movies CSV to item catalog JSON"""
    print(f"Converting {csv_file} to {catalog_file}...")
    
    catalog = {}
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            movie_id = row.get('movieId', row.get('id', ''))
            
            if not movie_id:
                continue
            
            # Extract movie information
            catalog[movie_id] = {
                'title': row.get('movieName', row.get('title', f'Movie {movie_id}')),
                'year': row.get('year', ''),
                'rating': float(row.get('rating', 0)) if row.get('rating') else 0.0,
                'genres': row.get('genres', '').split('|') if row.get('genres') else [],
                'category': row.get('genres', '').split('|')[0] if row.get('genres') else 'Unknown',
                'mentions': int(row.get('mentions', 0)) if row.get('mentions') else 0
            }
    
    with open(catalog_file, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2)
    
    print(f"✓ Converted {len(catalog)} movies")
    return catalog

def split_test_data(test_data, val_file, test_file, val_ratio=0.5):
    """Split test data into validation and test sets"""
    print(f"Splitting test data into validation and test sets...")
    
    split_idx = int(len(test_data) * val_ratio)
    val_data = test_data[:split_idx]
    test_data_split = test_data[split_idx:]
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data_split, f, indent=2)
    
    print(f"✓ Created {len(val_data)} validation samples")
    print(f"✓ Created {len(test_data_split)} test samples")

def main():
    """Main conversion process"""
    print("="*70)
    print("ReDial Dataset Conversion")
    print("="*70)
    
    # Convert training data
    if os.path.exists('train_data.jsonl'):
        convert_jsonl_to_json('train_data.jsonl', 'train_data.json')
    else:
        print("Warning: train_data.jsonl not found")
    
    # Convert test data and split
    if os.path.exists('test_data.jsonl'):
        test_data = convert_jsonl_to_json('test_data.jsonl', 'test_data_full.json')
        split_test_data(test_data, 'val_data.json', 'test_data.json')
    else:
        print("Warning: test_data.jsonl not found")
    
    # Convert movies catalog
    if os.path.exists('movies_with_mentions.csv'):
        convert_movies_csv_to_catalog('movies_with_mentions.csv', 'item_catalog_redial.json')
        print("\nNote: Created 'item_catalog_redial.json' from ReDial movies")
        print("You can rename it to 'item_catalog.json' to replace the dummy catalog")
    else:
        print("Warning: movies_with_mentions.csv not found")
    
    print("\n" + "="*70)
    print("Conversion Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - train_data.json (training dialogues)")
    print("  - val_data.json (validation dialogues)")
    print("  - test_data.json (test dialogues)")
    print("  - item_catalog_redial.json (movie catalog)")
    print("\nNext steps:")
    print("  1. Backup current item_catalog.json if needed")
    print("  2. Rename item_catalog_redial.json to item_catalog.json")
    print("  3. Run training: cd src && python train.py --mode both")

if __name__ == '__main__':
    main()
