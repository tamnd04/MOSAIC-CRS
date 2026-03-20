"""
Convert ReDial data to the format expected by MO-CRS
Transforms ReDial's native JSON format to our dialogue format
"""

import json
import os
from collections import defaultdict

def convert_redial_to_mocrs(input_file, output_file):
    """
    Convert ReDial format to MO-CRS format
    
    ReDial format has:
    - messages: list of [sender_id, message_text]
    - movieMentions: {message_idx: [movie_ids]}
    - convers ationId: id
    
    MO-CRS format expects:
    - dialogue_id
    - turns: list of {utterance, intent, items_mentioned}
    - user_id
    - accepted_items
    """
    print(f"Converting {input_file} to MO-CRS format...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        redial_data = json.load(f)
    
    mocrs_dialogues = []
    
    for dialogue in redial_data:
        try:
            # Extract basic info
            dialogue_id = dialogue.get('conversationId', '')
            messages = dialogue.get('messages', [])
            movie_mentions = dialogue.get('movieMentions', {})
            
            if not messages:
                continue
            
            # Convert messages to turns
            turns = []
            for msg_idx, message in enumerate(messages):
                # Handle both formats (tuple and dict)
                if isinstance(message, dict):
                    message_text = message.get('text', '')
                    sender_id = message.get('senderWorkerId', '')
                else:
                    message_text = message[1] if len(message) > 1 else ''
                    sender_id = message[0] if len(message) > 0 else ''
                
                if not message_text:
                    continue
                
                turn = {
                    'user_utterance': message_text,  # Field name expected by code
                    'speaker': 'user' if msg_idx % 2 == 0 else 'system',
                    'intent': infer_intent(message_text),
                    'items_mentioned': []
                }
                
                # Add movie mentions for this message
                if str(msg_idx) in movie_mentions:
                    turn['items_mentioned'] = [str(m) for m in movie_mentions[str(msg_idx)]]
                
                turns.append(turn)
            
            if len(turns) < 2:
                continue
            
            # Create dialogue
            mocrs_dialogue = {
                'dialogue_id': str(dialogue_id),
                'turns': turns,
                'user_id': str(dialogue.get('initiatorWorkerId', 'unknown')),
                'accepted_items': extract_accepted_items(dialogue, movie_mentions)
            }
            
            mocrs_dialogues.append(mocrs_dialogue)
            
        except Exception as e:
            print(f"Error processing dialogue {dialogue.get('conversationId', 'unknown')}: {e}")
            continue
    
    # Save converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mocrs_dialogues, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Converted {len(mocrs_dialogues)} dialogues")
    return mocrs_dialogues

def infer_intent(utterance: str) -> str:
    """Infer intent from utterance text"""
    utterance_lower = utterance.lower()
    
    if any(w in utterance_lower for w in ['recommend', 'suggest', 'how about', 'would you like']):
        return 'recommend'
    elif any(w in utterance_lower for w in ['like', 'love', 'enjoy', 'prefer', 'favorite']):
        return 'provide_preference'
    elif any(w in utterance_lower for w in ['yes', 'yeah', 'ok', 'okay', 'sure', 'sounds good', 'great']):
        return 'accept'
    elif any(w in utterance_lower for w in ['no', 'nope', 'not interested', 'hate', 'dislike']):
        return 'reject'
    elif any(w in utterance_lower for w in ['tell', 'info', 'more', 'about', 'genre', 'year', 'actor']):
        return 'request_info'
    elif any(w in utterance_lower for w in ['thanks', 'bye', 'goodbye', 'quit', 'exit', 'stop']):
        return 'goodbye'
    else:
        return 'general'

def extract_accepted_items(dialogue: dict, movie_mentions: dict) -> list:
    """Extract items that were positively received"""
    accepted = []
    messages = dialogue.get('messages', [])
    
    for msg_idx, (_, message_text) in enumerate(messages):
        # If message contains acceptance and following message mentions movies
        if any(w in message_text.lower() for w in ['yes', 'okay', 'sure', 'sounds good', 'interested']):
            # Look for movies mentioned after acceptance
            for following_idx in range(msg_idx, min(msg_idx + 3, len(messages))):
                if str(following_idx) in movie_mentions:
                    accepted.extend(movie_mentions[str(following_idx)])
    
    return list(set(str(m) for m in accepted))

def main():
    """Convert all ReDial splits"""
    print("="*70)
    print("ReDial Format Converter for MO-CRS")
    print("="*70)
    
    # Convert train split
    if os.path.exists('train_data.json'):
        convert_redial_to_mocrs('train_data.json', 'train_data_mocrs.json')
    
    # Convert val split
    if os.path.exists('val_data.json'):
        convert_redial_to_mocrs('val_data.json', 'val_data_mocrs.json')
    
    # Convert test split
    if os.path.exists('test_data.json'):
        convert_redial_to_mocrs('test_data.json', 'test_data_mocrs.json')
    
    print("\n" + "="*70)
    print("Conversion Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - train_data_mocrs.json")
    print("  - val_data_mocrs.json")
    print("  - test_data_mocrs.json")
    print("\nNext steps:")
    print("  1. Backup original splits")
    print("  2. Rename *_mocrs.json files to replace original")
    print("  3. Re-run training")

if __name__ == '__main__':
    main()
