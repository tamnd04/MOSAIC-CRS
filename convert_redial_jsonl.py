"""
Convert ReDial JSONL data to the format expected by MO-CRS
Transforms ReDial's native JSONL format to our dialogue format with user_utterance field
"""

import json
import os

def infer_intent(utterance: str) -> str:
    """Infer intent from utterance text"""
    utterance_lower = utterance.lower()
    
    if any(w in utterance_lower for w in ['recommend', 'suggest', 'how about', 'would you like']):
        return 'recommend'
    elif any(w in utterance_lower for w in ['like', 'love', 'enjoy', 'prefer', 'favorite']):
        return 'provide_preference'
    elif any(w in utterance_lower for w in ['what', 'how', 'when', 'where', 'why', 'who']):
        return 'ask_question'
    elif any(w in utterance_lower for w in ['tell', 'explain', 'describe', 'about']):
        return 'provide_information'
    else:
        return 'general'

def format_movie_id(match_str: str) -> str:
    """Extract movie ID from @ID format"""
    if match_str.startswith('@'):
        return match_str[1:]
    return match_str

def convert_redial_jsonl_to_mocrs(input_file, output_file):
    """
    Convert ReDial JSONL format to MO-CRS format
    
    ReDial JSONL has one dialogue per line with structure:
    {
        "conversationId": "...",
        "messages": [{"text": "...", "senderWorkerId": 0/1, ...}, ...],
        "movieMentions": {"msg_idx": "movie_name", ...},
        "initiatorWorkerId": ...,
        "respondentWorkerId": ...,
        ...
    }
    
    MO-CRS format expected:
    [{
        "dialogue_id": "...",
        "turns": [{"user_utterance": "...", "intent": "...", "items_mentioned": [...]}, ...],
        "user_id": "...",
        "accepted_items": [...]
    }, ...]
    """
    print(f"Converting {input_file} to MO-CRS format...")
    
    mocrs_dialogues = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                    
                try:
                    dialogue = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
                
                try:
                    # Extract basic info
                    dialogue_id = dialogue.get('conversationId', '')
                    messages = dialogue.get('messages', [])
                    movie_mentions_dict = dialogue.get('movieMentions', {})
                    
                    if not messages or len(messages) < 2:
                        continue
                    
                    # Extract movie mentions into simpler dict: message_idx -> [movie_ids]
                    # movie_mentions_dict has format: {movie_id: movie_name}
                    # We need to find which messages mention these movies
                    movie_mentions_by_msg = {}
                    
                    # Build reverse lookup: message_idx with movies from messages text
                    for msg_idx, message in enumerate(messages):
                        msg_text = message.get('text', '')
                        mentioned_movieids = []
                        
                        # Find @ID patterns in message text
                        import re
                        for match in re.findall(r'@\d+', msg_text):
                            movie_id = match[1:]  # Remove @ prefix
                            mentioned_movieids.append(movie_id)
                        
                        if mentioned_movieids:
                            movie_mentions_by_msg[msg_idx] = mentioned_movieids
                    
                    # Convert messages to turns
                    turns = []
                    for msg_idx, message in enumerate(messages):
                        msg_text = message.get('text', '')
                        if not msg_text:
                            continue
                        
                        # Clean up movie mentions in text: replace @ID with ID
                        import re
                        cleaned_text = re.sub(r'@(\d+)', r'\1', msg_text)
                        
                        turn = {
                            'user_utterance': cleaned_text,  # Field name expected by code
                            'intent': infer_intent(msg_text),
                            'items_mentioned': movie_mentions_by_msg.get(msg_idx, [])
                        }
                        
                        turns.append(turn)
                    
                    if len(turns) < 2:
                        continue
                    
                    # Extract accepted items (movies mentioned in respondent's questions)
                    respondent_q = dialogue.get('respondentQuestions', {})
                    initiator_q = dialogue.get('initiatorQuestions', {})
                    
                    # Accepted items are those the user liked
                    accepted_items = []
                    for item_id in respondent_q:
                        if respondent_q[item_id].get('liked', 0) == 1:
                            accepted_items.append(str(item_id))
                    
                    for item_id in initiator_q if isinstance(initiator_q, dict) else []:
                        if initiator_q[item_id].get('liked', 0) == 1 and str(item_id) not in accepted_items:
                            accepted_items.append(str(item_id))
                    
                    # Create dialogue
                    mocrs_dialogue = {
                        'dialogue_id': str(dialogue_id),
                        'turns': turns,
                        'user_id': str(dialogue.get('initiatorWorkerId', 'unknown')),
                        'accepted_items': accepted_items
                    }
                    
                    mocrs_dialogues.append(mocrs_dialogue)
                    
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        return []
    
    # Save converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mocrs_dialogues, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Converted {len(mocrs_dialogues)} dialogues to {output_file}")
    return mocrs_dialogues

def main():
    """Convert all ReDial splits"""
    print("="*70)
    print("ReDial JSONL Format Converter for MO-CRS")
    print("="*70 + "\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Convert train split
    train_input = os.path.join(base_dir, 'data', 'train_data.jsonl')
    train_output = os.path.join(base_dir, 'data', 'train_data.json')
    if os.path.exists(train_input):
        convert_redial_jsonl_to_mocrs(train_input, train_output)
    else:
        print(f"[!] {train_input} not found")
    
    # Convert val split
    # Note: ReDial uses test split as validation in some scenarios
    test_input = os.path.join(base_dir, 'data', 'test_data.jsonl')
    val_output = os.path.join(base_dir, 'data', 'val_data.json')
    if os.path.exists(test_input):
        convert_redial_jsonl_to_mocrs(test_input, val_output)
    else:
        print(f"[!] {test_input} not found")
    
    # For test split, use a portion of train_data if separate test doesn't exist
    # Or just duplicate val for now
    test_output = os.path.join(base_dir, 'data', 'test_data.json')
    if os.path.exists(test_input):
        convert_redial_jsonl_to_mocrs(test_input, test_output)
    elif os.path.exists(train_input):
        # Use last 1000 from train as test
        dialogues = convert_redial_jsonl_to_mocrs(train_input, '')
        if len(dialogues) > 1000:
            test_dialogues = dialogues[-671:]  # Use 671 for test to match ReDial split
            with open(test_output, 'w', encoding='utf-8') as f:
                json.dump(test_dialogues, f, indent=2, ensure_ascii=False)
            print(f"[OK] Created test split with {len(test_dialogues)} dialogues")
    
    print("\n" + "="*70)
    print("Conversion Complete!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  - {train_output}")
    print(f"  - {val_output}")
    print(f"  - {test_output}")
    
if __name__ == '__main__':
    main()
