"""
Demo script for MO-CRS
Interactive conversational recommendation demonstration
"""

import torch
import yaml
import os
import sys
import numpy as np
from typing import Dict, List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from mocrs import MOCRS
from data_utils import ItemCatalog


class InteractiveDemo:
    """
    Interactive demo of the conversational recommender system
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        print("="*70)
        print(" Multi-Objective Conversational Recommender System (MO-CRS) Demo")
        print("="*70)
        
        # Load config
        print("\n[1/4] Loading configuration...")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        print("  ✓ Configuration loaded")
        
        # Create device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  ✓ Using device: {self.device}")
        
        # Initialize model
        print("\n[2/4] Initializing MO-CRS model...")
        self.model = MOCRS(self.config).to(self.device)
        self.model.eval()
        print("  ✓ Model initialized")
        
        # Load item catalog
        print("\n[3/4] Loading item catalog...")
        self.item_catalog = ItemCatalog(self.config['data']['catalog_file'])
        print(f"  ✓ Loaded {self.item_catalog.num_items} items")
        
        # Initialize conversation state
        self.reset_conversation()
        
        print("\n[4/4] Demo ready!")
        print("="*70)
    
    def reset_conversation(self):
        """Reset conversation state"""
        self.conversation_history = []
        self.recommended_items = []
        self.last_candidate_item_ids = []
        self.turn_count = 0
    
    def process_user_input(self, user_utterance: str) -> Dict:
        """
        Process user input and generate system response
        
        Args:
            user_utterance: User's text input
            
        Returns:
            System response dictionary
        """
        self.turn_count += 1
        
        # Add to conversation history
        self.conversation_history.append({
            'turn': self.turn_count,
            'speaker': 'user',
            'utterance': user_utterance
        })
        
        # Prepare batch for model
        batch = self._create_model_batch(user_utterance)
        
        # Get model response
        with torch.no_grad():
            response = self.model.generate_response(batch)
        
        # Format response
        formatted_response = self._format_response(response)
        
        # Add system response to history
        self.conversation_history.append({
            'turn': self.turn_count,
            'speaker': 'system',
            'utterance': formatted_response['text'],
            'recommendations': formatted_response.get('items', [])
        })
        
        return formatted_response
    
    def _create_model_batch(self, utterance: str) -> Dict:
        """Create batch for model inference"""
        batch_size = 1
        
        # Sample candidate items
        candidate_items = self.item_catalog.sample_items(50)
        self.last_candidate_item_ids = candidate_items
        candidate_item_names = []
        for item_id in candidate_items:
            item = self.item_catalog.get_item(item_id)
            if item is None:
                candidate_item_names.append(str(item_id))
            else:
                candidate_item_names.append(item.get('name', item.get('title', str(item_id))))
        candidate_embeddings = [
            self.item_catalog.get_item_embedding(item_id)
            for item_id in candidate_items
        ]
        candidate_embeddings = np.asarray(candidate_embeddings, dtype=np.float32)
        candidate_embeddings = torch.from_numpy(candidate_embeddings).unsqueeze(0).to(self.device)
        
        # Static features (demographics) - dummy for demo
        static_features = torch.randn(
            batch_size,
            self.config['model']['personalization']['static_features_dim']
        ).to(self.device)
        
        # Dialogue history
        history = self.conversation_history[:-1] if len(self.conversation_history) > 0 else None
        
        batch = {
            'utterances': [utterance],
            'dialogue_history': history,
            'static_features': static_features,
            'candidate_items': candidate_embeddings,
            'candidate_item_ids': [candidate_items],
            'candidate_item_names': [candidate_item_names],
            'user_demographics': [{'age_group': '26-35', 'gender': 'U'}]
        }
        
        return batch
    
    def _format_response(self, model_response: Dict) -> Dict:
        """Format model response for display"""
        # Extract recommendations
        recommendations = model_response.get('recommendations', [[]])[0]
        
        # Get item details
        items = []
        if recommendations:
            for rec in recommendations[:5]:  # Top 5
                item_id = rec.get('item_id')
                resolved_item_id, item = self._resolve_item(item_id)
                if item:
                    items.append({
                        'name': item.get('name', item.get('title', f'Item {resolved_item_id}')),
                        'category': item.get('category', 'Unknown'),
                        'rating': item.get('rating', 'N/A'),
                        'score': rec['score']
                    })
        
        # Extract explanation
        explanations = model_response.get('explanations', [])
        explanation = explanations[0] if explanations else "Here are some recommendations for you."
        
        # Build response text
        if items:
            response_text = f"{explanation}\n\nTop Recommendations:"
            for i, item in enumerate(items, 1):
                response_text += f"\n  {i}. {item['name']} ({item['category']}) - Score: {item['score']:.3f}"
        else:
            response_text = "I'm still learning about your preferences. Could you tell me more about what you're looking for?"
        
        return {
            'text': response_text,
            'items': items,
            'explanation': explanation
        }

    def _resolve_item(self, rec_item_id):
        """Resolve a recommendation id/index into a catalog item."""
        # Case 1: rec id is an index into the most recent candidate list.
        if isinstance(rec_item_id, int) and self.last_candidate_item_ids:
            if 0 <= rec_item_id < len(self.last_candidate_item_ids):
                candidate_id = self.last_candidate_item_ids[rec_item_id]
                return candidate_id, self.item_catalog.get_item(candidate_id)

        # Case 2: rec id already matches a catalog key.
        key = str(rec_item_id)
        item = self.item_catalog.get_item(key)
        if item is not None:
            return key, item

        # Case 3: fallback by position in global item id list.
        if isinstance(rec_item_id, int) and 0 <= rec_item_id < len(self.item_catalog.item_ids):
            fallback_id = self.item_catalog.item_ids[rec_item_id]
            return fallback_id, self.item_catalog.get_item(fallback_id)

        return key, None
    
    def run_interactive(self):
        """Run interactive conversation loop"""
        print("\n" + "="*70)
        print("Interactive Mode")
        print("="*70)
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'reset' to start a new conversation")
        print("-"*70)
        
        while True:
            # Get user input
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            # Check for commands
            if user_input.lower() in ['quit', 'exit']:
                print("\nThank you for using MO-CRS! Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                self.reset_conversation()
                print("\n[Conversation reset]\n")
                continue
            
            # Process input
            try:
                response = self.process_user_input(user_input)
                print(f"\nSystem: {response['text']}")
            except Exception as e:
                print(f"\nError processing input: {e}")
                print("Please try again.")
    
    def run_demo_scenario(self):
        """Run pre-defined demo scenario"""
        print("\n" + "="*70)
        print("Demo Scenario Mode")
        print("="*70)
        
        # Demo conversation turns
        demo_turns = [
            "Hi, I'm looking for a good movie to watch tonight.",
            "I like action movies with good storytelling.",
            "Something recent would be nice, maybe from the last few years.",
            "That sounds interesting! What else do you have?",
            "I'll take the first recommendation. Thanks!"
        ]
        
        for i, utterance in enumerate(demo_turns, 1):
            print(f"\n{'='*70}")
            print(f"Turn {i}/{len(demo_turns)}")
            print(f"{'='*70}")
            print(f"\nUser: {utterance}")
            
            try:
                response = self.process_user_input(utterance)
                print(f"\nSystem: {response['text']}")
                
                # Pause for readability
                input("\n[Press Enter to continue...]")
                
            except Exception as e:
                print(f"\nError: {e}")
                break
        
        print(f"\n{'='*70}")
        print("Demo Scenario Complete!")
        print(f"{'='*70}")
        print(f"\nConversation Summary:")
        print(f"  Total turns: {self.turn_count}")
        print(f"  Items recommended: {len(self.recommended_items)}")


def main():
    """Main demo function"""
    print("\n" + "="*70)
    print(" Welcome to MO-CRS Demo")
    print("="*70)
    print("\nSelect mode:")
    print("  1. Interactive Mode (chat with the system)")
    print("  2. Demo Scenario (pre-defined conversation)")
    print("  3. Exit")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        return
    
    if choice == '3':
        print("\nGoodbye!")
        return
    
    # Initialize demo
    try:
        demo = InteractiveDemo()
    except Exception as e:
        print(f"\nError initializing demo: {e}")
        print("\nMake sure:")
        print("  1. config.yaml exists in the current directory")
        print("  2. All dependencies are installed (pip install -r requirements.txt)")
        print("  3. Model files are available")
        return
    
    # Run selected mode
    if choice == '1':
        demo.run_interactive()
    elif choice == '2':
        demo.run_demo_scenario()
    else:
        print("\nInvalid choice. Exiting...")


if __name__ == "__main__":
    main()
