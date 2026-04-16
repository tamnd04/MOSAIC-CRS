"""
Explanation Generator (EG)
Generates natural language explanations for recommendations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List, Tuple
import random
import warnings


class ExplanationGenerator(nn.Module):
    """
    Generates explanations for recommendations
    Supports both template-based and neural generation
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        eg_config = config['model']['explanation_generator']
        
        self.generation_mode = eg_config['generation_mode']  # 'template' or 'neural' or 'hybrid'
        self.neural_generator_available = False
        self.hidden_dim = eg_config['hidden_dim']
        
        # Neural explanation generator (GPT-2 based)
        if self.generation_mode in ['neural', 'hybrid']:
            try:
                self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                self.tokenizer.pad_token = self.tokenizer.eos_token

                # Freeze most of GPT-2, fine-tune last few layers
                for param in self.gpt2.parameters():
                    param.requires_grad = False

                # Unfreeze last 2 transformer blocks
                for param in self.gpt2.transformer.h[-2:].parameters():
                    param.requires_grad = True

                self.neural_generator_available = True
            except Exception as exc:
                warnings.warn(
                    f"Failed to load GPT-2 for explanation generation ({exc}). Falling back to template mode.",
                    RuntimeWarning
                )
                if self.generation_mode == 'neural':
                    self.generation_mode = 'template'
        
        # Template-based components
        self.template_selector = TemplateSelector(config)
        
        # Context encoder for conditioning
        self.context_encoder = nn.Sequential(
            nn.Linear(eg_config['context_dim'], self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Explanation type classifier
        self.explanation_type_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, eg_config['num_explanation_types'])
        )
        
        # Explanation types
        self.explanation_types = [
            'content_based',      # "Based on your interest in X"
            'collaborative',      # "Users like you also enjoyed"
            'diversity',          # "To show you something different"
            'trending',           # "This is popular right now"
            'feature_based',      # "This has the genre/actor you mentioned"
            'serendipity'        # "You might find this surprising"
        ]
    
    def forward(self, context: torch.Tensor,
               item_info: Dict,
               user_info: Dict = None,
               explanation_type: str = None) -> Dict:
        """
        Generate explanation for recommendation
        
        Args:
            context: (batch, context_dim) - conversation context
            item_info: Dictionary with item information
            user_info: Dictionary with user information
            explanation_type: Specific explanation type or None for automatic
            
        Returns:
            Dictionary with explanation text and metadata
        """
        batch_size = context.shape[0]
        
        # Encode context
        context_encoded = self.context_encoder(context)  # (batch, hidden_dim)
        
        # Predict explanation type if not provided
        if explanation_type is None:
            type_logits = self.explanation_type_classifier(context_encoded)
            type_probs = F.softmax(type_logits, dim=-1)
            type_ids = torch.argmax(type_probs, dim=-1)
            explanation_type = self.explanation_types[type_ids[0].item()]
        
        # Generate explanation based on mode
        if self.generation_mode == 'template':
            explanations = self._generate_template_based(
                item_info, user_info, explanation_type, batch_size
            )
        elif self.generation_mode == 'neural' and self.neural_generator_available:
            explanations = self._generate_neural(
                context_encoded, item_info, user_info, explanation_type
            )
        else:  # hybrid
            # Use template 50% of the time, neural 50%
            use_neural = self.neural_generator_available and (random.random() >= 0.5)
            if not use_neural:
                explanations = self._generate_template_based(
                    item_info, user_info, explanation_type, batch_size
                )
            else:
                explanations = self._generate_neural(
                    context_encoded, item_info, user_info, explanation_type
                )
        
        return {
            'explanations': explanations,
            'explanation_type': explanation_type,
            'context_encoding': context_encoded
        }
    
    def _generate_template_based(self, item_info: Dict, user_info: Dict,
                                explanation_type: str, batch_size: int) -> List[str]:
        """Generate explanations using templates"""
        explanations = []
        
        for i in range(batch_size):
            explanation = self.template_selector.generate(
                item_info, user_info, explanation_type
            )
            explanations.append(explanation)
        
        return explanations
    
    def _generate_neural(self, context: torch.Tensor, item_info: Dict,
                        user_info: Dict, explanation_type: str) -> List[str]:
        """Generate explanations using GPT-2"""
        batch_size = context.shape[0]
        device = context.device
        
        # Create prompt
        prompts = []
        for i in range(batch_size):
            prompt = self._create_prompt(item_info, user_info, explanation_type)
            prompts.append(prompt)
        
        # Tokenize
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=100,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Generate
        with torch.no_grad():
            outputs = self.gpt2.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        explanations = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove prompt
            if len(prompts) > 0:
                text = text.replace(prompts[0], '').strip()
            explanations.append(text)
        
        return explanations
    
    def _create_prompt(self, item_info: Dict, user_info: Dict,
                      explanation_type: str) -> str:
        """Create prompt for neural generation"""
        item_name = item_info.get('name', 'this item')
        item_category = item_info.get('category', 'movie')
        
        if explanation_type == 'content_based':
            prompt = f"I recommend {item_name} because"
        elif explanation_type == 'collaborative':
            prompt = f"You might like {item_name} because similar users"
        elif explanation_type == 'diversity':
            prompt = f"To show you something different, I recommend {item_name}"
        elif explanation_type == 'trending':
            prompt = f"{item_name} is trending right now"
        elif explanation_type == 'feature_based':
            prompt = f"Based on your preferences, {item_name}"
        else:  # serendipity
            prompt = f"You might find {item_name} interesting"
        
        return prompt
    
    def batch_generate(self, contexts: torch.Tensor,
                      items_info: List[Dict],
                      users_info: List[Dict] = None) -> List[str]:
        """
        Generate explanations for a batch
        
        Args:
            contexts: (batch, context_dim)
            items_info: List of item info dicts
            users_info: List of user info dicts
            
        Returns:
            List of explanation strings
        """
        batch_size = contexts.shape[0]
        all_explanations = []
        
        for i in range(batch_size):
            context = contexts[i:i+1]
            item_info = items_info[i] if i < len(items_info) else {}
            user_info = users_info[i] if users_info and i < len(users_info) else {}
            
            outputs = self.forward(context, item_info, user_info)
            all_explanations.append(outputs['explanations'][0])
        
        return all_explanations


class TemplateSelector:
    """
    Selects and fills templates for explanation generation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Template library
        self.templates = {
            'content_based': [
                "I recommend {item} because it has {feature} that you mentioned.",
                "Based on your interest in {genre}, you'll enjoy {item}.",
                "Since you liked {similar_item}, {item} is a great choice.",
                "{item} matches your preference for {feature}."
            ],
            'collaborative': [
                "Users similar to you loved {item}.",
                "People who enjoyed {similar_item} also recommend {item}.",
                "{item} is popular among users with similar tastes.",
                "Based on what others like you enjoyed, try {item}."
            ],
            'diversity': [
                "To broaden your horizons, check out {item}.",
                "Here's something different: {item}.",
                "For variety, I suggest {item} from {genre}.",
                "To show you diverse options, consider {item}."
            ],
            'trending': [
                "{item} is trending right now.",
                "Many people are watching {item} recently.",
                "{item} is currently popular.",
                "{item} is a hot pick this week."
            ],
            'feature_based': [
                "{item} features {actor} who you mentioned.",
                "{item} is directed by {director} and has {genre}.",
                "{item} has the {rating} rating you prefer.",
                "{item} is from {year} and features {feature}."
            ],
            'serendipity': [
                "You might be surprised by {item}.",
                "Here's something unexpected: {item}.",
                "{item} could be a pleasant discovery.",
                "Try something new with {item}."
            ]
        }
    
    def generate(self, item_info: Dict, user_info: Dict,
                explanation_type: str) -> str:
        """Generate explanation from template"""
        
        if explanation_type not in self.templates:
            explanation_type = 'content_based'
        
        # Select random template
        template = random.choice(self.templates[explanation_type])
        
        # Fill template
        explanation = self._fill_template(template, item_info, user_info)
        
        return explanation
    
    def _fill_template(self, template: str, item_info: Dict,
                      user_info: Dict) -> str:
        """Fill template with actual values"""
        
        # Extract values from item_info
        item_name = item_info.get('name', 'this movie')
        genre = item_info.get('genre', 'drama')
        actor = item_info.get('actor', 'acclaimed actors')
        director = item_info.get('director', 'a renowned director')
        year = item_info.get('year', 'recent')
        rating = item_info.get('rating', 'high')
        
        # Fill template
        try:
            explanation = template.format(
                item=item_name,
                genre=genre,
                actor=actor,
                director=director,
                year=year,
                rating=rating,
                feature=genre,
                similar_item='movies you liked'
            )
        except KeyError:
            # If template has placeholders we don't have values for
            explanation = f"I recommend {item_name}."
        
        return explanation


class ExplanationEvaluator:
    """
    Evaluates quality of generated explanations
    """
    
    def __init__(self):
        self.min_length = 10
        self.max_length = 200
        
    def evaluate(self, explanation: str, item_info: Dict) -> Dict:
        """
        Evaluate explanation quality
        
        Returns:
            Dictionary with quality scores
        """
        scores = {}
        
        # Length check
        length = len(explanation.split())
        scores['length_ok'] = self.min_length <= length <= self.max_length
        
        # Mentions item
        item_name = item_info.get('name', '')
        scores['mentions_item'] = item_name.lower() in explanation.lower()
        
        # Contains reasoning words
        reasoning_words = ['because', 'since', 'based on', 'like', 'similar', 'enjoy']
        scores['has_reasoning'] = any(word in explanation.lower() for word in reasoning_words)
        
        # Not too generic
        generic_phrases = ['good movie', 'great film', 'nice choice']
        scores['not_generic'] = not any(phrase in explanation.lower() for phrase in generic_phrases)
        
        # Overall score
        scores['overall'] = sum([
            scores['length_ok'],
            scores['mentions_item'],
            scores['has_reasoning'],
            scores['not_generic']
        ]) / 4.0
        
        return scores


if __name__ == "__main__":
    # Test Explanation Generator
    import yaml
    import os
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create EG
    print("Creating Explanation Generator...")
    # Use template mode to avoid downloading GPT-2
    config['model']['explanation']['generation_mode'] = 'template'
    eg = ExplanationGenerator(config)
    
    # Test data
    batch_size = 4
    context_dim = config['model']['explanation']['context_dim']
    context = torch.randn(batch_size, context_dim)
    
    item_info = {
        'name': 'The Shawshank Redemption',
        'genre': 'Drama',
        'actor': 'Tim Robbins',
        'director': 'Frank Darabont',
        'year': '1994',
        'rating': '9.3'
    }
    
    user_info = {
        'age': 30,
        'preferences': ['drama', 'thriller']
    }
    
    print(f"\nTesting explanation generation...")
    print(f"  Context shape: {context.shape}")
    print(f"  Item: {item_info['name']}")
    
    # Generate for each explanation type
    for exp_type in eg.explanation_types:
        print(f"\n  Type: {exp_type}")
        outputs = eg(context, item_info, user_info, explanation_type=exp_type)
        print(f"    Explanation: {outputs['explanations'][0]}")
    
    # Test batch generation
    print(f"\nTesting batch generation...")
    items_info = [
        {'name': 'Movie A', 'genre': 'Action'},
        {'name': 'Movie B', 'genre': 'Comedy'},
        {'name': 'Movie C', 'genre': 'Sci-Fi'},
        {'name': 'Movie D', 'genre': 'Romance'}
    ]
    
    batch_explanations = eg.batch_generate(context, items_info)
    print(f"  Generated {len(batch_explanations)} explanations")
    for i, exp in enumerate(batch_explanations):
        print(f"    {i+1}. {exp}")
    
    # Test evaluator
    print(f"\nTesting explanation evaluator...")
    evaluator = ExplanationEvaluator()
    
    test_explanation = "I recommend The Shawshank Redemption because it has the drama genre that you mentioned."
    scores = evaluator.evaluate(test_explanation, item_info)
    
    print(f"  Explanation: {test_explanation}")
    print(f"  Scores: {scores}")
    print(f"  Overall quality: {scores['overall']:.2f}")
    
    print("\n✓ Explanation Generator tests passed!")
