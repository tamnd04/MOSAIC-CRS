"""
Multi-Objective Conversational Recommender System (MO-CRS)
"""

__version__ = '1.0.0'

from .mocrs import MOCRS
from .dialogue_state_tracker import DialogueStateTracker, BeliefStateTracker
from .personalization_engine import PersonalizationEngine
from .policy_network import MultiObjectivePolicyNetwork, PPOAgent
from .diversity_fairness_controller import DiversityFairnessController
from .explanation_generator import ExplanationGenerator
from .environment import ConversationalRecommenderEnv
from .data_utils import ReDialDataset, ItemCatalog, UserSimulator
from .evaluation import MOCRSEvaluator, evaluate_model

__all__ = [
    'MOCRS',
    'DialogueStateTracker',
    'BeliefStateTracker',
    'PersonalizationEngine',
    'MultiObjectivePolicyNetwork',
    'PPOAgent',
    'DiversityFairnessController',
    'ExplanationGenerator',
    'ConversationalRecommenderEnv',
    'ReDialDataset',
    'ItemCatalog',
    'UserSimulator',
    'MOCRSEvaluator',
    'evaluate_model'
]
