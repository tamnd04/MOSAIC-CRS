"""
Comprehensive test suite for MO-CRS
Tests all components and integration
"""

import sys
import os
import torch
import yaml
import traceback
from typing import Dict, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


class TestResult:
    """Container for test results"""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.time = 0.0


class MOCRSTester:
    """Comprehensive tester for MO-CRS"""
    
    def __init__(self):
        self.results = []
        self.config = None
        
    def run_all_tests(self):
        """Run all tests"""
        print("="*70)
        print(" MO-CRS Comprehensive Test Suite")
        print("="*70)
        
        # Configuration test
        self._test_config()
        
        # Component tests
        self._test_data_utils()
        self._test_dialogue_state_tracker()
        self._test_personalization_engine()
        self._test_policy_network()
        self._test_diversity_fairness_controller()
        self._test_explanation_generator()
        self._test_environment()
        
        # Integration tests
        self._test_mocrs_integration()
        
        # Print summary
        self._print_summary()
    
    def _test_config(self):
        """Test configuration loading"""
        test = TestResult("Configuration Loading")
        
        try:
            config_path = 'config.yaml'
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Override data paths to use test_data folder instead of real data
            # This prevents tests from interfering with actual training data
            self.config['data']['data_dir'] = './test_data'
            self.config['data']['catalog_file'] = './test_data/item_catalog.json'
            self.config['data']['processed_dir'] = './test_data/processed'
            
            # Verify required sections
            required = ['model', 'training', 'data', 'environment', 'logging']
            for section in required:
                assert section in self.config, f"Missing section: {section}"
            
            test.passed = True
            print("✓ Configuration loading passed")
            print("  → Using test_data folder for tests")
            
        except Exception as e:
            test.error = str(e)
            print(f"✗ Configuration loading failed: {e}")
        
        self.results.append(test)
    
    def _test_data_utils(self):
        """Test data utilities"""
        test = TestResult("Data Utilities")
        
        try:
            from data_utils import ItemCatalog, UserSimulator, create_dummy_data
            
            # Test ItemCatalog
            catalog = ItemCatalog(self.config['data']['catalog_file'])
            assert catalog.num_items > 0, "Empty catalog"
            
            # Test sampling
            samples = catalog.sample_items(10)
            assert len(samples) <= 10, "Invalid sample size"
            
            # Test UserSimulator
            simulator = UserSimulator(catalog, self.config)
            simulator.reset()
            preference = simulator.provide_preference()
            assert isinstance(preference, str), "Invalid preference type"
            
            test.passed = True
            print("✓ Data utilities test passed")
            
        except Exception as e:
            test.error = str(e)
            print(f"✗ Data utilities test failed: {e}")
            traceback.print_exc()
        
        self.results.append(test)
    
    def _test_dialogue_state_tracker(self):
        """Test Dialogue State Tracker"""
        test = TestResult("Dialogue State Tracker")
        
        try:
            from dialogue_state_tracker import DialogueStateTracker, BeliefStateTracker
            
            dst = DialogueStateTracker(self.config)
            
            # Test forward pass
            utterances = ["I'm looking for a good movie", "Something with action"]
            outputs = dst(utterances)
            
            assert 'state' in outputs, "Missing state output"
            assert 'intent_probs' in outputs, "Missing intent probs"
            assert 'sentiment_probs' in outputs, "Missing sentiment probs"
            
            # Test belief state tracker
            bst = BeliefStateTracker(self.config)
            dialogue_states = torch.randn(2, 5, self.config['model']['dialogue_state_tracker']['state_dim'])
            bst_outputs = bst(dialogue_states)
            
            assert 'belief_state' in bst_outputs, "Missing belief state"
            
            test.passed = True
            print("✓ Dialogue State Tracker test passed")
            
        except Exception as e:
            test.error = str(e)
            print(f"✗ Dialogue State Tracker test failed: {e}")
            traceback.print_exc()
        
        self.results.append(test)
    
    def _test_personalization_engine(self):
        """Test Personalization Engine"""
        test = TestResult("Personalization Engine")
        
        try:
            from personalization_engine import PersonalizationEngine
            
            pe = PersonalizationEngine(self.config)
            
            # Test forward pass
            batch_size = 2
            static_features = torch.randn(batch_size, 
                self.config['model']['personalization']['static_features_dim'])
            dialogue_states = torch.randn(batch_size, 10, 
                self.config['model']['dialogue_state_tracker']['state_dim'])
            
            outputs = pe(static_features, dialogue_states)
            
            assert 'user_profile' in outputs, "Missing user profile"
            assert outputs['user_profile'].shape[0] == batch_size, "Invalid batch size"
            
            test.passed = True
            print("✓ Personalization Engine test passed")
            
        except Exception as e:
            test.error = str(e)
            print(f"✗ Personalization Engine test failed: {e}")
            traceback.print_exc()
        
        self.results.append(test)
    
    def _test_policy_network(self):
        """Test Policy Network"""
        test = TestResult("Policy Network")
        
        try:
            from policy_network import MultiObjectivePolicyNetwork
            
            policy = MultiObjectivePolicyNetwork(self.config)
            
            # Test forward pass
            batch_size = 4
            state = torch.randn(batch_size, self.config['model']['policy']['state_dim'])
            
            outputs = policy(state)
            
            assert 'action_probs' in outputs, "Missing action probs"
            assert 'q_values' in outputs, "Missing Q-values"
            assert 'values' in outputs, "Missing values"
            
            # Test action selection
            actions, log_probs = policy.select_action(state)
            assert actions.shape[0] == batch_size, "Invalid action shape"
            
            test.passed = True
            print("✓ Policy Network test passed")
            
        except Exception as e:
            test.error = str(e)
            print(f"✗ Policy Network test failed: {e}")
            traceback.print_exc()
        
        self.results.append(test)
    
    def _test_diversity_fairness_controller(self):
        """Test Diversity & Fairness Controller"""
        test = TestResult("Diversity & Fairness Controller")
        
        try:
            from diversity_fairness_controller import DiversityFairnessController
            
            dfc = DiversityFairnessController(self.config)
            
            # Test reranking
            num_candidates = 50
            candidate_items = torch.randn(num_candidates, 
                self.config['model']['diversity_fairness']['item_embedding_dim'])
            candidate_scores = torch.rand(num_candidates)
            user_embedding = torch.randn(
                self.config['model']['diversity_fairness']['user_embedding_dim'])
            
            outputs = dfc(candidate_items, candidate_scores, user_embedding)
            
            assert 'reranked_indices' in outputs, "Missing reranked indices"
            assert 'diversity_scores' in outputs, "Missing diversity scores"
            assert 'fairness_scores' in outputs, "Missing fairness scores"
            
            test.passed = True
            print("✓ Diversity & Fairness Controller test passed")
            
        except Exception as e:
            test.error = str(e)
            print(f"✗ Diversity & Fairness Controller test failed: {e}")
            traceback.print_exc()
        
        self.results.append(test)
    
    def _test_explanation_generator(self):
        """Test Explanation Generator"""
        test = TestResult("Explanation Generator")
        
        try:
            from explanation_generator import ExplanationGenerator
            
            # Use template mode to avoid GPT-2 download
            self.config['model']['explanation_generator']['generation_mode'] = 'template'
            eg = ExplanationGenerator(self.config)
            
            # Test explanation generation
            context = torch.randn(2, self.config['model']['explanation_generator']['context_dim'])
            item_info = {'name': 'Test Movie', 'genre': 'Action'}
            
            outputs = eg(context, item_info)
            
            assert 'explanations' in outputs, "Missing explanations"
            assert len(outputs['explanations']) > 0, "Empty explanations"
            assert isinstance(outputs['explanations'][0], str), "Invalid explanation type"
            
            test.passed = True
            print("✓ Explanation Generator test passed")
            
        except Exception as e:
            test.error = str(e)
            print(f"✗ Explanation Generator test failed: {e}")
            traceback.print_exc()
        
        self.results.append(test)
    
    def _test_environment(self):
        """Test Environment"""
        test = TestResult("Environment")
        
        try:
            from environment import ConversationalRecommenderEnv
            from data_utils import ItemCatalog
            
            catalog = ItemCatalog(self.config['data']['catalog_file'])
            env = ConversationalRecommenderEnv(self.config, catalog, mode='train')
            
            # Test reset
            state = env.reset()
            assert state is not None, "Invalid initial state"
            
            # Test step
            action = {'action_type': 1, 'items': catalog.sample_items(5)}
            next_state, rewards, done, info = env.step(action)
            
            assert next_state is not None, "Invalid next state"
            assert isinstance(rewards, dict), "Invalid rewards type"
            assert 'accuracy' in rewards, "Missing accuracy reward"
            
            test.passed = True
            print("✓ Environment test passed")
            
        except Exception as e:
            test.error = str(e)
            print(f"✗ Environment test failed: {e}")
            traceback.print_exc()
        
        self.results.append(test)
    
    def _test_mocrs_integration(self):
        """Test complete MO-CRS integration"""
        test = TestResult("MO-CRS Integration")
        
        try:
            from mocrs import MOCRS
            
            model = MOCRS(self.config)
            
            # Test forward pass
            batch_size = 2
            batch = {
                'utterances': ['I want an action movie', 'Something exciting'],
                'dialogue_history': None,
                'static_features': torch.randn(batch_size, 
                    self.config['model']['personalization']['static_features_dim']),
                'candidate_items': torch.randn(batch_size, 50,
                    self.config['model']['diversity_fairness']['item_embedding_dim']),
                'user_demographics': [
                    {'age_group': '26-35', 'gender': 'M'},
                    {'age_group': '36-45', 'gender': 'F'}
                ]
            }
            
            outputs = model(batch)
            
            assert 'dialogue_state' in outputs, "Missing dialogue state"
            assert 'user_profile' in outputs, "Missing user profile"
            assert 'actions' in outputs, "Missing actions"
            assert 'reranked_indices' in outputs, "Missing reranked indices"
            assert 'explanations' in outputs, "Missing explanations"
            
            # Test response generation
            response = model.generate_response(batch)
            
            assert 'recommendations' in response, "Missing recommendations"
            assert 'explanations' in response, "Missing explanations"
            
            test.passed = True
            print("✓ MO-CRS Integration test passed")
            
        except Exception as e:
            test.error = str(e)
            print(f"✗ MO-CRS Integration test failed: {e}")
            traceback.print_exc()
        
        self.results.append(test)
    
    def _print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print(" Test Summary")
        print("="*70)
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {passed/total*100:.1f}%")
        
        if passed < total:
            print("\nFailed Tests:")
            for result in self.results:
                if not result.passed:
                    print(f"  ✗ {result.name}")
                    if result.error:
                        print(f"    Error: {result.error}")
        
        print("\n" + "="*70)
        
        if passed == total:
            print("✓ ALL TESTS PASSED!")
        else:
            print(f"✗ {total - passed} TESTS FAILED")
        
        print("="*70)


def main():
    """Main test function"""
    tester = MOCRSTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
