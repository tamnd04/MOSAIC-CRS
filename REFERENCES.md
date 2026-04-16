# References and Key Papers

## Conversational Recommender Systems

### Core CRS Papers

1. **Deep Conversational Recommendation Systems: A Survey**
   - Li et al. (2020)
   - Comprehensive overview of CRS approaches
   - Key concepts: dialogue management, knowledge integration

2. **Towards Knowledge-Based Recommender Dialog System (KBRD)**
   - Chen et al. (2019)
   - Knowledge graph integration in CRS
   - Multi-hop reasoning for recommendations

3. **Towards Conversational Recommendation over Multi-Type Dialogs**
   - Zhou et al. (2020)
   - Handles different dialogue types
   - Graph-based conversation modeling

4. **ReDial: A Dataset for Conversational Recommendation**
   - Li et al. (2018)
   - Benchmark dataset for CRS
   - Movie recommendations in natural conversations

5. **Conversation-Based Recommendation: A Survey**
   - Jannach et al. (2021)
   - Taxonomy of CRS approaches
   - Evaluation methodologies

## Reinforcement Learning

### RL for Dialogue

6. **Deep Reinforcement Learning for Dialogue Generation**
   - Li et al. (2016)
   - Sequence-to-sequence with RL
   - Reward shaping for dialogue quality

7. **Composite Task-Completion Dialogue Policy Learning via Hierarchical Deep RL**
   - Peng et al. (2017)
   - Hierarchical RL for complex dialogues
   - Subtask decomposition

8. **End-to-End Task-Completion Neural Dialogue Systems**
   - Liu & Lane (2017)
   - Neural dialogue state tracking
   - Policy learning with user simulation

### Multi-Objective RL

9. **Multi-Objective Reinforcement Learning: A Comprehensive Overview**
   - Van Moffaert & Nowe (2014)
   - Survey of MORL techniques
   - Scalarization vs Pareto approaches

10. **A Practical Guide to Multi-Objective Reinforcement Learning and Planning**
   - Roijers et al. (2013)
   - Practical algorithms for MORL
   - Empirical comparisons

11. **Multi-Objective Deep Reinforcement Learning**
   - Abels et al. (2019)
   - Neural network approaches to MORL
   - Applications in games

### PPO and Advanced RL

12. **Proximal Policy Optimization Algorithms**
   - Schulman et al. (2017)
   - PPO algorithm details
   - Performance comparisons with TRPO

13. **Generalized Advantage Estimation**
   - Schulman et al. (2016)
   - GAE for variance reduction
   - Bias-variance trade-off analysis

14. **Playing Atari with Deep Reinforcement Learning (DQN)**
   - Mnih et al. (2013)
   - Foundation of deep RL
   - Experience replay and target networks

### Off-Policy Evaluation

15. **Doubly Robust Policy Evaluation and Learning**
   - Dudik et al. (2011)
   - Counterfactual evaluation with doubly robust estimators
   - Low variance and reduced bias compared with IPS alone

16. **Counterfactual Risk Minimization: Learning from Logged Bandit Feedback**
   - Swaminathan & Joachims (2015)
   - Importance weighting and variance control for logged feedback
   - Foundations for IPS/SNIPS-style evaluation

## Diversity in Recommendation

### Diversity Algorithms

17. **Improving Recommendation Lists Through Topic Diversification**
    - Ziegler et al. (2005)
    - Topic-based diversification
    - User studies on diversity preference

18. **The Use of MMR, Diversity-Based Reranking for Reordering Documents**
    - Carbonell & Goldstein (1998)
    - Maximal Marginal Relevance
    - Information retrieval foundations

19. **Diversity in Recommender Systems - A Survey**
   - Kunaver & Pozrl (2017)
    - Comprehensive diversity survey
    - Taxonomy of diversity approaches

20. **Calibrated Recommendations**
    - Steck (2018)
    - Matching user distribution
    - Bayesian approaches

## Fairness in Recommendation

### User-Side Fairness

21. **Beyond Parity: Fairness Objectives for Collaborative Filtering**
    - Yao & Huang (2017)
    - Fairness metrics for CF
    - Parity vs calibration

22. **Fairness-Aware Ranking in Search & Recommendation Systems**
    - Singh & Joachims (2018)
    - Exposure-based fairness
    - Policy learning for fair ranking

23. **Fairness in Recommendation Ranking through Pairwise Comparisons**
    - Zehlike & Castillo (2020)
    - FA*IR algorithm
    - Demographic parity in ranking

### Item-Side Fairness

24. **Multistakeholder Recommendation: Survey and Research Directions**
    - Abdollahpouri et al. (2020)
    - Provider fairness
    - Multi-sided marketplaces

25. **Managing Popularity Bias in Recommender Systems with Personalized Re-ranking**
    - Abdollahpouri et al. (2019)
    - Long-tail promotion
    - Calibrated popularity

26. **Fairness of Exposure in Rankings**
    - Diaz et al. (2020)
    - Exposure metrics
    - Fair ranking algorithms

### Fairness Frameworks

27. **AI Fairness 360: An Extensible Toolkit for Detecting and Mitigating Algorithmic Bias**
    - Bellamy et al. (2019)
    - Comprehensive fairness toolkit
    - Multiple fairness metrics

## Personalization

### User Modeling

28. **Attention Is All You Need**
    - Vaswani et al. (2017)
    - Transformer architecture
    - Self-attention mechanism

29. **Neural Attentive Session-based Recommendation (NARM)**
    - Li et al. (2017)
    - RNN with attention for sessions
    - Sequential recommendation

30. **Self-Attentive Sequential Recommendation (SASRec)**
    - Kang & McAuley (2018)
    - Transformer for sequences
    - Item-to-item attention

### Meta-Learning & Cold Start

31. **Model-Agnostic Meta-Learning (MAML)**
    - Finn et al. (2017)
    - Fast adaptation with few examples
    - Gradient-based meta-learning

32. **Meta-Learning for User Cold-Start Recommendation**
    - Vartak et al. (2017)
    - Meta-learning in RecSys
    - Transfer learning approaches

### Contextual Bandits

33. **A Contextual-Bandit Approach to Personalized News Article Recommendation**
    - Li et al. (2010)
    - LinUCB algorithm
    - Online learning for recommendation

34. **Thompson Sampling for Contextual Bandits with Linear Payoffs**
    - Agrawal & Goyal (2013)
    - Bayesian approach to bandits
    - Theoretical guarantees

## Explainable AI & Recommendation

### Explanation Generation

35. **Explainable Recommendation: A Survey and New Perspectives**
    - Zhang & Chen (2020)
    - Comprehensive XAI survey
    - Taxonomy of explanation types

36. **Explainable Recommendation via Multi-Task Learning**
    - Chen et al. (2019)
    - Joint training for recommendation and explanation
    - Neural explanation generation

37. **Neural Template Extraction for Explainable Recommendation**
    - Li et al. (2021)
    - Template-based explanations
    - Interpretability vs flexibility

### Faithfulness & Trust

38. **Towards Faithful Neural Table-to-Text Generation with Content-Matching Constraints**
    - Ma et al. (2019)
    - Faithful text generation
    - Constraints for accuracy

39. **Explanation in Recommender Systems**
    - Tintarev & Masthoff (2007)
    - Psychology of explanations
    - Trust and persuasiveness

## Natural Language Processing

### Pre-trained Models

40. **BERT: Pre-training of Deep Bidirectional Transformers**
    - Devlin et al. (2019)
    - Bidirectional pre-training
    - Fine-tuning for downstream tasks

41. **GPT-2: Language Models are Unsupervised Multitask Learners**
    - Radford et al. (2019)
    - Large-scale language modeling
    - Zero-shot task transfer

### Dialogue Systems

42. **A Neural Conversational Model**
    - Vinyals & Le (2015)
    - Sequence-to-sequence for dialogue
    - End-to-end learning

43. **Hybrid Code Networks: Practical and Efficient End-to-End Dialog Control**
    - Williams et al. (2017)
    - Combining learning and engineering
    - Practical dialogue systems

## Datasets

### Conversational Recommendation

44. **ReDial Dataset**
   - Movie recommendations through conversation
   - ~10K conversations
   - Available: redialdata.github.io

45. **GoRecDial: Goal-Oriented Conversational Recommendation**
   - Task-oriented conversations
   - Explicit user goals
   - Available: github.com/salesforce/gorecommend

46. **Inspired Dataset**
   - Social recommendations
   - Instagram-based interactions
   - Available: github.com/sweetalysis/inspire

### General Recommendation

47. **MovieLens Datasets**
   - Rating data for movies
   - Multiple sizes (100K to 25M)
   - Available: grouplens.org/datasets/movielens

## Implementation Resources

### Libraries & Tools

48. **Stable-Baselines3: Reliable RL Implementations**
   - PPO, DQN, A2C implementations
   - Documentation and tutorials
   - GitHub: DLR-RM/stable-baselines3

49. **Hugging Face Transformers**
   - Pre-trained NLP models
   - Easy fine-tuning
   - transformers.huggingface.co

50. **PyTorch**
   - Deep learning framework
   - Dynamic computation graphs
   - pytorch.org

### Fairness Tools

51. **AIF360: AI Fairness 360**
   - IBM's fairness toolkit
   - Metrics and mitigation algorithms
   - github.com/Trusted-AI/AIF360

52. **Fairlearn**
   - Microsoft's fairness toolkit
   - Focus on classification and regression
   - fairlearn.org

## Evaluation & Metrics

53. **Evaluation Metrics for Conversational Agents**
   - Walker et al. (1997)
   - PARADISE framework
   - Task success and user satisfaction

54. **Evaluating Recommender Systems**
   - Shani & Gunawardana (2011)
   - Comprehensive evaluation survey
   - Online vs offline metrics

55. **Beyond Accuracy: Evaluating Recommender Systems**
   - McNee et al. (2006)
   - User-centric evaluation
   - Diversity and novelty metrics

56. **Rank and Relevance in Novelty and Diversity Metrics for Recommender Systems**
   - Vargas & Castells (2011)
   - Formal treatment of novelty/diversity
   - Useful for exposure-aware novelty interpretation

## Additional Topics

### Constrained Optimization

57. **Constrained Policy Optimization**
   - Achiam et al. (2017)
   - Safety constraints in RL
   - CPO algorithm details

58. **Lagrangian Methods for Constrained Optimization**
   - Bertsekas (1996)
   - Classical optimization theory
   - Application to ML problems

### Multi-Objective Optimization

59. **A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II**
   - Deb et al. (2002)
   - Evolutionary multi-objective optimization
   - Non-dominated sorting

60. **Hypervolume Indicator**
   - Zitzler & Thiele (1999)
   - Quality metric for Pareto fronts
   - Comparison of solution sets

## Implementation to Reference Map

This project's current implementation aligns with the literature as follows:

- `src/policy_network.py` + `src/train.py`
   - PPO and GAE rollout updates follow Schulman et al. (2017, 2016).

- `src/off_policy_evaluation.py`
   - Off-policy estimators (IPS, SNIPS, DR, DM) align with Dudik et al. (2011) and Swaminathan and Joachims (2015).
   - Novelty reporting is exposure/popularity-aware and can be interpreted using Vargas and Castells (2011).

- `src/personalization_engine.py`
   - Cold-start design follows meta-learning intuition from MAML references.
   - Thompson-sampling exploration is based on contextual-bandit literature.

- `src/diversity_fairness_controller.py`
   - MMR-inspired reranking (Carbonell and Goldstein, 1998).
   - Exposure-aware fairness considerations follow Singh and Joachims (2018) and Diaz et al. (2020).

- `src/explanation_generator.py`
   - Hybrid explanation strategy is motivated by explainable recommendation survey guidance and neural template methods.

## Suggested Reading Order

### Phase 1: Foundations (Week 1-2)
1. CRS Survey (Jannach et al.)
2. ReDial Paper (Li et al.)
3. BERT Paper (Devlin et al.)
4. RL Book: Sutton & Barto (selected chapters)

### Phase 2: Core Techniques (Week 3-4)
5. PPO Paper (Schulman et al.)
6. Multi-Objective RL Survey (Van Moffaert & Nowe)
7. MMR Paper (Carbonell & Goldstein)
8. Fairness in Ranking (Singh & Joachims)

### Phase 3: Advanced Topics (Week 5-6)
9. Explainable Recommendation Survey (Zhang & Chen)
10. MAML Paper (Finn et al.)
11. Attention mechanisms (Vaswani et al.)
12. Multi-stakeholder Recommendation (Abdollahpouri et al.)

### Phase 4: Implementation (Week 7-8)
13. Transformer library documentation
14. Stable-Baselines3 documentation
15. Fairness toolkit tutorials
16. Related system implementations on GitHub

---

## Citation Format

For your thesis, use IEEE or ACM format. Example:

```
[1] T. Li et al., "Towards Knowledge-Based Recommender Dialog System," 
    in Proc. EMNLP, 2019, pp. 1803-1813.
```

## Keeping Up-to-Date

- **Conferences**: SIGIR, RecSys, NeurIPS, ICML, ACL, EMNLP
- **Arxiv**: cs.IR, cs.LG, cs.CL categories
- **Newsletters**: Papers with Code, ImportAI
- **Forums**: r/MachineLearning, Twitter ML community
