# Multi-Objective Conversational Recommender System (MO-CRS)

## Overview
This thesis project presents a novel Conversational Recommender System that integrates **Diversity**, **Fairness**, **Transparency**, and **Personalization** through a Reinforcement Learning framework.

## Key Features
- 🎯 **Reinforcement Learning Core**: Policy-based learning with multi-objective optimization
- 🌈 **Diversity Enhancement**: MMR-based diversification and coverage metrics
- ⚖️ **Fairness Mechanisms**: User-side and item-side fairness constraints
- 🔍 **Transparency Layer**: Natural language explanation generation
- 👤 **Deep Personalization**: User preference modeling with contextual awareness

## Model Architecture
The system consists of five integrated modules:
1. **Dialogue State Tracker (DST)** - Conversation context management
2. **Multi-Objective Policy Network (MOPN)** - RL-based decision making
3. **Diversity & Fairness Controller (DFC)** - Constraint enforcement
4. **Personalization Engine (PE)** - User modeling and preference learning
5. **Explanation Generator (EG)** - Transparency and trust building

## Papers & Methods Combined
This model synthesizes approaches from:
- Deep RL for conversational agents (DQN, PPO, Actor-Critic)
- Multi-objective optimization in recommender systems
- Fairness-aware ranking algorithms
- Attention-based user modeling
- Neural explanation generation

## Directory Structure
```
├── model_architecture.md          # Detailed architecture description
├── components/                   # Individual component specifications
│   ├── dialogue_state_tracker.md
│   ├── policy_network.md
│   ├── diversity_fairness_controller.md
│   ├── personalization_engine.md
│   └── explanation_generator.md
├── algorithms/                   # Algorithm descriptions
│   ├── rl_training.md
│   ├── multi_objective_optimization.md
│   └── fairness_constraints.md
├── implementation/               # Code structure
│   └── model_skeleton.py
└── diagrams/                    # Visual representations
    └── architecture_flow.md
```

## Getting Started
See `model_architecture.md` for the complete system design and component interactions.

## Thesis-Grade Training Profile

For a stronger final-project setup, use `config_thesis.yaml`:

```bash
cd src
python train.py --mode both --config ../config_thesis.yaml --refresh_splits
```

This profile enables:
- stronger validation split policy
- longer behavioral cloning warm-start
- heavier PPO update budget
- periodic off-policy evaluation and best-RL checkpoint selection

See `THESIS_UPGRADES.md` for protocol details and references.

## Research Contributions
1. Novel integration of fairness and diversity in conversational settings
2. Multi-objective RL framework balancing competing objectives
3. Context-aware explanation generation aligned with user preferences
4. Dynamic personalization adapting to conversation flow
