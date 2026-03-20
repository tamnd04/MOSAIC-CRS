# MO-CRS: Multi-Objective Conversational Recommender System

MO-CRS is a thesis-oriented conversational recommender prototype that combines four objectives in one RL training pipeline:

- diversity
- fairness
- transparency (explanations)
- personalization

The project is built around a modular architecture in `src/` and supports supervised pretraining, behavioral cloning warm-start, PPO-based RL fine-tuning, and offline evaluation with OPE metrics.

## Core Modules

- `src/dialogue_state_tracker.py`: dialogue state and intent/slot signals
- `src/personalization_engine.py`: user-profile representation and preference modeling
- `src/policy_network.py`: multi-objective policy and PPO agent
- `src/diversity_fairness_controller.py`: reranking with diversity/fairness adjustments
- `src/explanation_generator.py`: explanation generation
- `src/mocrs.py`: end-to-end integration of all components

## Repository Layout

```text
.
├── config.yaml
├── config_thesis.yaml
├── demo.py
├── REFERENCES.md
├── requirements.txt
├── src/
│   ├── train.py
│   ├── evaluation.py
│   ├── run_experiments.py
│   └── ...
├── data/
└── checkpoints/
```

## Environment Setup

Windows (PowerShell):

```powershell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If dependencies are already installed in your existing `venv`, you can skip installation.

## Training Workflows

Run from `src/` so relative paths match the provided configs.

### 1. Supervised pretraining only

```bash
cd src
python train.py --mode pretrain --config ../config.yaml
```

### 2. RL fine-tuning only

```bash
cd src
python train.py --mode rl --config ../config.yaml
```

### 3. Full pipeline (pretrain + RL)

```bash
cd src
python train.py --mode both --config ../config.yaml
```

### 4. Thesis-grade profile (recommended for final experiments)

```bash
cd src
python train.py --mode both --config ../config_thesis.yaml --refresh_splits
```

The thesis profile increases training budget, strengthens validation split policy, and enables periodic OPE-driven model selection.

## Checkpoints

Checkpoints are saved to `checkpoints/` (configured by `logging.save_dir`).

Typical files:

- `best_model.pt` (best supervised checkpoint)
- `rl_checkpoint_ep*.pt` (periodic RL checkpoints)
- `best_rl_model.pt` (best RL checkpoint by DR score during evaluation, if produced)

## Offline Evaluation (OPE)

At the end of RL training, `train.py` prints offline metrics from `src/evaluation.py`.

Reported metrics include:

- `ips`
- `snips`
- `dr`
- `dm`
- `behavior_recommend_rate`
- `num_samples`

Note: this is still an offline/proxy evaluation setup, so interpret with care and always report assumptions.

## Baselines and Ablations

Use the experiment runner:

```bash
cd src
python run_experiments.py --config ../config_thesis.yaml --episodes 2000 --seeds 42 43 44 --output ../logs/ablation_results_thesis.json
```

This runs seeded variants and writes aggregate results for reproducible comparison.

## Demo

Interactive demo:

```bash
python demo.py
```

The demo initializes the model from config and runs inference interactively. If you want demo outputs from a specific trained checkpoint, load that checkpoint before demo inference (or extend `demo.py` with a checkpoint argument).

## References

See `REFERENCES.md` for the paper list behind PPO/GAE, multi-objective RL, fairness/diversity reranking, and off-policy evaluation choices.
