# MO-CRS: Multi-Objective Conversational Recommender System

MO-CRS is a thesis-oriented conversational recommender prototype that combines four objectives in one RL training pipeline:

- diversity
- fairness
- transparency (explanations)
- personalization

The project is built around a modular architecture in `src/` and supports supervised pretraining, behavioral cloning warm-start, PPO-based RL fine-tuning, and offline evaluation with OPE metrics.

## Current Architecture Status

The main runtime path now actively uses the advanced modules (not only test paths):

- Personalization Engine (PE): cold-start handling and Thompson-sampling exploration are wired into rollout/inference batches through config-driven switches.
- Diversity and Fairness Controller (DFC): temporal diversity penalties and exposure updates are applied during top-k reranking.
- Explanation Generator (EG): default mode is `hybrid` (template + neural); if GPT-2 is unavailable at runtime, generation safely falls back to template mode.
- Evaluation: novelty is computed from empirical exposure-based self-information, replacing fixed heuristic placeholders.

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
│   ├── off_policy_evaluation.py
│   ├── run_experiments.py
│   └── ...
├── data/
│   ├── ReDial/
│   ├── GoRecDial/
│   ├── INSPIRED/
│   ├── MovieLens_1M/
│   ├── Yelp/
│   ├── DuRecDial/
│   ├── LastFM/
│   ├── OpenDialKG/
│   └── item_catalog.json
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
python train.py --mode pretrain --config ../config.yaml --dataset ReDial
```

### 2. RL fine-tuning only

```bash
cd src
python train.py --mode rl --config ../config.yaml --dataset GoRecDial --eval_output ../logs/GoRecDial_rl_ope.json
```

### 3. Full pipeline (pretrain + RL)

```bash
cd src
python train.py --mode both --config ../config.yaml --dataset INSPIRED
```

### 4. Thesis-grade profile (recommended for final experiments)

```bash
cd src
python train.py --mode both --config ../config_thesis.yaml --dataset ReDial --eval_output ../logs/ReDial_thesis_ope.json
```

Run `--refresh_splits` only once when you intentionally want to regenerate train/val/test files.
For final reported comparisons, keep splits fixed across all runs.

### 5. Test-only evaluation (no RL, no pretraining)

```bash
cd src
python train.py --mode test --config ../config.yaml --dataset ReDial --checkpoint best_rl_model.pt --eval_output ../logs/ReDial_test_ope.json
```

`--mode test` requires `--checkpoint` and evaluates the loaded model on validation/test splits using OPE.
Use `--eval_output` in RL or test modes to save evaluation metrics as a JSON report for paper comparison.

The thesis profile increases training budget, strengthens validation split policy, and enables periodic OPE-driven model selection.

Supported dataset names (case-insensitive aliases are accepted):

- ReDial
- GoRecDial
- INSPIRED
- MovieLens_1M
- Yelp
- DuRecDial
- LastFM
- OpenDialKG

## OpenDialKG Preparation

If you downloaded OpenDialKG raw files (`data/OpenDialKG/raw/opendialkg.csv` and KG txt files),
run the converter once to generate MO-CRS-ready splits and a dataset-specific catalog:

```bash
python data/convert_opendialkg.py --dataset_dir data/OpenDialKG
```

This creates:

- `data/OpenDialKG/train_data_full.json` (immutable full source)
- `data/OpenDialKG/train_data.json`
- `data/OpenDialKG/val_data.json`
- `data/OpenDialKG/test_data.json`
- `data/OpenDialKG/item_catalog.json`

Then train/test normally by selecting `--dataset OpenDialKG`.

## Key Config Toggles

Main switches in `config.yaml` and `config_thesis.yaml`:

- `model.personalization.cold_start.enabled`
- `model.personalization.thompson_sampling.enabled`
- `model.personalization.thompson_sampling.num_samples`
- `model.explanation_generator.generation_mode` (`template`, `neural`, `hybrid`)

For thesis-style runs, keep `generation_mode: hybrid` unless you need deterministic template-only outputs.

## Checkpoints

Checkpoints are saved to dataset-specific folders under `checkpoints/` (configured by `logging.save_dir`).

Typical files:

- `checkpoints/ReDial/best_model.pt` (best supervised checkpoint)
- `checkpoints/ReDial/rl_checkpoint_ep*.pt` (periodic RL checkpoints)
- `checkpoints/ReDial/best_rl_model.pt` (best RL checkpoint by DR score during evaluation, if produced)

When loading a checkpoint name (not full path), `train.py` first looks in the selected dataset folder, then falls back to the checkpoint root for backward compatibility.

## Offline Evaluation (OPE)

At the end of RL training, `train.py` prints offline metrics from `src/off_policy_evaluation.py`.
If `--eval_output` is provided, the same OPE metrics are also saved to a JSON file.

Reported metrics include:

- `ips`
- `snips`
- `dr`
- `dm`
- `dr_ci_low`, `dr_ci_high`
- `dm_ci_low`, `dm_ci_high`
- `behavior_recommend_rate`
- `logged_reward_mean`, `logged_reward_std`
- `num_samples`

Note: this is still an offline/proxy evaluation setup, so interpret with care and always report assumptions.

## Aggregate Evaluation CSV

You can aggregate all per-dataset JSON evaluation outputs into one thesis-ready CSV table:

```bash
cd src
python aggregate_eval_reports.py --input_dir ../logs --recursive --output ../logs/thesis_eval_summary.csv
```

Each row in the CSV corresponds to one dataset/split (`validation` or `test`) from an evaluation JSON report.

## Notes for Reproducibility

- Run training commands from `src/` as shown above so relative config paths resolve correctly.
- If neural explanation loading fails because pretrained GPT-2 assets are unavailable, the system continues with template generation automatically.
- Existing checkpoints trained with older policy action dimensions are not guaranteed to be directly comparable with current configs.

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
