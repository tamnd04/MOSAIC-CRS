# MOSAIC-CRS: Multi-Objective System for Adaptive, Inclusive Conversational Recommendation

**MOSAIC-CRS** is a thesis-oriented conversational recommender system for movie recommendation.

The name **MOSAIC-CRS** stands for:

> **M**ulti-**O**bjective **S**ystem for **A**daptive, **I**nclusive **C**onversational Recommendation

The name reflects the goal of the project: combining several complementary objectives into one conversational recommendation framework. Like a mosaic, the system brings together multiple components and evaluation goals rather than optimizing only one ranking metric.

MOSAIC-CRS focuses on four main objectives:

- **recommendation accuracy**: recommending relevant movies
- **conversational success**: reaching successful recommendations through dialogue
- **diversity**: avoiding repetitive or overly narrow recommendation lists
- **fairness**: reducing popularity bias and improving tail-item exposure

This repository snapshot is set up to run end-to-end experiments on **ReDial** and **INSPIRED**, including:

- dataset conversion (raw -> MOSAIC-CRS JSON + item catalog)
- supervised pretraining + PPO-based RL fine-tuning
- test-only evaluation with a unified metric suite (ranking/accuracy + conversation + diversity + fairness, with auxiliary transparency fields)
- reproducible ablations (same metric suite for every variant/seed)

## Repository layout

```text
.
├── config_redial.yaml
├── config_inspired.yaml
├── requirements.txt
├── demo.py
├── plot_training_eval.py
├── data/
│   ├── convert_redial.py
│   ├── convert_inspired.py
│   ├── ReDial/
│   └── INSPIRED/
├── src/
│   ├── train.py
│   ├── run_experiments.py
│   ├── test_evaluation_suite.py
│   └── ...
├── checkpoints/
└── logs/
```

## Thesis focus

The main thesis focus is not only offline recommendation ranking. MOSAIC-CRS is evaluated as a multi-objective conversational recommender system using:

1. **Recommendation results**: Recall, MRR, and NDCG.
2. **Conversation results**: success rate, success within top-K/turn constraints, average turns, and linguistic diversity.
3. **Diversity results**: intra-list diversity, genre/category coverage, and calibration error.
4. **Fairness results**: average popularity, Gini coefficient, KL divergence, tail/head exposure difference, and entropy.

Explanation generation exists as a supporting component, but it is not treated as the primary thesis contribution.

## Setup

Python 3.10+ is recommended.

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Notes:

- `requirements.txt` is configured for CUDA PyTorch wheels by default. If you are on CPU-only or a different CUDA version, install a compatible PyTorch build first, then install the remaining requirements.

## Data preparation

If `data/<DATASET>/{train,val,test}_data.json` and `data/<DATASET>/item_catalog.json` already exist, you can skip conversion.

### ReDial

Place the following files under `data/ReDial/`:

- `train_data.jsonl`
- `test_data.jsonl`
- `movies_with_mentions_with_genre_category_filled.csv`
  - must contain movie IDs and enriched metadata (genre/category + mention counts)

Convert/rebuild artifacts:

```powershell
python data/convert_redial.py --dataset_dir data/ReDial
```

This generates:

- `data/ReDial/train_data.json`
- `data/ReDial/val_data.json`
- `data/ReDial/test_data.json`
- `data/ReDial/train_data_full.json`
- `data/ReDial/test_data_full.json`
- `data/ReDial/item_catalog.json`

### INSPIRED

Place the raw INSPIRED TSV files under `data/INSPIRED/raw/` with this structure:

```text
data/INSPIRED/raw/
├── dialog_data/
│   ├── train.tsv
│   ├── dev.tsv
│   └── test.tsv
├── survey_data/
│   ├── list_of_dialog_ids_with_movie_id_all.tsv
│   └── seeker_demographic.tsv
└── movie_database.tsv
```

Convert/rebuild artifacts:

```powershell
python data/convert_inspired.py --raw_root data/INSPIRED/raw --out_root data/INSPIRED
```

This generates:

- `data/INSPIRED/train_data.json`
- `data/INSPIRED/val_data.json`
- `data/INSPIRED/test_data.json`
- `data/INSPIRED/train_data_full.json`
- `data/INSPIRED/item_catalog.json`

## Training

All commands below can be run from the repository root.

### ReDial: full pipeline (pretrain + RL)

```powershell
python src/train.py --mode both --config config_redial.yaml --dataset ReDial --eval_output logs/ReDial_train_eval.json
```

### INSPIRED: full pipeline (pretrain + RL)

```powershell
python src/train.py --mode both --config config_inspired.yaml --dataset INSPIRED --eval_output logs/INSPIRED_train_eval.json
```

### Training modes

`src/train.py` supports:

- `--mode pretrain`: supervised pretraining only (saves `best_model.pt`)
- `--mode rl`: RL fine-tuning only (recommended to pass `--checkpoint best_model.pt`)
- `--mode both`: pretrain then RL
- `--mode test`: test-only evaluation (requires `--checkpoint`)

Examples:

```powershell
# RL-only warm start from the supervised checkpoint
python src/train.py --mode rl --config config_redial.yaml --dataset ReDial --checkpoint best_model.pt --eval_output logs/ReDial_train_eval.json

# W&B logging (optional)
python src/train.py --mode both --config config_redial.yaml --dataset ReDial --wandb
```

## Test-only evaluation (full metric suite)

`--mode test` runs OPE on the test split and also runs the full test evaluation suite implemented in `src/test_evaluation_suite.py`.

ReDial:

```powershell
python src/train.py --mode test --config config_redial.yaml --dataset ReDial --checkpoint best_rl_model.pt --eval_output logs/ReDial_best_rl_test_eval.json
```

INSPIRED:

```powershell
python src/train.py --mode test --config config_inspired.yaml --dataset INSPIRED --checkpoint best_rl_model.pt --eval_output logs/INSPIRED_best_rl_test_eval.json
```

Checkpoint selection notes:

- `best_model.pt` is the best supervised checkpoint.
- `best_rl_model.pt` is the best RL checkpoint (saved during RL).
- Test mode loads **exactly what you pass in `--checkpoint`**.

### Metric groups

The test evaluation JSON includes:

- Recommendation quality: Recall@{10,50}, MRR@{10,50}, NDCG@{10,50}
- Conversation quality: Dist-2/3, BLEU-2/3, SR@{5,10,20}, AT
- Diversity: ILD@{5,10,20}, genre/category coverage, calibration error
- Fairness (exposure distribution): A@{5,10,20}, Gini (G), KL-to-uniform (L), tail-head difference (D), normalized entropy
- Auxiliary transparency proxies: groundedness/factual consistency, hallucination proxy, persuasiveness, transparency, trust, usefulness

## Ablations

Use the stable runner in `src/run_experiments.py`. It:

- starts every variant/seed from the same supervised checkpoint (`best_model.pt` by default)
- fine-tunes RL per variant/seed
- evaluates the **best RL checkpoint** found during that run
- writes a single JSON containing per-seed results + aggregates

ReDial example:

```powershell
python src/run_experiments.py --config config_redial.yaml --dataset ReDial --checkpoint best_model.pt --episodes 1500 --eval_episodes 300 --seeds 42 43 44 --output logs/ReDial_ablation_results.json
```

INSPIRED example:

```powershell
python src/run_experiments.py --config config_inspired.yaml --dataset INSPIRED --checkpoint best_model.pt --episodes 1500 --eval_episodes 300 --seeds 42 43 44 --output logs/INSPIRED_ablation_results.json
```

Variants included:

- `full_model`
- `ablation_no_diversity`
- `ablation_no_fairness`
- `ablation_accuracy_only`
- `baseline_no_bc_warmstart`

To run only a subset:

```powershell
python src/run_experiments.py --config config_redial.yaml --dataset ReDial --episodes 1500 --eval_episodes 300 --seeds 42 43 44 --variants full_model ablation_no_fairness --output logs/ReDial_ablation_subset.json
```

## Plotting

Generate a compact dashboard + individual figures from a checkpoint (train_stats) and an evaluation JSON:

```powershell
python plot_training_eval.py --config config_redial.yaml --output_dir logs/thesis_plots/ReDial
python plot_training_eval.py --config config_inspired.yaml --output_dir logs/thesis_plots/INSPIRED
```

Optional overrides:

- `--checkpoint checkpoints/<DATASET>/best_rl_model.pt`
- `--eval_json logs/<DATASET>_best_rl_test_eval.json`

## Checkpoints and outputs

Checkpoints are saved under:

- `checkpoints/ReDial/`
- `checkpoints/INSPIRED/`

Common files:

- `best_model.pt` (best supervised model)
- `best_rl_model.pt` (best RL model)
- `checkpoint_epoch_*.pt` (periodic supervised checkpoints)
- `rl_checkpoint_ep*.pt` (periodic RL checkpoints)

Evaluation JSON reports are written where you point `--eval_output` (typically under `logs/`).

## References

See `REFERENCES.md` for the paper list behind PPO/GAE, multi-objective RL, fairness/diversity reranking, and off-policy evaluation choices.
