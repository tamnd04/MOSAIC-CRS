# Thesis Upgrade Plan and Research Grounding

This project can be positioned as a legitimate final-year thesis by combining methodological rigor, stronger evaluation, and reproducibility.

## What Was Upgraded in Code

1. Offline evaluation was upgraded in `src/evaluation.py`.
- Uses `accepted_items` and mention overlap for reward proxy.
- Uses empirical behavior action probabilities instead of fixed 0.25.
- Reports stabilized OPE metrics: `ips`, `snips`, `dr`, `dm`.
- Uses clipped importance weights (`ips_clip_c`) for variance control.

2. RL now performs periodic held-out validation in `src/train.py`.
- Runs OPE every `training.rl.eval_interval_episodes`.
- Saves `best_rl_model.pt` based on best DR score.

3. Thesis configuration profile added in `config_thesis.yaml`.
- Larger candidate pool.
- More PPO updates and minibatches.
- Longer BC warm-start.
- Stronger train/val/test split policy.

## Why Your Previous Training Was Lightweight

1. Environment is a simulator, not an expensive online system.
2. Rollout horizon is short (20 turns).
3. Candidate pool and update budget were moderate.
4. OPE used lightweight proxies, not full logged propensities.

A 1-2 hour training run is expected for this prototype setup.

## Thesis-Grade Protocol

1. Data protocol
- Regenerate stronger splits with stratification (already supported).
- Report split sizes and demographic composition.

2. Training protocol
- Use `config_thesis.yaml`.
- Train with 3 seeds (`42, 43, 44`) and report mean/std.
- Keep periodic checkpoints and select best via DR.

3. Evaluation protocol
- Report OPE (`IPS`, `SNIPS`, `DR`, `DM`) on validation and test.
- Report online-simulator metrics (success, turns, diversity, fairness).
- Add ablation and baseline comparisons via `src/run_experiments.py`.

4. Reproducibility protocol
- Record exact command lines and config hash.
- Save JSON output of ablations and final selected checkpoint.

## Suggested Run Commands

```bash
cd src
python train.py --mode both --config ../config_thesis.yaml --refresh_splits
```

```bash
cd src
python run_experiments.py --config ../config_thesis.yaml --episodes 2000 --seeds 42 43 44 --output ../logs/ablation_results_thesis.json
```

## References Supporting These Choices

1. PPO for stable policy optimization:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)

2. GAE and bias-variance in advantage estimation:
- Schulman et al., "Generalized Advantage Estimation" (2016)

3. Multi-objective RL foundations:
- Roijers et al., "A Practical Guide to Multi-Objective Reinforcement Learning and Planning" (2013)
- Van Moffaert and Nowe, "Multi-Objective Reinforcement Learning: A Comprehensive Overview" (2014)

4. Diversity reranking with MMR:
- Carbonell and Goldstein, "The Use of MMR" (1998)
- Kunaver and Pozrl, "Diversity in Recommender Systems - A Survey" (2017)

5. Fairness and exposure in ranking/recommendation:
- Singh and Joachims, "Fairness-Aware Ranking in Search and Recommendation Systems" (2018)
- Diaz et al., "Fairness of Exposure in Rankings" (2020)
- Yao and Huang, "Beyond Parity" (2017)

6. Off-policy evaluation best practices:
- Dudik et al., "Doubly Robust Policy Evaluation and Learning" (2011)
- Swaminathan and Joachims, "Counterfactual Risk Minimization" (2015)

These references are already aligned with `REFERENCES.md` themes; add the last two OPE papers there for completeness.
