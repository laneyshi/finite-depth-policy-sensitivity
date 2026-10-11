# Finite-Depth Policy Sensitivity

Code for the small-instance experiments (Figures 1–2) accompanying
“Adapting to Changes in Agent Behavior via Finite-Depth Policy Sensitivity,”
submitted to ACC 2027.

This release includes simulation, policy training, Bellman derivative kernels,
estimator comparisons, tests, and plotting code. Datasets, pretrained weights,
and generated figures are not included. The commands below generate local
weights and results from scratch. The large-instance adaptation experiment is
outside the scope of this release.

## Experimental settings

| Component | Setting |
| --- | --- |
| Environment | 6 × 6 pursuit grid; two attackers and one defender |
| Horizon / discount | 15 steps / 0.9 |
| Defender policy | Mixture of blind and belief-aware endpoint policies; reference κ = 0.7 |
| Attacker policy | Softmax with two trainable coefficients |
| Initial belief | Uniform(0.1, 0.9), integrated stochastic-projection weights |
| Initial positions | Starting-row fraction 0.5; both attackers within distance two of the defender, subject to the environment's validity checks |
| Reward | Diversion weight 0.25; global scale 1 |
| Belief grid | Nonuniform log-odds grid on [−86.4, 86.4], with finer spacing on [−21.6, 21.6] |
| Reference / refinement grids | 1,217 / 1,825 nodes; relative H, M, S change threshold 2.5 × 10⁻⁴ |
| Defender pretraining | Two 64-unit tanh hidden layers; 2,500 supervised updates; seed 12006 |
| Attacker training | PPO, 1,000,000 steps, seed 891000; uniform 1,025-node log-odds grid during training |
| Reference refinement | Newton/BFGS polishing on the nonuniform finite model |
| Critic calibration | 100,000 fresh trajectories; seed 1819000 |
| Estimation seeds | 1851000–1851009 (10 runs) |
| Sampling budgets | 100, 300, 1,000, 3,000, 10,000, 30,000, 100,000, 300,000, 1,000,000 |
| Comparison depths | 3, 5, 7 |
| Reported errors | Relative spectral norm for H; Euclidean norm for vector M and S |
| Aggregation | Median and interquartile range across ten seeds |

The comparisons include the proposed estimator, trajectory likelihood-ratio
(LR), and an additional critic-controlled LR baseline. Sample count N denotes
initial states for the proposed estimator and complete trajectories for LR.
Exact-depth calculations refer to the finite belief-grid model.

## Installation

Use Python 3.11 and install:

```sh
python -m pip install -r requirements.txt
```

Native kernels require Linux and g++. GPU execution also requires an NVIDIA
GPU, nvcc, and a CUDA-enabled PyTorch build. The original full-grid experiments
used an RTX 3090 with 24 GB GPU memory. Full-grid CPU execution requires
substantial memory and runtime.

Run all commands from this directory. Use `run.py` for experiment stages so
that the stated environment settings are applied consistently.

## Generate weights and run the experiment

```sh
python setup_experiment.py --cuda
python run.py endpoints
python test_cpu_compact.py
python run.py ppo
export SIM_DEVICE=cuda:0
export SIM_COMPACT_BACKEND=cuda
export SIM_WORKERS=10
python run.py all
```

`endpoints` trains the size-6 defender policies and creates a size-3 alias
for the compact-kernel tests. `ppo` trains the attacker and initial critic.
`all` refines the reference, checks the belief-grid refinement, calibrates the
critic, computes conditional tables, runs ten estimation seeds, and plots.
It requires the preceding training stages and refuses to overwrite an
existing `results/b1217/reference.npz`.

For CPU execution, build with `python setup_experiment.py` (without `--cuda`)
and set `SIM_COMPACT_BACKEND=cpu` instead of `cuda`.

To resume individual stages:

```sh
python run.py reference --bins 1217
python run.py audit --bins 1825 --start results/b1217/reference.npz
python run.py prepare --bins 1217
python run.py baseline --bins 1217
python run.py plot --bins 1217
```

When resuming manually, check that the audit JSON has `passed: true` before
continuing to `prepare`. `run.py all` enforces this check automatically.

## Outputs

Generated weights and arrays are stored in `results/` and ignored by Git.
Plots are written to `results/b1217/figures/`: `error_depth`, `comparison`,
`comparison_supplementary`, and `theorem_bounds`, each as PDF and PNG.
After generating results, replot with:

```sh
python report_small.py results/b1217
```

Freshly trained weights can differ across software versions and hardware;
this code-only release does not provide the archived weights or numerical
outputs used in the paper. The full training pipeline has not been rerun
for this packaging update.

## Code map

- `run.py`: experiment stages and settings.
- `environment.py`, `belief.py`, `uniform_belief.py`: simulation and belief grids.
- `networks.py`, `train.py`: policies, endpoint pretraining, and PPO.
- `exact_model.py`, `gpu_exact_model.py`: finite-model evaluation.
- `bellman*.cpp`, `bellman*.cu`, `gpu_bellman.py`, `gpu_forward.py`: CPU/CUDA computation.
- `small_validation.py`: critic calibration and estimator sampling.
- `bounds.py`, `report_small.py`: theoretical bounds and plots.
- `test_cpu_compact.py`, `test_compact.py`: numerical consistency tests.
