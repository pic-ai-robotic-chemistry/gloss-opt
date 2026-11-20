# GLOSS: Global-LOcal-unexplored Sampling Strategy

GLOSS is a batch recommendation algorithm for surrogate-based optimization in
large, sparsely sampled chemical spaces. It lessens over-reliance on the
surrogate model by dividing each recommended batch across three complementary
strategies:

- **Global** ranks discrete candidates by upper confidence bound (UCB). In continuous space it selects and refines high-UCB starts, then ranks refined points by predicted mean. Its fallback also ranks by predicted mean. The primary GLOSS configuration uses a diversity radius of 0.02.
- **Local** seeks local optima of the surrogate. In a discrete library, it checks the neighbourhoods of the highest-ranked candidate centres using a BallTree.
- **Unexplored** ranks candidates by their predicted objective after filtering by distance from measured points and the Global and Local points already selected for the batch. If the distance filter leaves no candidates, it ranks the remaining pool by predicted objective.

The paper benchmark requests four Global, two Local and two Unexplored recommendations in each batch of eight. The library also supports other allocations.

[Library quickstart](#quickstart) · [Rebuild the paper figures and tables](#reproducing-the-manuscript-and-supporting-information) · [Rerun the experiments](#run-the-experiments)

## Installation

The PyPI 1.0.1 release imports all three surrogate backends at startup. Install its surrogate dependencies to use the library:

```bash
python -m pip install "gloss-opt[surrogates]==1.0.1"
```

In this source checkout, the core installation supports the quickstart with scikit-learn. For paper reproduction, use the benchmark installation below. From the root of this checkout:

```bash
python -m pip install -e .
```

Requires Python 3.9 to 3.12. Core dependencies: numpy, scipy, scikit-learn.
Optional surrogate backends (torch, xgboost, lightgbm) and benchmark extras
(pandas, matplotlib, rdkit, openpyxl) install with:

```bash
pip install -e ".[full]"
```

## Quickstart

```python
import numpy as np
from gloss import GLOSS

# Discrete candidate pool: 10,000 points in 5 dimensions
rng = np.random.default_rng(0)
candidates = rng.uniform(0.0, 1.0, size=(10_000, 5))

opt = GLOSS(
    space={"candidates": candidates},
    mode="discrete",
    direction="maximize",
    ucb_kappa=2.0,
    diversity_radius=0.02,
    seed=0,
)

# Initial observations
X_obs = candidates[:8]
y_obs = np.sin(X_obs).sum(axis=1)

# One recommendation round: 4 global + 2 local + 2 unexplored points
batch = opt.recommend(
    X_train=X_obs,
    y_train=y_obs,
    X_existing=X_obs,
    strategy_points={"global_best": 4, "local_best": 2, "unexplored": 2, "unconverged": 0},
)

points = np.array([r["point"] for r in batch])   # (8, 5) points to evaluate next
```

`recommend()` fits a surrogate on the observations (or accepts a pre-fitted
one via `surrogate=`) and runs the three strategies in sequence with
deduplication. It returns a list of dicts, one per recommended point, each
carrying the `point` itself, its `predicted_value`, and the `strategy` that
proposed it. Pass measured coordinates in `X_existing` to exclude them from future recommendations. Continuous spaces are supported via `mode="continuous"` and
`space={"bounds": [(lo, hi), ...]}`.

## Reproducing the manuscript and Supporting Information

Run the following commands in the repository root. The paper environment used Python 3.9.6, NumPy 1.26.4, SciPy 1.13.1, pandas 2.3.3 and scikit-learn 1.6.1. The public lock file records how the historical RDKit installation was resolved. A new installation of the pinned RDKit distribution can report a different module version from the original machine.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c requirements-lock.txt -e ".[benchmarks]" pytest
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

python -m reproduction.rebuild
python -m reproduction.verify
```

Results are written to `build/paper/figures/` and `build/paper/tables/`. Figure 1 is the supplied method diagram. Figures 2–4 and S1–S5 are rendered from experimental records and the cached dataset projections. `--only tables` and `--only figures` run each part separately. Numerical verification compares entries with the displayed precision of the SI and the paired bootstrap results with their archived analysis. Rendering can vary with the available fonts.

The main paired bootstrap analysis writes two files in `build/paper/tables/`. In `main_contrasts.csv`, zero-valued bootstrap means count in both tails of the two-sided test. Identical paired results give p = 1. The companion `main_contrasts_archived_method.csv` reproduces the strict-tail calculation used in the manuscript. On the recorded Arrhenius–qPO comparison, the raw p value is 0.04016 with ties included and 0.03802 under the archived convention (before CSV rounding). The confidence intervals and all 24 pooled Holm decisions agree.

### Run the experiments

The primary random-forest (RF) study compares five GLOSS configurations and six baselines on QM9-gap, Buchwald–Hartwig and continuous Arrhenius-2D. All eleven configurations use the same twenty seeds, eight starting observations, and twenty recommendation rounds with eight requested experiments per round. The six baselines are BO-EI, BO-UCB, qPO, parallel Thompson sampling, the surrogate-assisted genetic algorithm and random sampling. The two additional allocation ratios are descriptive. The 24 primary statistical comparisons use the nine-arm subset.

```bash
# One seed, one recommendation round, all eleven methods and all three datasets
python -m benchmarks.bench_main --study paper --seeds 1 --rounds 1 --out-suffix _check
python -m reproduction.verify --rerun-prefix build/paper/reruns/bench_paper_check

# Full RF experiment (660 runs)
python -m benchmarks.bench_main --study paper
python -m reproduction.rebuild --only tables --main-summary build/paper/reruns/bench_paper_summary.csv
python -m reproduction.verify --full --rerun-prefix build/paper/reruns/bench_paper
```

`--study paper` saves round summaries and point-level records to `build/paper/reruns/`. The verifier checks candidate IDs for discrete spaces, coordinates on the continuous surface, objective values and strategy labels. The first recommendation has raw round index 0. For the tables, initialization is round 0 and a run that misses the target after 20 recommendations receives 21. Some discrete QM9 batches contain fewer than eight points. Verification checks the actual point IDs and count of each archived batch. Timing varies by machine. `--study main9` runs the nine-arm primary comparison with twenty seeds. `--study main` and `--study all` retain the earlier five-arm study.

### Figure and table index

Each generated file is listed relative to `build/paper/`. Tables S6 and S10 give protocol settings and are included as source tables.

| Paper item | Input or experiment | Output or source |
|---|---|---|
| Fig. 1 | Supplied method diagram | `figures/fig1_architecture.jpg` |
| Fig. 2, Tables S7–S9, S14, main paired tests | RF round records, 3 datasets × 11 methods × 20 seeds | `figures/fig2_main.jpg`, `tables/S7_*` to `S9_*`, `S14_main.csv`, `main_contrasts.csv` |
| Figs. 3 and S5, Table S15 | QM9, 20 pool sizes × 7 methods × 10 seeds | `figures/fig3_scaling_bar.jpg`, `si_fig_scaling_convergence.jpg`, `tables/S15_QM9_scaling.csv` |
| Fig. 4 | RF trajectories on Arrhenius, representative seed 127 selected from 20 seeds | `figures/fig4_trajectory.jpg` |
| Figs. S1–S3, Table S1 | GP comparison, scaling and seed-127 trajectories | `figures/si_fig_gp_*.jpg`, `tables/S1_GP.csv` |
| Fig. S4, Table S2 | Local top-K ablation, five seeds except full scan (seed 47 only) | `figures/si_fig_local_topk.jpg`, `tables/bench_local_topk_table.csv` |
| Tables S3 and S4 | QM9 allocation ratios and finite-pool Arrhenius complexity | `tables/S3_ratio_scaling.csv`, `S4_complexity.csv` |
| Tables S5 and S12 | Four shared initialization designs, 20 seeds per cell | `tables/S5_initialization_contrasts.csv`, `S12_initialization.csv` |
| Table S11 | Paired local-penalisation runs and per-round occupancy diagnostics | `tables/S11_local_penalisation.csv` |
| Table S13 | Batch distance and spread for the seed-47, ten-round diagnostic | `tables/S13_batch_distances.csv` |
| Tables S6 and S10 | Protocol and algorithm settings | Source tables in `reproduction/reference/S6_seeds.tex` and `S10_hyperparameters.tex` |

### Supplementary experiments

Run each command from the repository root.

```bash
python -m reproduction.studies.scaling
python -m reproduction.studies.scaling --study ratios
python -m reproduction.studies.gp --column both
python -m reproduction.studies.gp_refinement
python -m reproduction.studies.initialization
python -m reproduction.studies.local_penalty
python -m reproduction.studies.dimensions --all
python -m reproduction.studies.radius
python -m reproduction.studies.tournament
python -m benchmarks.bench_local_topk
python calibrate_reference_values.py
```

### Historical comparator in Table S4

Table S4 uses UCB tournaments with a within-batch distance penalty. The original CSV labels the method `GA`, and the table generator maps that label to `UCB tournament`. The main benchmark and the current `--study complexity` command use the surrogate-assisted genetic algorithm with population selection, crossover and mutation.

To rerun the historical column, use `python -m reproduction.studies.tournament`. Compare its output in `build/paper/reruns/S4_tournament.csv` with the archived column in [the complexity study records](benchmarks/results/bench_complexity_summary.csv). Mean rounds to target over the five seeds are shown below.

| Source | C1 | C2 | C3 | C4 | C5 |
|---|---:|---:|---:|---:|---:|
| Archived records used in Table S4 | 6.4 | 5.4 | 15.0 | 9.8 | 17.8 |
| Local rerun (Python 3.9.6, arm64) | 5.6 | 4.8 | 17.2 | 9.8 | 19.4 |

The `bench_main` study commands save new results in `build/paper/reruns/`. In particular, `--study complexity` and `--study all` write the current GA results to `bench_complexity_summary.csv` in that directory.

The cause of the differences at C1, C2, C3 and C5 is unresolved. The table rebuild uses the archived records.

### Datasets and archived results

The revised QM9 scaling study uses `load_qm9(n_samples=size, target="gap")`, sampled with random state 42 at each size. QM9 and the Buchwald–Hartwig workbook are bundled in `benchmarks/data`. The stratified QM9 pools in that directory belong to a separate legacy study. The compressed study records are in `reproduction/records`. The default figure command `python -m benchmarks.plot_benchmark` renders the manuscript figures. Pass `--legacy` to draw the older plots.

## Tests

```bash
python -m pytest tests/ -q
```

Install `.[full]` to include tests of the optional neural-network backends.

## License

MIT. See `LICENSE`.
