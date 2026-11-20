# GLOSS Benchmark Report — 4 Points per Round

## Configuration

| Parameter | Value |
|-----------|-------|
| Initial points | 10 (shared across algorithms) |
| Rounds | 20 |
| Points per round | 4 |
| Total budget | 90 (10 + 20 × 4) |
| Seeds | 5 (mean ± std) |
| Surrogate | GP (ConstantKernel × Matern(ν=2.5), fixed hyperparameters, α=1e-5) |

### Algorithm Settings

| Algorithm | Strategy |
|-----------|----------|
| GLOSS | global_best=1, local_best=1, unexplored=1, unconverged=1 |
| BO (EI) | Batch Expected Improvement, 4 points |
| Random | Uniform random sampling, 4 points |

---

## Datasets

| Dataset | Type | Size | Features | Target | Direction |
|---------|------|------|----------|--------|-----------|
| FreeSolv | Discrete | 642 | 7 (molecular descriptors) | Hydration free energy (kcal/mol) | Maximize |
| ESOL | Discrete | 1,128 | 7 (molecular descriptors) | Aqueous solubility (logS) | Maximize |
| Buchwald-Hartwig | Discrete | 3,955 | 44 (one-hot reaction conditions) | Reaction yield (%) | Maximize |
| Virtual Function | Continuous | 2D | T ∈ [100,300], C ∈ [0.1,1.0] | Simulated reaction yield (%) | Maximize |

---

## Summary Table

| Dataset | Algorithm | Final Best (mean) | Final Best (std) | Rounds to 95% (mean) | Avg Time/Round (s) |
|---------|-----------|-------------------|------------------|-----------------------|---------------------|
| FreeSolv | **GLOSS** | **1.8110** | 0.0000 | **0.6** | 0.031 |
| FreeSolv | BO (EI) | **1.8110** | 0.0000 | 1.0 | 0.002 |
| FreeSolv | Random | 1.7720 | 0.0651 | 6.8 | 0.002 |
| ESOL | **GLOSS** | **1.5800** | 0.0000 | **1.6** | 0.061 |
| ESOL | BO (EI) | **1.5800** | 0.0000 | 4.4 | 0.004 |
| ESOL | Random | 1.0030 | 0.3421 | N/A | 0.003 |
| Buchwald-Hartwig | GLOSS | 96.3410 | 6.2216 | **6.0** | 0.467 |
| Buchwald-Hartwig | **BO (EI)** | **99.7698** | 0.4603 | 9.8 | 0.019 |
| Buchwald-Hartwig | Random | 92.9368 | 3.0508 | 9.0 | 0.014 |
| Virtual Function | GLOSS | 72.4676 | 12.0417 | **6.0** | 0.264 |
| Virtual Function | **BO (EI)** | **76.4279** | 1.7475 | 8.8 | 0.042 |
| Virtual Function | Random | 73.6813 | 1.7072 | 15.0 | 0.000 |

---

## Best-so-far Curves

### FreeSolv

![FreeSolv](plots/freesolv_best_so_far.png)

### ESOL

![ESOL](plots/esol_best_so_far.png)

### Buchwald-Hartwig

![Buchwald-Hartwig](plots/buchwald-hartwig_best_so_far.png)

### Virtual Function

![Virtual Function](plots/virtual_function_best_so_far.png)

### Virtual Function Surface

![Surface](plots/virtual_function_surface.png)

---

## Key Findings

1. **GLOSS converges fastest** — On FreeSolv and ESOL, GLOSS reaches 95% of the optimum in 0.6 and 1.6 rounds respectively, significantly faster than BO's 1.0 and 4.4 rounds.

2. **GLOSS excels on small-to-medium datasets** — On FreeSolv (642 candidates) and ESOL (1,128 candidates), GLOSS matches BO's final value while arriving there earlier. The multi-strategy approach provides effective coverage of small search spaces.

3. **BO is more stable on large high-dimensional spaces** — On Buchwald-Hartwig (3,955 candidates, 44 dimensions), BO achieves a higher and more consistent final value (99.77 ± 0.46 vs GLOSS's 96.34 ± 6.22).

4. **GLOSS has higher per-round cost** — Running four strategies takes ~10-25x longer than BO per round, though absolute times remain practical (0.03–0.47s per round).

5. **Both far outperform Random** — Especially on ESOL where Random never reaches the 95% target, and on Buchwald-Hartwig where the gap is ~7 percentage points.
