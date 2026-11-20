"""Frozen configurations for the continuous dimensionality scan."""

# The three arms under test. Only the quota differs; unconverged stays 0, as
# in every published configuration.
RATIOS = {
    "2:3:3": {"global_best": 2, "local_best": 3, "unexplored": 3,
              "unconverged": 0},
    "4:2:2": {"global_best": 4, "local_best": 2, "unexplored": 2,
              "unconverged": 0},
    "6:1:1": {"global_best": 6, "local_best": 1, "unexplored": 1,
              "unconverged": 0},
}
ARM_ORDER = ["2:3:3", "4:2:2", "6:1:1"]

# The same ten seeds are paired across all configurations and allocations.
SEEDS = [47, 53, 59, 61, 67, 71, 73, 79, 83, 89]

N_ROUNDS = 20
N_POINTS = 8
N_INIT = 8
INIT_KIND = "bottom20"

# Shared GLOSS settings. Only the ratio moves between arms.
UCB_KAPPA = 2.0
DIVERSITY_RADIUS = 0.02

# Bootstrap settings for paired dimensionality comparisons.
BOOTSTRAP_SEED = 20261025
BOOTSTRAP_N = 10_000
DELTA_T95 = 2.0        # rounds; smaller differences are not practically useful
DELTA_PCT = 2.0        # percentage points of final pct_opt
DISCORDANT_LIMIT = 4   # at or above this, no "consistent" wording
# At or above this censoring fraction, the t95 comparison for that
# (configuration, arm) is reported but cannot carry a claim; the reach rate
# carries it instead. A bounded score gives a zero difference when both arms
# sit at the cap, and zero then means "no room", not "no effect".
CENSOR_LIMIT = 0.5
# Above this share of runner-filled points in a round, an arm is no longer
# delivering the quota its name claims, and the fill rule (the same for every
# arm) dilutes the only independent variable. The confirmatory gate then
# refuses to make a claim on that configuration.
FILL_LIMIT = 0.25
# Two comparisons per "best arm" label, so each is held at 1 - 0.05/2.
BEST_ARM_LEVEL = 0.975
CENSOR_T95 = N_ROUNDS + 1  # 21: the score for "not reached in 20 rounds"

POOL_N = 125_000   # nD configurations and the matched Arrhenius run
POOL_N_SMALL = 10_000  # current main protocol, Arrhenius sensitivity only

# The frozen experiment matrix. `axis` says what a row is for; the analysis
# reads it rather than re-deriving a role from the name.
CONFIGS = [
    {"label": "RC1-2D-3C", "function": "nd_clusters",
     "params": {"ndim": 2, "n_peaks": 3, "peak_width": 0.10},
     "pool_size": POOL_N, "axis": "complexity_low"},
    {"label": "RC2-4D-5C", "function": "nd_clusters",
     "params": {"ndim": 4, "n_peaks": 5, "peak_width": 0.08},
     "pool_size": POOL_N, "axis": "complexity"},
    {"label": "RC3-6D-5C", "function": "nd_clusters",
     "params": {"ndim": 6, "n_peaks": 5, "peak_width": 0.15},
     "pool_size": POOL_N, "axis": "complexity"},
    {"label": "RC4-8D-7C", "function": "nd_clusters",
     "params": {"ndim": 8, "n_peaks": 7, "peak_width": 0.14},
     "pool_size": POOL_N, "axis": "complexity"},
    {"label": "RC5-10D-7C", "function": "nd_clusters",
     "params": {"ndim": 10, "n_peaks": 7, "peak_width": 0.20},
     "pool_size": POOL_N, "axis": "complexity_high"},
    # Same dimension and cluster count as RC3; only the peak width moves, to
    # the two ends of the ratio-complexity axis. They separate the density
    # effect from the dimension effect and take no part in the complexity
    # claim.
    {"label": "RC-D10-6D-5C", "function": "nd_clusters",
     "params": {"ndim": 6, "n_peaks": 5, "peak_width": 0.10},
     "pool_size": POOL_N, "axis": "density_control"},
    {"label": "RC-D20-6D-5C", "function": "nd_clusters",
     "params": {"ndim": 6, "n_peaks": 5, "peak_width": 0.20},
     "pool_size": POOL_N, "axis": "density_control"},
    {"label": "Arrhenius-2D-cont-p125000", "function": "arrhenius",
     "params": {"n_peaks": 3, "peak_width": 0.08},
     "pool_size": POOL_N, "axis": "catalytic_matched"},
    {"label": "Arrhenius-2D-cont-p10000", "function": "arrhenius",
     "params": {"n_peaks": 3, "peak_width": 0.08},
     "pool_size": POOL_N_SMALL, "axis": "catalytic_sensitivity"},
]

# Contrast between the extreme dimensions and the within-10D allocation contrast.
GATE1 = {"high": "RC5-10D-7C", "low": "RC1-2D-3C",
         "arm_a": "2:3:3", "arm_b": "6:1:1"}
GATE2 = {"config": "RC5-10D-7C", "arm_a": "2:3:3", "arm_b": "6:1:1"}
DENSITY_CONTROL = {"high": "RC-D20-6D-5C", "low": "RC-D10-6D-5C",
                   "arm_a": "2:3:3", "arm_b": "6:1:1"}

TAG_FORMAL = "cont1"

# The terrain seed is fixed at 42 in every configuration, matching the stored
# reference values.
TERRAIN_SEED = 42
NOISE_SIGMA = 0.0


def config_by_label(label):
    for cfg in CONFIGS:
        if cfg["label"] == label:
            return cfg
    known = ", ".join(c["label"] for c in CONFIGS)
    raise KeyError(f"no dimensionality-study configuration named {label!r}. On file: {known}")


def unit_name(cfg, seed):
    return f"{cfg['label']}-p{cfg['pool_size']}-s{seed}"
