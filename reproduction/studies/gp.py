"""Run the nine-arm GP benchmark and the five-arm GP scaling study."""
import argparse
import glob
import json
import os
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.bench_main import (  # noqa: E402
    N_ROUNDS, continuous_landscape, run_continuous, run_discrete, make_arms)
from benchmarks.datasets import load_qm9, load_buchwald_hartwig  # noqa: E402
from benchmarks.gp_surrogate import GPSurrogate  # noqa: E402

OUT = os.path.join(ROOT, "build", "paper", "reruns", "gp")
UNITS = os.path.join(OUT, "units")

#: The nine arms that do not need joint posterior samples.
MAIN_ARMS = (
    "GLOSS(2:3:3)", "GLOSS(4:2:2)", "GLOSS(6:1:1)",
    "GLOSS(8:0:0)+div", "GLOSS(8:0:0)",
    "UCB-BO", "BO(EI)", "GA", "Random",
)

#: Figure 3 plots five of them; the two other ratio arms are a Figure 2 story.
SCALING_ARMS = ("GLOSS(4:2:2)", "UCB-BO", "BO(EI)", "GA", "Random")

#: The twenty pool sizes and the ten seeds of the published scaling study.
SCALING_SIZES = tuple(range(5000, 100001, 5000))
SEEDS_MAIN = (47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
              97, 101, 103, 107, 109, 113, 127, 131, 137, 139)
SEEDS_SCALING = SEEDS_MAIN[:10]

MAIN_DATASETS = ("qm9", "bh", "arr")

# Refinement displacement measured by the probe module.
EVIDENCE = ("reproduction.studies.gp_refinement, largest displacement 1.183366 "
            "(random forest 0.000000)")

#: sklearn warns once per dimension when a length scale sits on its bound.
#: That is a property of the frozen kernel, not a fault of the run.
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:                                    # pragma: no cover
    pass


def _safe(arm):
    return arm.replace("(", "_").replace(")", "").replace(":", "").replace("+", "p")


def unit_path(column, ds, arm, seed):
    return os.path.join(UNITS, f"{column}_{ds}_{_safe(arm)}_s{seed}_summary.csv")


def _qm9(n):
    ds = load_qm9(n_samples=n, target="gap")
    ds["name"] = f"QM9-gap-{n // 1000}k"
    return ds


#: Building a dataset is expensive next to a Gaussian-process fit: QM9 at
#: 100,000 rows takes minutes to load, a unit takes one.  Workers therefore keep
#: what they built.  The scaling column walks twenty pool sizes, so it holds the
#: two most recent and lets the rest go -- 100,000 rows float64 is 16 MB a set,
#: and twenty sets would be 320 MB in each of several processes.
_CACHE = {}
_CACHE_KEEP = 2


def _cached(key, build):
    if key in _CACHE:
        return _CACHE[key]
    value = build()
    _CACHE[key] = value
    while len(_CACHE) > (_CACHE_KEEP if key[0] == "scaling" else 8):
        _CACHE.pop(next(iter(_CACHE)))
    return value


def dataset_payload(kind, ds):
    """Everything one unit needs, built the way the published studies build it."""
    return _cached((kind, ds), lambda: _build_payload(kind, ds))


def _build_payload(kind, ds):
    if kind == "main":
        if ds == "qm9":
            return {"kind": "discrete", "ds": _qm9(100000)}
        if ds == "bh":
            return {"kind": "discrete", "ds": load_buchwald_hartwig()}
        if ds == "arr":
            land = continuous_landscape("arrhenius", n_peaks=3, peak_width=0.08,
                                        noise_sigma=0.0, seed=42)
            return {"kind": "continuous", "land": land,
                    "name": "Arrhenius-2D-cont"}
        raise KeyError(ds)
    if kind == "scaling":
        return {"kind": "discrete", "ds": _qm9(int(ds))}
    raise KeyError(kind)


def run_unit(column, ds, arm, seed):
    payload = dataset_payload(column, ds)
    algo = make_arms(seed, [arm])[arm]
    factory = lambda s: GPSurrogate()                 # noqa: E731
    t0 = time.time()
    if payload["kind"] == "discrete":
        summary, points = run_discrete(
            payload["ds"], arm, algo, seed, surrogate_factory=factory,
            surrogate_tag="gp")
    else:
        summary, points = run_continuous(
            payload["land"], payload["name"], arm, algo, seed,
            surrogate_factory=factory, surrogate_tag="gp",
            refinement_evidence=EVIDENCE)
    return summary, points, time.time() - t0


def _one(job):
    column, ds, arm, seed = job
    path = unit_path(column, ds, arm, seed)
    if os.path.exists(path):
        return f"skip {column} {ds} {arm} s{seed}"
    summary, points, secs = run_unit(column, ds, arm, seed)
    os.makedirs(UNITS, exist_ok=True)
    pd.DataFrame(points).to_csv(
        path.replace("_summary.csv", "_summary_points.csv"), index=False)
    pd.DataFrame(summary).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    best = max(float(r["pct_opt"]) for r in summary)
    return f"ok   {column} {ds} {arm:16s} s{seed}: pct={best:5.1f} ({secs:.0f}s)"


def jobs(column, datasets=None, arms=None, seeds=None):
    out = []
    if column == "main":
        for ds in (datasets or MAIN_DATASETS):
            for arm in (arms or MAIN_ARMS):
                for seed in (seeds or SEEDS_MAIN):
                    out.append(("main", ds, arm, seed))
    else:
        for size in (datasets or SCALING_SIZES):
            for arm in (arms or SCALING_ARMS):
                for seed in (seeds or SEEDS_SCALING):
                    out.append(("scaling", str(size), arm, seed))
    return out


def run(column, workers, datasets=None, arms=None, seeds=None):
    todo = jobs(column, datasets, arms, seeds)
    done = [j for j in todo if os.path.exists(unit_path(*j))]
    left = [j for j in todo if not os.path.exists(unit_path(*j))]
    print(f"{len(todo)} units, {len(done)} already on disk, {len(left)} to run",
          flush=True)
    if not left:
        return
    t0 = time.time()
    if workers <= 1:
        for i, job in enumerate(left, 1):
            line = _one(job)
            print(f"[{i}/{len(left)}] {line}", flush=True)
    else:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for i, line in enumerate(pool.imap_unordered(_one, left), 1):
                print(f"[{i}/{len(left)}] {line}  ({time.time() - t0:.0f}s)",
                      flush=True)


#: The scaling column carries the pool size in the dataset name and the main
#: column must not: the published main rows say plain "QM9-gap", and a label
#: mismatch silently produced an empty merge once.  Normalise on the way in.
_MAIN_LABEL = {"QM9-gap-100k": "QM9-gap"}


def _merge_column(column):
    files = sorted(glob.glob(os.path.join(UNITS, f"{column}_*_summary.csv")))
    if not files:
        raise SystemExit(f"no units for column {column!r}")
    summary = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    points = pd.concat(
        [pd.read_csv(f.replace("_summary.csv", "_summary_points.csv"))
         for f in files], ignore_index=True)
    if column == "main":
        for frame in (summary, points):
            frame["dataset"] = frame["dataset"].replace(_MAIN_LABEL)
    summary["surrogate"] = "gp"
    points["surrogate"] = "gp"
    if column == "main":
        expect = len(MAIN_ARMS) * len(SEEDS_MAIN)
        for ds, suffix in (("qm9", "qm9"), ("bh", "bh"), ("arr", "arr")):
            name = {"qm9": "QM9-gap", "bh": "Buchwald-Hartwig",
                    "arr": "Arrhenius-2D-cont"}[ds]
            s = summary[summary["dataset"] == name]
            p = points[points["dataset"] == name]
            cells = s.groupby(["algorithm", "seed"]).ngroups
            if cells != expect:
                raise SystemExit(f"main {suffix}: {cells} cells, expected {expect}")
            labels = set(s["dataset"].unique())
            if len(labels) != 1:
                raise SystemExit(f"main {suffix}: dataset labels {sorted(labels)}")
            s.to_csv(os.path.join(OUT, f"gp_main_{suffix}_summary.csv"),
                     index=False)
            p.to_csv(os.path.join(OUT, f"gp_main_{suffix}_points.csv"),
                     index=False)
            print(f"gp_main_{suffix}: {len(s)} summary rows, "
                  f"{s['algorithm'].nunique()} arms, {cells} cells, "
                  f"{len(p)} point rows")
    else:
        expect = len(SCALING_ARMS) * len(SEEDS_SCALING) * len(SCALING_SIZES)
        cells = summary.groupby(["algorithm", "seed", "dataset"]).ngroups
        if cells != expect:
            raise SystemExit(f"scaling: {cells} cells, expected {expect}")
        summary.to_csv(os.path.join(OUT, "gp_scaling_summary.csv"), index=False)
        points.to_csv(os.path.join(OUT, "gp_scaling_points.csv"), index=False)
        print(f"gp_scaling: {len(summary)} summary rows, "
              f"{summary['algorithm'].nunique()} arms, {cells} cells, "
              f"{len(points)} point rows")
    return summary


def environment():
    import sklearn, scipy
    lines = [
        "GP figure columns, environment",
        f"python {platform.python_version()}",
        f"numpy {np.__version__}",
        f"pandas {pd.__version__}",
        f"scikit-learn {sklearn.__version__}",
        f"scipy {scipy.__version__}",
        f"platform {platform.platform()}",
        f"machine {platform.machine()}",
        f"arms_main {','.join(MAIN_ARMS)}",
        f"arms_scaling {','.join(SCALING_ARMS)}",
        f"seeds_main {','.join(str(s) for s in SEEDS_MAIN)}",
        f"seeds_scaling {','.join(str(s) for s in SEEDS_SCALING)}",
        f"sizes {','.join(str(n) for n in SCALING_SIZES)}",
        f"rounds {N_ROUNDS}",
        f"protocol {__import__('benchmarks.protocol', fromlist=['VERSION']).VERSION}",
        "surrogate benchmarks.gp_surrogate.GPSurrogate (frozen kernel)",
        "note qPO and Thompson cannot run under this surrogate (no joint samples)",
    ]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "gp_figs_environment.txt"), "w") as fh:
        fh.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--column", choices=("main", "scaling", "both", "merge"),
                    default="merge")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.column == "merge":
        environment()
        for column in ("main", "scaling"):
            _merge_column(column)
        return
    columns = ("main", "scaling") if args.column == "both" else (args.column,)
    for column in columns:
        run(column, args.workers, datasets=args.datasets,
            arms=args.arms, seeds=args.seeds)
    environment()
    if args.datasets or args.arms or args.seeds:
        return
    for column in columns:
        _merge_column(column)


if __name__ == "__main__":
    main()
