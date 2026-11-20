"""Run the continuous dimensionality study. Use --check for the protocol checks and --all for the complete scan."""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks import protocol, reference_values  # noqa: E402
from benchmarks.algorithms import GLOSSAlgorithm  # noqa: E402
from benchmarks.bench_main import (  # noqa: E402
    RUNNER_VERSION, _initial_points, continuous_landscape, run_continuous,
)

from . import dimension_config as cfgmod  # noqa: E402

RESULTS = os.path.join(ROOT, "build", "paper", "reruns", "dimensions")


def unit_paths(tag, cfg, seed):
    stem = f"{tag}_{cfgmod.unit_name(cfg, seed)}"
    return (os.path.join(RESULTS, stem + "_summary.csv"),
            os.path.join(RESULTS, stem + "_points.csv"))


def build_landscape(cfg):
    return continuous_landscape(cfg["function"], **cfg["params"],
                                noise_sigma=cfgmod.NOISE_SIGMA,
                                seed=cfgmod.TERRAIN_SEED)


def make_arm(arm, seed):
    return GLOSSAlgorithm(
        strategy_points=dict(cfgmod.RATIOS[arm]),
        ucb_kappa=cfgmod.UCB_KAPPA,
        diversity_radius=cfgmod.DIVERSITY_RADIUS,
        seed=seed,
    )


# -- preflight -----------------------------------------------------------

def preflight():
    """Validate the terrain, allocation and evaluation budget before running experiments."""
    problems = []
    if protocol.POOL_SIZE != cfgmod.POOL_N_SMALL:
        problems.append(
            f"protocol.POOL_SIZE is {protocol.POOL_SIZE}, expected "
            f"{cfgmod.POOL_N_SMALL}: the default must not move")
    if len(cfgmod.CONFIGS) != 9:
        problems.append(f"{len(cfgmod.CONFIGS)} configurations, expected 9")
    if len(cfgmod.SEEDS) != 10:
        problems.append(f"{len(cfgmod.SEEDS)} seeds, expected 10")
    if cfgmod.CENSOR_T95 != cfgmod.N_ROUNDS + 1:
        problems.append("the censored t95 is not one round past the limit")

    seen = set()
    for cfg in cfgmod.CONFIGS:
        label = cfg["label"]
        if label in seen:
            problems.append(f"duplicate configuration {label}")
        seen.add(label)
        land = build_landscape(cfg)
        key = reference_values.key_for(land["name"], **land["params"])
        ref = reference_values.get(land["name"], **land["params"])
        # The stored ends and the surface must agree, or the denominator is
        # not the one the analysis assumes.
        if abs(float(land["y_opt"]) - float(ref["y_opt"])) > 0:
            problems.append(f"{label}: y_opt disagrees with the stored value")
        if abs(float(land["y_min"]) - float(ref["y_min"])) > 0:
            problems.append(f"{label}: y_min disagrees with the stored value")
        # The 2D families carry no ndim key, so the surface's own dimension is
        # the authority; where the key exists it must agree.
        if land["ndim"] != land["params"].get("ndim", 2):
            problems.append(f"{label}: landscape ndim disagrees with params")
        expected_key = ref.get("key_dims") or list(range(land["ndim"]))
        if list(land["key_dims"]) != list(expected_key):
            problems.append(f"{label}: key dims disagree with the stored value")
        print(f"  {label:26s} {key}")
        print(f"    ndim={land['ndim']} key_dims={len(land['key_dims'])} "
              f"pool={cfg['pool_size']:,} y_opt={land['y_opt']!r} "
              f"y_min={land['y_min']!r}")

    # The three arms must differ only in the quota.
    for arm, quota in cfgmod.RATIOS.items():
        if set(quota) != {"global_best", "local_best", "unexplored",
                          "unconverged"}:
            problems.append(f"{arm}: unexpected quota keys")
        if quota["unconverged"] != 0:
            problems.append(f"{arm}: unconverged must stay 0")
        if sum(quota.values()) != cfgmod.N_POINTS:
            problems.append(f"{arm}: quota does not sum to the batch size")

    if problems:
        for line in problems:
            print("FAIL " + line)
        raise SystemExit("preflight failed; not starting the run")
    print("preflight OK")


# -- one unit ------------------------------------------------------------

def check_unit(summary, points, cfg, arm, seed, n_rounds=None):
    """Hard checks on one arm, before its file is written.

    n_rounds defaults to the frozen value; a smoke run passes its own, and is
    still held to every other rule, including the batch size per round.
    """
    n_rounds = cfgmod.N_ROUNDS if n_rounds is None else n_rounds
    label = cfg["label"]
    rounds = [r for r in summary if r["round"] >= 0]
    done = {r["round"] for r in summary}
    if done != {-1} | set(range(n_rounds)):
        raise AssertionError(f"{label}/{arm}/{seed}: rounds {sorted(done)}")
    if len(rounds) != n_rounds:
        raise AssertionError(f"{label}/{arm}/{seed}: {len(rounds)} rounds")
    if len(points) != cfgmod.N_INIT + n_rounds * cfgmod.N_POINTS:
        raise AssertionError(f"{label}/{arm}/{seed}: {len(points)} points")
    for row in rounds:
        if row["n_points"] != cfgmod.N_POINTS:
            raise AssertionError(
                f"{label}/{arm}/{seed} round {row['round']}: bought "
                f"{row['n_points']} points, the batch is {cfgmod.N_POINTS}")
        if row["n_init"] != cfgmod.N_INIT:
            raise AssertionError(f"{label}/{arm}/{seed}: n_init moved")
        if row["init_kind"] != cfgmod.INIT_KIND:
            raise AssertionError(f"{label}/{arm}/{seed}: init kind moved")
        if row["denominator"] != "true_optimum":
            raise AssertionError(f"{label}/{arm}/{seed}: denominator moved")
        if row["raw_pool_size"] != cfg["pool_size"]:
            raise AssertionError(
                f"{label}/{arm}/{seed} round {row['round']}: the raw pool held "
                f"{row['raw_pool_size']} rows, the configuration says "
                f"{cfg['pool_size']} -- the pool size was not applied")
        if row["pool_size"] != cfg["pool_size"]:
            raise AssertionError(
                f"{label}/{arm}/{seed}: pool_size is {row['pool_size']}, "
                f"the configuration says {cfg['pool_size']}")
        gap = row["y_opt"] - row["y_min"]
        want = (row["best_so_far"] - row["y_min"]) / gap * 100 if gap > 0 else 0.0
        if abs(want - row["pct_opt"]) > 1e-9:
            raise AssertionError(
                f"{label}/{arm}/{seed} round {row['round']}: pct_opt "
                f"{row['pct_opt']!r} does not recompute to {want!r}")
        if row["pct_opt"] > 100.0 * (1.0 + reference_values.PCT_OPT_RTOL):
            raise AssertionError(
                f"{label}/{arm}/{seed} round {row['round']}: pct_opt above 100")
        # Coverage can only shrink when the pool is filtered.
        if row["pool_n_above95"] > row["raw_pool_n_above95"]:
            raise AssertionError(
                f"{label}/{arm}/{seed} round {row['round']}: filtered pool "
                f"holds more qualifying points than the raw pool")
        if row["pool_max_y"] > row["raw_pool_max_y"] + 1e-12:
            raise AssertionError(
                f"{label}/{arm}/{seed} round {row['round']}: filtered pool "
                f"maximum exceeds the raw pool maximum")

    # The quota is the only independent variable. short_* records what each
    # strategy failed to deliver, and the runner tops the batch up from the
    # pool by distance. If the bookkeeping and the batch disagree, the arm's
    # realised mix is not the one its name claims, so the numbers must add up
    # before any file is written.
    for row in rounds:
        quota = cfgmod.RATIOS[arm]
        short = sum(row.get(f"short_{name}", 0) for name in quota)
        delivered_by_name = sum(quota.values()) - short
        if delivered_by_name < 0:
            raise AssertionError(
                f"{label}/{arm}/{seed} round {row['round']}: strategy "
                f"shortfalls {short} exceed the whole quota {sum(quota.values())}")
        if delivered_by_name + row["n_fill_runner"] != row["n_points"]:
            raise AssertionError(
                f"{label}/{arm}/{seed} round {row['round']}: quota deliveries "
                f"{delivered_by_name} plus fills {row['n_fill_runner']} do not "
                f"add up to the batch {row['n_points']}")
        for name in quota:
            if row.get(f"short_{name}", 0) > quota[name]:
                raise AssertionError(
                    f"{label}/{arm}/{seed} round {row['round']}: {name} "
                    f"shortfall exceeds its quota")
        # That sum is an identity, because short_* is written as quota minus
        # delivered. It catches a bookkeeping slip, not dilution. The check
        # that can actually fail is this one: an arm that delivers nothing
        # from its own strategies and fills the whole batch is not running the
        # arm its name claims, and no later correction can recover the
        # contrast.
        if row["n_points"] - row["n_fill_runner"] < 1:
            raise AssertionError(
                f"{label}/{arm}/{seed} round {row['round']}: every point came "
                f"from the fill rule, so this arm delivered none of its own "
                f"quota and the run is not the arm it claims")

    best = -np.inf
    for row in summary:
        if row["best_so_far"] < best - 1e-12:
            raise AssertionError(
                f"{label}/{arm}/{seed}: best_so_far decreased at round "
                f"{row['round']}")
        best = max(best, row["best_so_far"])


def run_unit(cfg, seed, tag, n_rounds=None):
    """Run the three arms of one (configuration, seed) unit."""
    n_rounds = cfgmod.N_ROUNDS if n_rounds is None else n_rounds
    land = build_landscape(cfg)
    # One set of starting points, shared by the three arms. _initial_points is
    # seeded by the run seed, so this is also what each arm would have drawn on
    # its own; passing it makes that a fact rather than a coincidence.
    init_X = _initial_points(land, seed, cfgmod.N_INIT, cfgmod.INIT_KIND)

    all_summary = []
    all_points = []
    for arm in cfgmod.ARM_ORDER:
        algo = make_arm(arm, seed)
        summary, points = run_continuous(
            land, cfg["label"], arm, algo, seed,
            n_rounds=n_rounds, n_points=cfgmod.N_POINTS,
            n_init=cfgmod.N_INIT, init_kind=cfgmod.INIT_KIND,
            init_X=init_X, pool_size=cfg["pool_size"], pool_coverage=True)
        check_unit(summary, points, cfg, arm, seed, n_rounds)
        # The three arms must have started from the same coordinates.
        first = [p for p in points if p["round"] == -1]
        if arm == cfgmod.ARM_ORDER[0]:
            reference_init = [(round(p["x0"], 12), round(p["x1"], 12))
                              for p in first]
        else:
            now = [(round(p["x0"], 12), round(p["x1"], 12)) for p in first]
            if now != reference_init:
                raise AssertionError(
                    f"{cfg['label']}/{seed}: {arm} started from different "
                    f"initial points")
        all_summary.extend(summary)
        all_points.extend(points)

    summary_path, points_path = unit_paths(tag, cfg, seed)
    os.makedirs(RESULTS, exist_ok=True)
    summary_df = pd.DataFrame(all_summary)
    points_df = pd.DataFrame(all_points)
    summary_df.to_csv(summary_path, index=False)
    points_df.to_csv(points_path, index=False)
    return summary_df, points_df


def write_manifest(tag):
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              cwd=ROOT).stdout.strip()
    except Exception:
        head = "unknown"
    manifest = {
        "tag": tag,
        "formal": tag == cfgmod.TAG_FORMAL,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_head": head,
        "protocol_version": protocol.VERSION,
        # Recorded so a result set carries the thresholds it was judged by.
        "censor_limit": cfgmod.CENSOR_LIMIT,
        "fill_limit": cfgmod.FILL_LIMIT,
        "runner_version": RUNNER_VERSION,
        "pool_size_default": protocol.POOL_SIZE,
        "seeds": cfgmod.SEEDS,
        "n_rounds": cfgmod.N_ROUNDS,
        "n_points": cfgmod.N_POINTS,
        "n_init": cfgmod.N_INIT,
        "init_kind": cfgmod.INIT_KIND,
        "ucb_kappa": cfgmod.UCB_KAPPA,
        "diversity_radius": cfgmod.DIVERSITY_RADIUS,
        "ratios": cfgmod.RATIOS,
        "configs": cfgmod.CONFIGS,
        "terrain_seed": cfgmod.TERRAIN_SEED,
        "noise_sigma": cfgmod.NOISE_SIGMA,
    }
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{tag}_manifest.json")
    # Written once. Every unit process starts with the same arguments, so
    # rewriting it would be a race between concurrent units, and a resumed
    # sweep would overwrite the original creation time and code revision with
    # the values of a later run.
    if os.path.exists(path):
        with open(path) as handle:
            existing = json.load(handle)
        differing = sorted(k for k in set(existing) | set(manifest)
                           if k not in ("created", "git_head")
                           and existing.get(k) != manifest.get(k))
        if differing:
            raise SystemExit(
                f"the manifest on file disagrees with this run on {differing}; "
                f"refusing to mix two designs under one tag")
        return path
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return path


def missing_units(tag, configs, seeds):
    todo = []
    for cfg in configs:
        for seed in seeds:
            summary_path, points_path = unit_paths(tag, cfg, seed)
            if not (os.path.exists(summary_path) and os.path.exists(points_path)):
                todo.append((cfg, seed))
    return todo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="run the preflight and exit")
    parser.add_argument("--all", action="store_true",
                        help="run every missing unit for the tag")
    parser.add_argument("--config", default=None,
                        help="one configuration label (default: all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="one seed (default: all missing)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="override the round count; smoke runs only")
    parser.add_argument("--tag", default=cfgmod.TAG_FORMAL)
    args = parser.parse_args(argv)

    if args.check:
        preflight()
        return

    if args.rounds is not None and args.tag == cfgmod.TAG_FORMAL:
        raise SystemExit(
            "a round override is a smoke run; use a tag other than "
            f"{cfgmod.TAG_FORMAL!r}")

    if args.config is None:
        configs = list(cfgmod.CONFIGS)
    else:
        configs = [cfgmod.config_by_label(args.config)]
    if args.seed is None:
        seeds = list(cfgmod.SEEDS)
    else:
        if args.seed not in cfgmod.SEEDS:
            raise SystemExit(f"seed {args.seed} is not in the frozen set")
        seeds = [args.seed]

    preflight()
    write_manifest(args.tag)
    todo = missing_units(args.tag, configs, seeds)
    if not todo:
        print(f"nothing to do: every unit for tag {args.tag!r} exists")
        return
    print(f"{len(todo)} of {len(configs) * len(seeds)} units to run "
          f"for tag {args.tag!r}")

    started = time.time()
    for index, (cfg, seed) in enumerate(todo, start=1):
        begin = time.time()
        print(f"[{index}/{len(todo)}] {cfg['label']} seed={seed} "
              f"pool={cfg['pool_size']:,}", flush=True)
        run_unit(cfg, seed, args.tag, n_rounds=args.rounds)
        print(f"    done in {time.time() - begin:.1f}s "
              f"(elapsed {time.time() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
