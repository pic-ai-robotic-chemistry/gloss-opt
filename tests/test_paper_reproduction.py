"""Regressions for the paper's rerun and verification entry points."""
from itertools import product

import numpy as np
import pandas as pd
import pytest

from benchmarks import bench_main
from reproduction import statistics, verify


def test_bootstrap_handles_identical_pairs_and_zero_resampled_means():
    rng = np.random.default_rng(42)
    assert statistics.bootstrap(np.zeros(20), rng) == (0, 0, 1, False)

    # For [0, 1], the resampled mean is zero one quarter of the time and
    # positive otherwise. Twice the smaller inclusive tail is about 0.5.
    lo, hi, p, floor = statistics.bootstrap(np.array([0., 1.]), rng)
    assert (lo, hi) == (0., 1.)
    assert 0.48 < p < 0.52
    assert not floor


def test_short_batches_match_archive_but_missing_points_and_rounds_fail(tmp_path, monkeypatch):
    archive = tmp_path / "main"
    archive.mkdir()
    monkeypatch.setattr(verify, "DATA", tmp_path)
    prefix = tmp_path / "rerun"
    summaries = []
    points = []
    for ds, arm, r in product(statistics.MAIN.values(), bench_main.PAPER_ARMS, [-1, 0]):
        key = dict(dataset=ds, algorithm=arm, seed=47, round=r)
        summaries.append(dict(key, pct_opt=80., best_so_far=2.))
        size = 6 if (ds, arm, r) == ("QM9-gap", "GLOSS(2:3:3)", 0) else 8
        for i in range(size):
            points.append(dict(key, point_idx=i, candidate_idx=i, x0=0., x1=0., y=2.,
                               strategy="init" if r == -1 else "global_best"))
    for suffix, rows in [("summary", summaries), ("points", points)]:
        df = pd.DataFrame(rows)
        df.to_csv(archive / f"paper_{suffix}.csv.gz", index=False)
        df.to_csv(f"{prefix}_{suffix}.csv", index=False)
    verify.check_rerun(prefix)
    with pytest.raises(ValueError, match="twenty seeds and rounds"):
        verify.check_rerun(prefix, full=True)
    pd.DataFrame(points[:-1]).to_csv(f"{prefix}_points.csv", index=False)
    with pytest.raises(AssertionError):
        verify.check_rerun(prefix)
    pd.DataFrame(summaries[:-1]).to_csv(f"{prefix}_summary.csv", index=False)
    with pytest.raises(AssertionError):
        verify.check_rerun(prefix)


def test_complexity_study_preserves_historical_comparator(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    archive.mkdir()
    original = archive / "bench_complexity_summary.csv"
    original.write_text("historical UCB tournament\n")
    output = tmp_path / "reruns"
    monkeypatch.setattr(bench_main, "RESULTS_DIR", str(archive))
    monkeypatch.setattr(bench_main, "RERUN_DIR", output)
    monkeypatch.setattr(bench_main, "COMPLEXITY_LEVELS", [dict(
        label="small", n_peaks=2, noise_sigma=0, peak_width=0.1, n_candidates=100)])
    frame = bench_main.study_complexity(seeds=[47], n_rounds=1)
    assert set(frame.algorithm) == {"GA", "GLOSS", "BO(EI)", "UCB-BO", "Random"}
    assert (output / original.name).is_file()
    assert original.read_text() == "historical UCB tournament\n"
