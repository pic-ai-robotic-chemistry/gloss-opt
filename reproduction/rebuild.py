"""Rebuild the manuscript figures and SI tables from archived experiments."""
import argparse
from shutil import copyfile

from reproduction.paths import FIGURES, ROOT, TABLES


def figures():
    from benchmarks import plot_local_topk
    from reproduction.figures import convergence, gp, main_comparison, scaling, scaling_curves, trajectory

    FIGURES.mkdir(parents=True, exist_ok=True)
    copyfile(ROOT / "reproduction/assets/fig1_architecture.jpg", FIGURES / "fig1_architecture.jpg")
    convergence.main()
    main_comparison.main()
    main_comparison.STRIP.unlink()
    scaling.main()
    trajectory.main()
    gp.main()
    plot_local_topk.main(output_dir=FIGURES, table_dir=TABLES)
    scaling_curves.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["tables", "figures"])
    parser.add_argument("--main-summary", help="Use a complete eleven-arm RF rerun for S7–S9, S14 and its contrasts (with --only tables)")
    args = parser.parse_args()
    if args.main_summary and args.only != "tables":
        parser.error("--main-summary requires --only tables")
    if args.only != "figures":
        from reproduction.tables import tables
        tables(args.main_summary)
        from benchmarks.plot_local_topk import summarize as topk_table
        topk_table().to_csv(TABLES / "bench_local_topk_table.csv", index=False)
    if args.only != "tables":
        figures()


if __name__ == "__main__":
    main()
