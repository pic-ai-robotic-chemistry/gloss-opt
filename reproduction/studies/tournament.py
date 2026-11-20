"""Historical discrete comparator used in SI Table S4."""
import numpy as np


class UCBTournament:
    """Select each batch by UCB tournaments with a distance penalty."""

    def __init__(self, kappa=2.0, tournament_size=20, seed=None):
        self.kappa = kappa
        self.tournament_size = tournament_size
        self.rng = np.random.default_rng(seed)

    def _fitness(self, gp, X):
        """UCB fitness scores for a batch of points X."""
        mu, sigma = gp.predict(X, return_std=True)
        return mu + self.kappa * sigma

    def recommend_discrete(self, X_obs, y_obs, candidates, y_true, n_points, gp):
        """Recommend from discrete candidates via tournament selection + niche penalty."""
        fitness = self._fitness(gp, candidates)

        obs_set = set(map(tuple, np.round(X_obs, 10)))
        available = np.array([
            i for i in range(len(candidates))
            if tuple(np.round(candidates[i], 10)) not in obs_set
        ])
        if len(available) == 0:
            return []

        selected = []
        selected_X = []

        avail_fit = fitness[available].copy()

        for _ in range(n_points):
            if len(available) == 0:
                break

            # Niche sharing: penalise candidates close to already-selected
            if selected_X:
                sel = np.array(selected_X)
                # min distance from each available candidate to any selected point
                dists = np.min(
                    np.linalg.norm(
                        candidates[available][:, None, :] - sel[None, :, :],
                        axis=2,
                    ),
                    axis=1,
                )
                niche_r = np.median(dists) + 1e-10
                penalty = np.exp(-dists / niche_r) * np.std(avail_fit)
                scores = avail_fit - penalty
            else:
                scores = avail_fit

            # Tournament selection
            t_size = min(self.tournament_size, len(available))
            tournament = self.rng.choice(len(available), size=t_size, replace=False)
            winner_local = tournament[np.argmax(scores[tournament])]
            winner_global = available[winner_local]

            selected.append(winner_global)
            selected_X.append(candidates[winner_global].copy())

            # Remove winner from available pool
            mask = np.ones(len(available), dtype=bool)
            mask[winner_local] = False
            available = available[mask]
            avail_fit = avail_fit[mask]

        return selected


def main():
    from benchmarks import bench_main as bench
    from reproduction.paths import OUTPUT
    import pandas as pd

    rows = []
    for level in bench.COMPLEXITY_LEVELS:
        dataset = bench.get_virtual_dataset(
            "arrhenius", n_candidates=level.get("n_candidates", 10000),
            n_peaks=level["n_peaks"], noise_sigma=level["noise_sigma"],
            peak_width=level["peak_width"], seed=42)
        dataset["name"] = level["label"]
        for seed in bench.SEEDS:
            summary, _ = bench.run_discrete(
                dataset, "UCB tournament", UCBTournament(seed=seed), seed)
            rows.extend(summary)
    path = OUTPUT / "reruns" / "S4_tournament.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(path)


if __name__ == "__main__":
    main()
