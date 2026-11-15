"""Virtual chemical reaction optimization function for continuous-space benchmarking.

6D function simulating a complex chemical process optimization:
- x0: Temperature (100-300 °C)
- x1: Catalyst concentration (0.1-1.0 mol/L)
- x2: Pressure (1-50 atm)
- x3: Reaction time (0.5-24 h)
- x4: Solvent polarity index (0-10)
- x5: pH (2-12)

Multi-modal with deceptive peaks, ridges, and interaction terms.
"""

import numpy as np


BOUNDS = [
    (100, 300),   # Temperature
    (0.1, 1.0),   # Catalyst conc
    (1, 50),      # Pressure
    (0.5, 24),    # Time
    (0, 10),      # Solvent polarity
    (2, 12),      # pH
]

PARAM_NAMES = ["Temperature", "Catalyst", "Pressure", "Time", "Polarity", "pH"]


def _normalize_to_unit(X):
    """Normalize from real bounds to [0,1] for internal computation."""
    X = np.atleast_2d(X)
    bounds = np.array(BOUNDS)
    return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def chemical_yield(X):
    """Evaluate the 6D virtual chemical yield function.

    Args:
        X: np.ndarray of shape (n, 6) in real parameter space.

    Returns:
        np.ndarray of shape (n,) — yield values.
    """
    X = np.atleast_2d(X)
    u = _normalize_to_unit(X)  # all in [0, 1]

    T, C, P, t, S, pH = u[:, 0], u[:, 1], u[:, 2], u[:, 3], u[:, 4], u[:, 5]

    y = np.zeros(len(X))

    # Peak 1: global optimum — requires specific T, C, P, pH alignment
    # (T~180°C -> u~0.4, C~0.6, P~20atm -> u~0.39, pH~7 -> u~0.5)
    d1 = ((T - 0.40)**2 / 0.02 + (C - 0.60)**2 / 0.03 +
          (P - 0.39)**2 / 0.04 + (pH - 0.50)**2 / 0.03)
    y += 95 * np.exp(-d1)

    # Peak 2: deceptive local optimum — high yield but wrong region
    # (T~250°C -> u~0.75, C~0.3, P~40atm -> u~0.80, pH~4 -> u~0.2)
    d2 = ((T - 0.75)**2 / 0.01 + (C - 0.30)**2 / 0.02 +
          (P - 0.80)**2 / 0.02 + (pH - 0.20)**2 / 0.02)
    y += 85 * np.exp(-d2)

    # Peak 3: broad moderate peak
    d3 = ((T - 0.55)**2 / 0.08 + (C - 0.50)**2 / 0.08 +
          (P - 0.50)**2 / 0.08 + (pH - 0.60)**2 / 0.08)
    y += 55 * np.exp(-d3)

    # Time-temperature interaction: optimal reaction time depends on temperature
    y += 15 * np.exp(-((t - 0.3 - 0.4 * T)**2) / 0.05)

    # Solvent-pH interaction: some solvent-pH combos are catastrophic
    y -= 20 * np.exp(-((S - 0.8)**2 / 0.02 + (pH - 0.9)**2 / 0.02))

    # Pressure-temperature ridge: higher pressure helps at higher temps
    y += 10 * np.exp(-((P - 0.3 - 0.5 * T)**2) / 0.03)

    # High-frequency oscillations (experimental noise / micro-kinetics)
    y += 3 * np.sin(8 * np.pi * T) * np.cos(6 * np.pi * C)
    y += 2 * np.sin(5 * np.pi * P) * np.cos(4 * np.pi * pH)

    # Penalty for extreme conditions
    y -= 10 * (T > 0.9).astype(float) * (T - 0.9)**2 / 0.01  # too hot
    y -= 10 * (pH < 0.1).astype(float) * (0.1 - pH)**2 / 0.01  # too acidic

    return y


def get_true_optimum(n_grid=500000):
    """Find the true optimum by dense random search."""
    rng = np.random.default_rng(0)
    bounds = np.array(BOUNDS)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_grid, 6))
    y = chemical_yield(X)
    best_idx = np.argmax(y)
    return X[best_idx], y[best_idx]
