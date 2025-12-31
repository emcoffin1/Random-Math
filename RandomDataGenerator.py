from __future__ import annotations
import numpy as np

def generate_noisy_trend_txt(
    path: str = "noisy_trend_data.txt",
    n: int = 200,
    trend: str = "linear",          # "linear" | "exp" | "sin"
    seed: int | None = 0,
    x_start: float = 0.0,
    x_step: float = 1.0,
    slope: float = 0.5,             # used for linear and sin-trend
    intercept: float = 10.0,
    noise_std: float = 5.0,
    exp_rate: float = 0.02,         # used for exp
    sin_amp: float = 20.0,          # used for sin
    sin_period: float = 50.0        # used for sin
) -> str:
    """
    Generates (x, y) data with a trend plus Gaussian noise and saves to a .txt file.
    Output format: CSV with header 'x,y' (easy to open in Sheets/Excel).
    Returns the output path.
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    rng = np.random.default_rng(seed)
    x = x_start + x_step * np.arange(n)

    if trend.lower() == "linear":
        y_true = slope * x + intercept
    elif trend.lower() in ("exp", "exponential"):
        # exponential growth around an intercept baseline
        y_true = intercept + np.exp(exp_rate * (x - x_start)) * (slope * 10)
    elif trend.lower() in ("sin", "sine"):
        # sinusoid + linear trend
        y_true = slope * x + intercept + sin_amp * np.sin(2 * np.pi * (x - x_start) / sin_period)
    else:
        raise ValueError("trend must be one of: 'linear', 'exp', 'sin'")

    y = y_true + rng.normal(0.0, noise_std, size=n)

    with open(path, "w", encoding="utf-8") as f:
        f.write("x,y\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi},{yi:.6f}\n")

    return path

# Example usage:
# generate_noisy_trend_txt("linear_noisy.txt", n=150, trend="linear", slope=0.3, noise_std=3.0, seed=42)
generate_noisy_trend_txt("Random Stuff/sin_noisy.txt", n=300, trend="sin", slope=0.05, sin_amp=10, sin_period=60, noise_std=2.0, seed=1)
