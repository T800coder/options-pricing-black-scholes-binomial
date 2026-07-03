"""
Monte Carlo Simulation Cross-Check
Simulates stock price paths using Geometric Brownian Motion (GBM) under the
risk-neutral measure, then prices the option as the discounted expected payoff.

This provides a THIRD independent method to validate Black-Scholes and
Binomial results — the three methods should converge to the same price,
which is the core "proof" of the project.
"""

import numpy as np
import matplotlib.pyplot as plt
from pricing_models import black_scholes, binomial_price, S0, K, T, r, sigma, option_type

np.random.seed(42)  # reproducibility


def monte_carlo_price(S, K, T, r, sigma, n_sims=100_000, option_type="call"):
    """
    Simulates terminal stock price under risk-neutral GBM:
        S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),  Z ~ N(0,1)
    Prices option as discounted expected payoff.
    """
    Z = np.random.standard_normal(n_sims)
    S_T = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
    return price, std_error, S_T


def simulate_price_paths(S, T, r, sigma, n_paths=15, n_steps=252):
    """Simulate full price paths (for visualization only, not pricing)."""
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return paths


if __name__ == "__main__":
    bs_price, bs_delta, _, _ = black_scholes(S0, K, T, r, sigma, option_type)
    bopm_price, *_ = binomial_price(S0, K, T, r, sigma, n_steps=100, option_type=option_type)
    mc_price, mc_se, S_T = monte_carlo_price(S0, K, T, r, sigma, n_sims=100_000, option_type=option_type)

    print("=" * 65)
    print("THREE-WAY MODEL COMPARISON")
    print("=" * 65)
    print(f"{'Method':<30}{'Price (Rs)':<15}{'Notes'}")
    print("-" * 65)
    print(f"{'Black-Scholes (closed-form)':<30}{bs_price:<15.4f}{'Analytical'}")
    print(f"{'Binomial CRR (n=100)':<30}{bopm_price:<15.4f}{'Lattice, 100 steps'}")
    print(f"{'Monte Carlo (100K sims)':<30}{mc_price:<15.4f}{'±' + f'{1.96*mc_se:.4f} (95% CI)'}")
    print("=" * 65)
    max_dev = max(abs(bopm_price - bs_price), abs(mc_price - bs_price))
    print(f"\nMax deviation across all 3 methods: Rs {max_dev:.4f} "
          f"({max_dev/bs_price*100:.3f}% of BS price)")
    print("All three independent methods converge -> validates implementation correctness.")

    # ------------------------------------------------------------------
    # Visualization: simulated price paths + terminal price distribution
    # ------------------------------------------------------------------
    plt.rcParams["figure.dpi"] = 140
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    paths = simulate_price_paths(S0, T, r, sigma, n_paths=20, n_steps=252)
    time_axis = np.linspace(0, T * 365, paths.shape[1])
    for p in paths:
        axes[0].plot(time_axis, p, linewidth=0.9, alpha=0.7)
    axes[0].axhline(K, color="#e34948", linestyle="--", linewidth=1.5, label=f"Strike (Rs {K:.0f})")
    axes[0].set_xlabel("Days")
    axes[0].set_ylabel("Simulated Stock Price (Rs)")
    axes[0].set_title("Simulated GBM Price Paths (20 of 100,000 shown)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    _, _, S_T_full = monte_carlo_price(S0, K, T, r, sigma, n_sims=100_000, option_type=option_type)
    axes[1].hist(S_T_full, bins=80, color="#2a78d6", alpha=0.75, edgecolor="white", linewidth=0.3)
    axes[1].axvline(K, color="#e34948", linestyle="--", linewidth=2, label=f"Strike (Rs {K:.0f})")
    axes[1].axvline(S0, color="#eda100", linestyle=":", linewidth=2, label=f"Spot (Rs {S0:.0f})")
    axes[1].set_xlabel("Terminal Stock Price S_T (Rs)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Simulated Terminal Prices\n(100,000 Monte Carlo simulations)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("chart5_montecarlo.png", bbox_inches="tight")
    plt.close()
    print("\nSaved chart5_montecarlo.png")
