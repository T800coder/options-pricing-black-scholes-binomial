"""
Visualization suite for Options Pricing Project
Generates:
1. Convergence chart (Binomial -> Black-Scholes)
2. Payoff diagram (intrinsic value vs time value)
3. Delta sensitivity chart (Delta vs spot price)
4. Binomial lattice tree diagram (small n, illustrative)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pricing_models import black_scholes, binomial_price, binomial_delta, S0, K, T, r, sigma, option_type

plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 10

# ------------------------------------------------------------------
# CHART 1: Convergence — Binomial price vs number of steps
# ------------------------------------------------------------------
df = pd.read_csv("convergence_data.csv")
bs_price = df["black_scholes_price"].iloc[0]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["n_steps"], df["binomial_price"], marker="o", color="#2a78d6",
        linewidth=2, markersize=5, label="Binomial (CRR) Price")
ax.axhline(bs_price, color="#e34948", linestyle="--", linewidth=2,
           label=f"Black-Scholes Price (Rs {bs_price:.2f})")
ax.set_xscale("log")
ax.set_xlabel("Number of Time Steps (n) — log scale")
ax.set_ylabel("Option Price (Rs)")
ax.set_title("Binomial Model Convergence to Black-Scholes\nInfosys (INFY) 30-Day At-The-Money Call")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("chart1_convergence.png", bbox_inches="tight")
plt.close()
print("Saved chart1_convergence.png")

# ------------------------------------------------------------------
# CHART 2: Payoff diagram — intrinsic value vs time value
# ------------------------------------------------------------------
spot_range = np.linspace(S0 * 0.75, S0 * 1.25, 200)
intrinsic = np.maximum(spot_range - K, 0)
bs_prices_range = [black_scholes(s, K, T, r, sigma, option_type)[0] for s in spot_range]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(spot_range, intrinsic, color="#333333", linewidth=2, linestyle="--",
        label="Intrinsic Value (Payoff at Expiry)")
ax.plot(spot_range, bs_prices_range, color="#1baf7a", linewidth=2.5,
        label="Black-Scholes Price (30 days to expiry)")
ax.axvline(S0, color="#999999", linestyle=":", linewidth=1.5, label=f"Current Spot (Rs {S0:.0f})")
ax.axvline(K, color="#eda100", linestyle=":", linewidth=1.5, label=f"Strike (Rs {K:.0f})")
ax.fill_between(spot_range, intrinsic, bs_prices_range, where=(np.array(bs_prices_range) >= intrinsic),
                 color="#1baf7a", alpha=0.12, label="Time Value")
ax.set_xlabel("Underlying Stock Price (Rs)")
ax.set_ylabel("Option Value (Rs)")
ax.set_title("Call Option Payoff: Intrinsic Value vs Time Value\nInfosys (INFY) — Isolating Time Value Component")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("chart2_payoff_timevalue.png", bbox_inches="tight")
plt.close()
print("Saved chart2_payoff_timevalue.png")

# ------------------------------------------------------------------
# CHART 3: Delta sensitivity — Delta vs Spot Price (both models)
# ------------------------------------------------------------------
spot_range_delta = np.linspace(S0 * 0.85, S0 * 1.15, 40)
bs_deltas = [black_scholes(s, K, T, r, sigma, option_type)[1] for s in spot_range_delta]
bopm_deltas = [binomial_delta(s, K, T, r, sigma, n_steps=100, option_type=option_type) for s in spot_range_delta]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(spot_range_delta, bs_deltas, color="#e34948", linewidth=2.5, label="Black-Scholes Delta")
ax.plot(spot_range_delta, bopm_deltas, color="#2a78d6", linewidth=2, linestyle="--",
        marker="o", markersize=3, label="Binomial Delta (n=100)")
ax.axvline(S0, color="#999999", linestyle=":", linewidth=1.5, label="Current Spot")
ax.set_xlabel("Underlying Stock Price (Rs)")
ax.set_ylabel("Delta")
ax.set_title("Delta Sensitivity Across Models\nPrice Sensitivity of Call Option to Underlying Movement")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("chart3_delta_sensitivity.png", bbox_inches="tight")
plt.close()
print("Saved chart3_delta_sensitivity.png")

# ------------------------------------------------------------------
# CHART 4: Small binomial lattice tree (illustrative, n=4)
# ------------------------------------------------------------------
def draw_lattice(S, K, T, r, sigma, n_steps=4):
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(n_steps + 1):
        for j in range(i + 1):
            price = S * (u ** (i - j)) * (d ** j)
            x, y = i, i / 2 - j
            ax.scatter(x, y, color="#2a78d6", zorder=3, s=40)
            ax.annotate(f"{price:.0f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
            if i < n_steps:
                ax.plot([x, x + 1], [y, y + 0.5], color="#cccccc", zorder=1, linewidth=1)
                ax.plot([x, x + 1], [y, y - 0.5], color="#cccccc", zorder=1, linewidth=1)

    ax.axhline(0, color="#eda100", linestyle=":", alpha=0.5)
    ax.set_title(f"Binomial Lattice Structure (Illustrative, n={n_steps} steps)\nu={u:.4f}, d={d:.4f} — Stock Price Paths")
    ax.set_xlabel("Time Step")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("chart4_binomial_lattice.png", bbox_inches="tight")
    plt.close()
    print("Saved chart4_binomial_lattice.png")

draw_lattice(S0, K, T, r, sigma, n_steps=4)

print("\nAll charts generated successfully.")
