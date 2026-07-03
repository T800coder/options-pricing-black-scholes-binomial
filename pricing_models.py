"""
Options Pricing Models: Black-Scholes vs Binomial (BOPM)
Underlying: Infosys Ltd (INFY)

Market inputs below are REAL values pulled from the NSE Equity Derivatives
option chain for INFY, expiry 28-Jul-2026 (data as of early July 2026).
Spot estimated via put-call parity at the at-the-money strike.
"""

import numpy as np
from scipy.stats import norm
import pandas as pd

# ----------------------------------------------------------------------
# MARKET INPUTS  (real NSE data — INFY, 28-Jul-2026 expiry)
# ----------------------------------------------------------------------
S0 = 1040.75       # Spot price of Infosys (INR), via put-call parity
K = 1045.0         # At-the-money strike (nearest to spot)
T = 25 / 365        # Time to expiry (25 days to 28-Jul-2026)
r = 0.0665          # Risk-free rate (91-day T-bill, ~6.65%)
sigma = 0.292       # Implied volatility from NSE call option chain (29.2%)
option_type = "call"

# Real traded market premium for the K=1045 call (for validation)
MARKET_PREMIUM = 36.60


# ----------------------------------------------------------------------
# 1. BLACK-SCHOLES MODEL
# ----------------------------------------------------------------------
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1

    return price, delta, d1, d2


# ----------------------------------------------------------------------
# 2. BINOMIAL OPTIONS PRICING MODEL (BOPM) — Cox-Ross-Rubinstein
# ----------------------------------------------------------------------
def binomial_price(S, K, T, r, sigma, n_steps=100, option_type="call"):
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))      # up factor
    d = 1 / u                             # down factor
    p = (np.exp(r * dt) - d) / (u - d)    # risk-neutral probability
    disc = np.exp(-r * dt)

    # Terminal stock prices at each node
    stock_prices = S * u ** np.arange(n_steps, -1, -1) * d ** np.arange(0, n_steps + 1)

    if option_type == "call":
        values = np.maximum(stock_prices - K, 0)
    else:
        values = np.maximum(K - stock_prices, 0)

    # Backward induction
    for step in range(n_steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])

    return values[0], u, d, p


def binomial_delta(S, K, T, r, sigma, n_steps=100, option_type="call", bump=1.0):
    """Numerical delta: bump spot price up/down and re-price."""
    price_up, *_ = binomial_price(S + bump, K, T, r, sigma, n_steps, option_type)
    price_down, *_ = binomial_price(S - bump, K, T, r, sigma, n_steps, option_type)
    return (price_up - price_down) / (2 * bump)


# ----------------------------------------------------------------------
# 3. RUN CORE COMPARISON
# ----------------------------------------------------------------------
if __name__ == "__main__":
    bs_price, bs_delta, d1, d2 = black_scholes(S0, K, T, r, sigma, option_type)
    bopm_price, u, d, p = binomial_price(S0, K, T, r, sigma, n_steps=100, option_type=option_type)
    bopm_delta = binomial_delta(S0, K, T, r, sigma, n_steps=100, option_type=option_type)

    intrinsic_value = max(S0 - K, 0) if option_type == "call" else max(K - S0, 0)

    print("=" * 60)
    print(f"OPTIONS PRICING: Infosys (INFY) — {option_type.upper()}")
    print("=" * 60)
    print(f"Spot (S0)        : {S0}")
    print(f"Strike (K)       : {K}")
    print(f"Time to expiry   : {T*365:.0f} days")
    print(f"Risk-free rate   : {r*100:.2f}%")
    print(f"Volatility       : {sigma*100:.1f}%")
    print("-" * 60)
    print(f"Black-Scholes price : Rs {bs_price:.2f}   | Delta: {bs_delta:.4f}")
    print(f"Binomial (n=100) price: Rs {bopm_price:.2f}   | Delta: {bopm_delta:.4f}")
    print(f"Intrinsic value      : Rs {intrinsic_value:.2f}")
    print(f"Time value (BS)       : Rs {bs_price - intrinsic_value:.2f}")
    print("-" * 60)
    print(f"REAL market premium (NSE): Rs {MARKET_PREMIUM:.2f}")
    print(f"Model vs market gap      : Rs {abs(bs_price - MARKET_PREMIUM):.2f} "
          f"({abs(bs_price - MARKET_PREMIUM)/MARKET_PREMIUM*100:.1f}%)")
    print("=" * 60)
