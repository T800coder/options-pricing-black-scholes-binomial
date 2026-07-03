"""
Convergence Analysis: Binomial Model -> Black-Scholes as n_steps increases
This demonstrates the theoretical result that BOPM converges to BS in the
limit of infinite time steps (CRR, 1979).
"""

import numpy as np
import pandas as pd
from pricing_models import black_scholes, binomial_price, S0, K, T, r, sigma, option_type

# Range of step counts to test
step_counts = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]

bs_price, bs_delta, _, _ = black_scholes(S0, K, T, r, sigma, option_type)

results = []
for n in step_counts:
    bopm_price, u, d, p = binomial_price(S0, K, T, r, sigma, n_steps=n, option_type=option_type)
    abs_error = abs(bopm_price - bs_price)
    pct_error = (abs_error / bs_price) * 100
    results.append({
        "n_steps": n,
        "binomial_price": round(bopm_price, 4),
        "black_scholes_price": round(bs_price, 4),
        "abs_error": round(abs_error, 4),
        "pct_error": round(pct_error, 4),
    })

df = pd.DataFrame(results)
df.to_csv("convergence_data.csv", index=False)
print(df.to_string(index=False))
print(f"\nAt n=1000 steps, Binomial converges to within {df.iloc[-1]['pct_error']:.3f}% of Black-Scholes price.")
