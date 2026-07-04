# Options Pricing Models: Black-Scholes vs Binomial vs Monte Carlo

A self-initiated project implementing and cross-validating three independent European option pricing methods on Infosys (INFY), using **real NSE option chain data**.

## Summary

This project prices an at-the-money European call option on Infosys (INFY, 28-Jul-2026 expiry) using three theoretically distinct approaches, validates them against each other, and benchmarks against the **actual traded market premium**.

| Method | Price (Rs) | Delta |
|---|---|---|
| Black-Scholes (closed-form) | 31.96 | 0.5177 |
| Binomial CRR Lattice (n=100) | 32.01 | 0.4993 |
| Monte Carlo (100,000 simulations) | 32.04 ± 0.30 | — |
| **Real NSE market premium** | **36.60** | — |

**Max deviation across the three models: 0.24%** — validating implementation correctness across closed-form, discrete-lattice, and simulation-based approaches.

## Market inputs (real NSE data, INFY 28-Jul-2026)

- **Spot:** Rs 1,040.75 (estimated via put-call parity at the ATM strike)
- **Strike:** Rs 1,045 (nearest to spot)
- **Expiry:** 28-Jul-2026 (25 days out)
- **Risk-free rate:** 6.65% (91-day T-bill proxy)
- **Implied volatility:** 29.2% (from NSE call option chain)
- **Traded call premium:** Rs 36.60 (LTP)

## What's inside

- **`pricing_models.py`** — Black-Scholes and Binomial (Cox-Ross-Rubinstein) implementations with numerical Delta
- **`convergence_analysis.py`** — Binomial to Black-Scholes convergence (1 to 1,000 steps)
- **`monte_carlo.py`** — Risk-neutral GBM simulation (100,000 paths)
- **`generate_charts.py`** — All visualization code
- **`Options_Pricing_Models.ipynb`** — Full interactive notebook with math, code, and results
- **`chart1-5*.png`** — Convergence, payoff/time-value, Delta sensitivity, binomial lattice, and Monte Carlo charts

## Key findings

**1. Three-way model agreement:** Black-Scholes, Binomial (n=100), and Monte Carlo (100K sims) all price the option within 0.24% of each other — cross-validating three fundamentally different approaches.

**2. Convergence:** The Binomial model's error vs Black-Scholes shrinks predictably as steps increase (23% at n=1 to 0.02% at n=1000), empirically confirming the CRR convergence theorem. The oscillation at low step counts is expected for at-the-money options — it arises from whether the strike lands near a node or between nodes at each step count, and it damps out as n grows.

**3. Model vs market gap (~12.7%):** The models price the call ~Rs 4.6 below the traded premium. This gap is itself informative and reflects real market microstructure: (a) the strike sits slightly above spot, making the option marginally out-of-the-money; (b) the bid-ask spread on the traded premium; and (c) volatility skew — NSE puts on INFY showed ~37% IV vs ~29% for calls, so the single-IV assumption is a simplification. Recognizing why a model diverges from the market is as important as the model itself.

**4. Delta consistency:** Both models agree on Delta (~0.50-0.52), confirming consistent hedging sensitivity independent of methodology.

## How to run

```bash
pip install numpy pandas matplotlib scipy jupyter
python3 pricing_models.py          # core comparison + market validation
python3 convergence_analysis.py    # convergence table
python3 generate_charts.py         # all static charts
python3 monte_carlo.py             # Monte Carlo cross-check
jupyter notebook Options_Pricing_Models.ipynb   # full walkthrough
```

## Possible extensions

- Extend Binomial framework to American options (early exercise)
- Add Gamma, Vega, Theta sensitivity analysis
- Back out implied volatility from the traded premium and compare to NSE's reported IV
- Model the call-put IV skew rather than assuming a single volatility

---
*Self-initiated project | Ishitva Yadav, IIT Bombay*
