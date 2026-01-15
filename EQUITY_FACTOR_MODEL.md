# Equity Factor Model Simulation

A Python implementation of a multi-factor equity model simulator with correlated factors and stock-specific residual risk.

## Overview

This module simulates stock returns using a factor model with:
- **1 Market Factor**: Represents broad market movements
- **8 Sector Factors**: Represent sector-specific movements
- **Factor Correlations**: Realistic correlations between factors
- **Residual Risk**: Stock-specific idiosyncratic risk

## Factor Model Formula

The return for stock `i` at time `t` is calculated as:

```
r_i,t = Σ(β_i,j × f_j,t) + ε_i,t
```

Where:
- `r_i,t` = return of stock i at time t
- `β_i,j` = beta (exposure) of stock i to factor j
- `f_j,t` = return of factor j at time t
- `ε_i,t` = idiosyncratic (residual) return of stock i at time t

## Key Features

1. **Correlated Factor Returns**: Uses Cholesky decomposition to generate properly correlated factor returns
2. **Flexible Parameters**: Customize betas, volatilities, correlations, and residual risk
3. **Price Simulation**: Simulates daily price paths from returns
4. **Average Price Accumulation**: Calculates period average price factors for portfolio analysis
5. **Validated Inputs**: Automatic validation of correlation matrices and dimensions

## Installation

```bash
pip install numpy
```

## Quick Start

### Basic Example

```python
from equity_factor_simulation import create_example_model

# Create and run a default model (50 stocks, 252 days)
inputs, results = create_example_model()

# Access results
print("Average accumulation factors:", results.avg_price_accumulation)
print("Final prices:", results.stock_prices[-1, :])
```

### Custom Model

```python
import numpy as np
from equity_factor_simulation import FactorModelInputs, simulate_equity_factor_model

# Define model parameters
n_stocks = 20
n_factors = 9  # 1 market + 8 sectors

# Stock betas (exposures to factors)
betas = np.random.randn(n_stocks, n_factors)
betas[:, 0] = 1.0  # Set market beta to 1.0

# Factor standard deviations (daily)
factor_std = np.full(n_factors, 0.01)  # 1% daily std

# Factor correlation matrix (must be positive definite)
factor_corr = np.eye(n_factors)
# Add some correlation between factors
factor_corr[0, 1:] = 0.6  # Market correlates with sectors
factor_corr[1:, 0] = 0.6

# Stock residual standard deviations
residual_std = np.full(n_stocks, 0.02)  # 2% daily std

# Create model inputs
inputs = FactorModelInputs(
    betas=betas,
    factor_std=factor_std,
    factor_corr=factor_corr,
    residual_std=residual_std,
    n_days=252
)

# Run simulation
results = simulate_equity_factor_model(inputs, random_seed=42)
```

## Output Structure

The simulation returns a `SimulationResults` object with:

- **`stock_returns`**: Array of shape `(n_days, n_stocks)` - Daily stock returns
- **`factor_returns`**: Array of shape `(n_days, n_factors)` - Daily factor returns
- **`residual_returns`**: Array of shape `(n_days, n_stocks)` - Daily residual returns
- **`stock_prices`**: Array of shape `(n_days+1, n_stocks)` - Stock prices (including initial)
- **`avg_price_accumulation`**: Array of shape `(n_stocks,)` - Average price accumulation factors

## Average Price Accumulation Factor

The average price accumulation factor is calculated as:

```
avg_accumulation_i = mean(P_i,t / P_i,0) for t=0 to T
```

This factor is useful for calculating the average price over a period:

```
average_price_i = initial_price_i × avg_accumulation_i
```

This is commonly used in options pricing (e.g., Asian options) and portfolio analytics.

## Examples

See the included example files:

1. **`equity_factor_simulation.py`**: Main module with built-in example
   ```bash
   python equity_factor_simulation.py
   ```

2. **`example_usage.py`**: Comprehensive examples including:
   - Custom model with sector assignments
   - Monte Carlo simulations
   ```bash
   python example_usage.py
   ```

## Input Validation

The module automatically validates:

- ✓ Dimension consistency across all inputs
- ✓ Correlation matrix is symmetric
- ✓ Correlation matrix diagonal equals 1.0
- ✓ Correlation matrix is positive definite (via Cholesky decomposition)

## Technical Details

### Generating Correlated Returns

The module uses Cholesky decomposition to generate correlated factor returns:

1. Construct covariance matrix: `Σ = D × Corr × D`
2. Compute Cholesky decomposition: `Σ = L × L^T`
3. Generate independent normals: `Z ~ N(0, I)`
4. Transform to correlated returns: `F = Z × L^T`

### Price Path Calculation

Prices are calculated using simple returns (not log returns):

```
P_t = P_{t-1} × (1 + r_t)
```

This approach is standard in equity modeling and matches typical return calculations.

## Use Cases

- **Portfolio Risk Analysis**: Simulate portfolio returns under different scenarios
- **Options Pricing**: Calculate average price accumulation for Asian options
- **Stress Testing**: Test portfolio behavior under extreme market conditions
- **Strategy Backtesting**: Generate realistic return scenarios for strategy testing
- **Risk Management**: Estimate VaR and CVaR using Monte Carlo simulations

## Mathematical Background

The factor model decomposes stock returns into:
1. **Systematic risk**: Explained by common factors (market, sectors)
2. **Idiosyncratic risk**: Stock-specific risk uncorrelated with factors

This decomposition is fundamental to modern portfolio theory and is used extensively in:
- Risk management (e.g., Barra risk models)
- Portfolio construction
- Performance attribution
- Hedging strategies

## References

- Fama-French factor models
- Barra risk models
- APT (Arbitrage Pricing Theory)
- Multi-factor portfolio construction

## License

See LICENSE file in the repository.
