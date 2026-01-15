"""
Equity Factor Model Simulation

This module implements a multi-factor equity model simulation with:
- 1 market factor
- 8 sector factors (9 factors total)
- Factor correlations
- Stock-specific residual risk

The simulation generates daily returns and calculates period average price accumulation factors.
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class FactorModelInputs:
    """
    Container for factor model parameters.

    Attributes:
        betas: Array of shape (n_stocks, n_factors) containing factor exposures
        factor_std: Array of shape (n_factors,) containing factor standard deviations (daily)
        factor_corr: Array of shape (n_factors, n_factors) containing factor correlation matrix
        residual_std: Array of shape (n_stocks,) containing stock residual standard deviations (daily)
        n_days: Number of days to simulate
        initial_prices: Optional array of initial stock prices (defaults to 100 for all stocks)
    """
    betas: np.ndarray
    factor_std: np.ndarray
    factor_corr: np.ndarray
    residual_std: np.ndarray
    n_days: int
    initial_prices: np.ndarray = None

    def __post_init__(self):
        """Validate inputs and set defaults."""
        self.n_stocks, self.n_factors = self.betas.shape

        # Validate dimensions
        assert self.factor_std.shape == (self.n_factors,), \
            f"factor_std must have shape ({self.n_factors},)"
        assert self.factor_corr.shape == (self.n_factors, self.n_factors), \
            f"factor_corr must have shape ({self.n_factors}, {self.n_factors})"
        assert self.residual_std.shape == (self.n_stocks,), \
            f"residual_std must have shape ({self.n_stocks},)"

        # Validate correlation matrix
        assert np.allclose(self.factor_corr, self.factor_corr.T), \
            "factor_corr must be symmetric"
        assert np.allclose(np.diag(self.factor_corr), 1.0), \
            "factor_corr diagonal must be 1.0"

        # Set default initial prices
        if self.initial_prices is None:
            self.initial_prices = np.full(self.n_stocks, 100.0)
        else:
            assert self.initial_prices.shape == (self.n_stocks,), \
                f"initial_prices must have shape ({self.n_stocks},)"


@dataclass
class SimulationResults:
    """
    Container for simulation results.

    Attributes:
        stock_returns: Array of shape (n_days, n_stocks) containing daily stock returns
        factor_returns: Array of shape (n_days, n_factors) containing daily factor returns
        residual_returns: Array of shape (n_days, n_stocks) containing daily residual returns
        stock_prices: Array of shape (n_days+1, n_stocks) containing stock prices
        avg_price_accumulation: Array of shape (n_stocks,) containing average price accumulation factors
    """
    stock_returns: np.ndarray
    factor_returns: np.ndarray
    residual_returns: np.ndarray
    stock_prices: np.ndarray
    avg_price_accumulation: np.ndarray


def generate_correlated_factor_returns(
    n_days: int,
    factor_std: np.ndarray,
    factor_corr: np.ndarray,
    random_seed: int = None
) -> np.ndarray:
    """
    Generate correlated factor returns using Cholesky decomposition.

    Args:
        n_days: Number of days to simulate
        factor_std: Factor standard deviations (daily)
        factor_corr: Factor correlation matrix
        random_seed: Random seed for reproducibility

    Returns:
        Array of shape (n_days, n_factors) containing factor returns
    """
    n_factors = len(factor_std)

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create covariance matrix from correlation and standard deviations
    # Cov = D * Corr * D, where D is diagonal matrix of standard deviations
    D = np.diag(factor_std)
    cov_matrix = D @ factor_corr @ D

    # Cholesky decomposition for generating correlated normal variables
    L = np.linalg.cholesky(cov_matrix)

    # Generate independent standard normal variables
    z = np.random.standard_normal((n_days, n_factors))

    # Transform to correlated factor returns
    factor_returns = z @ L.T

    return factor_returns


def generate_residual_returns(
    n_days: int,
    residual_std: np.ndarray,
    random_seed: int = None
) -> np.ndarray:
    """
    Generate uncorrelated residual returns for each stock.

    Args:
        n_days: Number of days to simulate
        residual_std: Residual standard deviations for each stock (daily)
        random_seed: Random seed for reproducibility

    Returns:
        Array of shape (n_days, n_stocks) containing residual returns
    """
    n_stocks = len(residual_std)

    # Set random seed if provided (offset by 1 to differ from factor seed)
    if random_seed is not None:
        np.random.seed(random_seed + 1)

    # Generate independent normal variables for each stock
    z = np.random.standard_normal((n_days, n_stocks))

    # Scale by residual standard deviations
    residual_returns = z * residual_std[np.newaxis, :]

    return residual_returns


def calculate_stock_returns(
    factor_returns: np.ndarray,
    residual_returns: np.ndarray,
    betas: np.ndarray
) -> np.ndarray:
    """
    Calculate stock returns from factor returns using the factor model.

    Stock return model: r_i,t = Σ(β_i,j * f_j,t) + ε_i,t

    Args:
        factor_returns: Array of shape (n_days, n_factors)
        residual_returns: Array of shape (n_days, n_stocks)
        betas: Array of shape (n_stocks, n_factors)

    Returns:
        Array of shape (n_days, n_stocks) containing stock returns
    """
    # Matrix multiplication: (n_days, n_factors) @ (n_factors, n_stocks)
    systematic_returns = factor_returns @ betas.T

    # Add residual returns
    stock_returns = systematic_returns + residual_returns

    return stock_returns


def calculate_price_path(
    returns: np.ndarray,
    initial_prices: np.ndarray
) -> np.ndarray:
    """
    Calculate price path from returns assuming simple returns (not log returns).

    Price formula: P_t = P_{t-1} * (1 + r_t)

    Args:
        returns: Array of shape (n_days, n_stocks) containing daily returns
        initial_prices: Array of shape (n_stocks,) containing initial prices

    Returns:
        Array of shape (n_days+1, n_stocks) containing prices (including initial price)
    """
    n_days, n_stocks = returns.shape

    # Initialize price array
    prices = np.zeros((n_days + 1, n_stocks))
    prices[0, :] = initial_prices

    # Calculate cumulative price path
    for t in range(n_days):
        prices[t + 1, :] = prices[t, :] * (1 + returns[t, :])

    return prices


def calculate_average_price_accumulation(
    prices: np.ndarray
) -> np.ndarray:
    """
    Calculate the average price accumulation factor for each stock.

    This factor is used to calculate each stock's average price over the period:
    avg_price = initial_price * avg_accumulation_factor

    The accumulation factor is calculated as the average of all price ratios:
    avg_accumulation = mean(P_t / P_0) for t=0 to T

    Args:
        prices: Array of shape (n_days+1, n_stocks) containing prices

    Returns:
        Array of shape (n_stocks,) containing average price accumulation factors
    """
    initial_prices = prices[0, :]

    # Calculate price ratios relative to initial price
    price_ratios = prices / initial_prices[np.newaxis, :]

    # Average over time (including t=0 where ratio is 1.0)
    avg_accumulation = np.mean(price_ratios, axis=0)

    return avg_accumulation


def simulate_equity_factor_model(
    inputs: FactorModelInputs,
    random_seed: int = None
) -> SimulationResults:
    """
    Run the complete equity factor model simulation.

    Args:
        inputs: FactorModelInputs object containing all model parameters
        random_seed: Random seed for reproducibility

    Returns:
        SimulationResults object containing all simulation outputs
    """
    # Generate correlated factor returns
    factor_returns = generate_correlated_factor_returns(
        n_days=inputs.n_days,
        factor_std=inputs.factor_std,
        factor_corr=inputs.factor_corr,
        random_seed=random_seed
    )

    # Generate uncorrelated residual returns
    residual_returns = generate_residual_returns(
        n_days=inputs.n_days,
        residual_std=inputs.residual_std,
        random_seed=random_seed
    )

    # Calculate stock returns from factor model
    stock_returns = calculate_stock_returns(
        factor_returns=factor_returns,
        residual_returns=residual_returns,
        betas=inputs.betas
    )

    # Calculate price paths
    stock_prices = calculate_price_path(
        returns=stock_returns,
        initial_prices=inputs.initial_prices
    )

    # Calculate average price accumulation factors
    avg_price_accumulation = calculate_average_price_accumulation(
        prices=stock_prices
    )

    return SimulationResults(
        stock_returns=stock_returns,
        factor_returns=factor_returns,
        residual_returns=residual_returns,
        stock_prices=stock_prices,
        avg_price_accumulation=avg_price_accumulation
    )


def create_example_model(
    n_stocks: int = 50,
    n_days: int = 252
) -> Tuple[FactorModelInputs, SimulationResults]:
    """
    Create and run an example factor model simulation.

    Creates a model with:
    - 1 market factor
    - 8 sector factors
    - Random but realistic parameters

    Args:
        n_stocks: Number of stocks to simulate
        n_days: Number of trading days to simulate (252 = 1 year)

    Returns:
        Tuple of (inputs, results)
    """
    np.random.seed(42)

    n_factors = 9  # 1 market + 8 sectors

    # Create factor labels
    factor_names = ['Market'] + [f'Sector_{i+1}' for i in range(8)]

    # Generate betas
    # - Market beta: random around 1.0
    # - Sector betas: each stock has high exposure to one sector
    betas = np.zeros((n_stocks, n_factors))
    betas[:, 0] = np.random.normal(1.0, 0.3, n_stocks)  # Market betas

    # Assign each stock to a sector (round-robin)
    for i in range(n_stocks):
        sector_idx = (i % 8) + 1  # Sectors are factors 1-8
        betas[i, sector_idx] = np.random.uniform(0.5, 1.5)

    # Factor standard deviations (daily, annualized ~16% for market)
    factor_std = np.zeros(n_factors)
    factor_std[0] = 0.01  # Market: 1% daily std (~16% annualized)
    factor_std[1:] = np.random.uniform(0.008, 0.012, 8)  # Sectors: 0.8-1.2% daily

    # Factor correlation matrix (ensure positive definite)
    # Generate random matrix and convert to valid correlation matrix
    A = np.random.randn(n_factors, n_factors)
    temp_corr = A @ A.T
    # Convert to correlation matrix (standardize to have 1s on diagonal)
    factor_corr = np.zeros((n_factors, n_factors))
    for i in range(n_factors):
        for j in range(n_factors):
            factor_corr[i, j] = temp_corr[i, j] / np.sqrt(temp_corr[i, i] * temp_corr[j, j])

    # Adjust to get desired correlation structure
    # Scale down correlations and boost diagonal
    factor_corr = 0.3 * factor_corr + 0.7 * np.eye(n_factors)

    # Set market-sector correlations to be moderate
    factor_corr[0, 1:] = 0.6 * factor_corr[0, 1:]
    factor_corr[1:, 0] = factor_corr[0, 1:]

    # Residual standard deviations (stock-specific risk)
    residual_std = np.random.uniform(0.015, 0.025, n_stocks)  # 1.5-2.5% daily

    # Create inputs
    inputs = FactorModelInputs(
        betas=betas,
        factor_std=factor_std,
        factor_corr=factor_corr,
        residual_std=residual_std,
        n_days=n_days,
        initial_prices=np.full(n_stocks, 100.0)
    )

    # Run simulation
    results = simulate_equity_factor_model(inputs, random_seed=42)

    return inputs, results


if __name__ == "__main__":
    """
    Example usage of the equity factor model simulation.
    """
    print("=" * 80)
    print("Equity Factor Model Simulation Example")
    print("=" * 80)
    print()

    # Create and run example model
    n_stocks = 50
    n_days = 252  # One trading year

    print(f"Simulating {n_stocks} stocks over {n_days} trading days...")
    print(f"Factor model: 1 market factor + 8 sector factors")
    print()

    inputs, results = create_example_model(n_stocks=n_stocks, n_days=n_days)

    # Display results
    print("Simulation Results:")
    print("-" * 80)
    print(f"Stock returns shape: {results.stock_returns.shape}")
    print(f"Factor returns shape: {results.factor_returns.shape}")
    print(f"Stock prices shape: {results.stock_prices.shape}")
    print()

    print("Average Price Accumulation Factors:")
    print("-" * 80)
    print(f"Mean: {np.mean(results.avg_price_accumulation):.4f}")
    print(f"Std:  {np.std(results.avg_price_accumulation):.4f}")
    print(f"Min:  {np.min(results.avg_price_accumulation):.4f}")
    print(f"Max:  {np.max(results.avg_price_accumulation):.4f}")
    print()

    # Display sample of average price accumulation factors
    print("Sample stocks (first 10):")
    for i in range(min(10, n_stocks)):
        initial_price = inputs.initial_prices[i]
        final_price = results.stock_prices[-1, i]
        avg_price = initial_price * results.avg_price_accumulation[i]
        print(f"  Stock {i:2d}: "
              f"Initial=${initial_price:7.2f}, "
              f"Final=${final_price:7.2f}, "
              f"Average=${avg_price:7.2f}, "
              f"AccumFactor={results.avg_price_accumulation[i]:.4f}")
    print()

    # Factor return statistics
    print("Factor Return Statistics (daily):")
    print("-" * 80)
    factor_names = ['Market'] + [f'Sector_{i+1}' for i in range(8)]
    for i, name in enumerate(factor_names):
        mean_ret = np.mean(results.factor_returns[:, i])
        std_ret = np.std(results.factor_returns[:, i])
        print(f"  {name:10s}: Mean={mean_ret:8.5f}, Std={std_ret:8.5f}")
    print()

    # Stock return statistics
    print("Stock Return Statistics (daily):")
    print("-" * 80)
    stock_mean = np.mean(results.stock_returns, axis=0)
    stock_std = np.std(results.stock_returns, axis=0)
    print(f"  Mean of means: {np.mean(stock_mean):.5f}")
    print(f"  Mean of stds:  {np.mean(stock_std):.5f}")
    print(f"  Min return:    {np.min(results.stock_returns):.5f}")
    print(f"  Max return:    {np.max(results.stock_returns):.5f}")
    print()

    print("=" * 80)
    print("Simulation complete!")
    print("=" * 80)
