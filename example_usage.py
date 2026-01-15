"""
Example usage of the equity factor simulation module with custom parameters.

This script demonstrates how to create a custom factor model and run simulations.
"""

import numpy as np
from equity_factor_simulation import (
    FactorModelInputs,
    simulate_equity_factor_model
)


def example_custom_model():
    """
    Example of creating a custom factor model with specific parameters.
    """
    print("=" * 80)
    print("Custom Equity Factor Model Example")
    print("=" * 80)
    print()

    # Define model dimensions
    n_stocks = 20
    n_factors = 9  # 1 market + 8 sectors
    n_days = 126  # Half a trading year

    print(f"Model configuration:")
    print(f"  - Number of stocks: {n_stocks}")
    print(f"  - Number of factors: {n_factors} (1 market + 8 sectors)")
    print(f"  - Simulation period: {n_days} days")
    print()

    # Define betas (factor exposures)
    # Each row represents a stock, each column a factor
    betas = np.zeros((n_stocks, n_factors))

    # Market betas: all stocks have exposure to market
    # Using different beta values to represent different stock types
    betas[0:5, 0] = 0.5   # Low beta stocks (defensive)
    betas[5:15, 0] = 1.0  # Market beta stocks
    betas[15:20, 0] = 1.5 # High beta stocks (aggressive)

    # Sector betas: assign stocks to sectors
    # First 5 stocks -> Sector 1 (Technology)
    betas[0:5, 1] = 1.2

    # Next 3 stocks -> Sector 2 (Healthcare)
    betas[5:8, 2] = 1.1

    # Next 3 stocks -> Sector 3 (Financials)
    betas[8:11, 3] = 1.3

    # Next 3 stocks -> Sector 4 (Energy)
    betas[11:14, 4] = 0.9

    # Next 2 stocks -> Sector 5 (Consumer)
    betas[14:16, 5] = 1.0

    # Next 2 stocks -> Sector 6 (Industrials)
    betas[16:18, 6] = 1.1

    # Next 1 stock -> Sector 7 (Materials)
    betas[18:19, 7] = 0.8

    # Last 1 stock -> Sector 8 (Utilities)
    betas[19:20, 8] = 0.7

    # Factor standard deviations (daily)
    # These represent the volatility of each factor
    factor_std = np.array([
        0.012,  # Market: 1.2% daily (~19% annualized)
        0.015,  # Sector 1 (Tech): Higher volatility
        0.010,  # Sector 2 (Healthcare): Lower volatility
        0.013,  # Sector 3 (Financials)
        0.020,  # Sector 4 (Energy): Very high volatility
        0.011,  # Sector 5 (Consumer)
        0.012,  # Sector 6 (Industrials)
        0.014,  # Sector 7 (Materials)
        0.008,  # Sector 8 (Utilities): Lowest volatility
    ])

    # Factor correlation matrix
    # Must be positive definite and symmetric with 1s on diagonal
    factor_corr = np.array([
        # Mkt   S1    S2    S3    S4    S5    S6    S7    S8
        [1.00, 0.70, 0.60, 0.75, 0.65, 0.68, 0.72, 0.67, 0.55],  # Market
        [0.70, 1.00, 0.40, 0.50, 0.35, 0.45, 0.48, 0.42, 0.30],  # Tech
        [0.60, 0.40, 1.00, 0.35, 0.25, 0.50, 0.38, 0.32, 0.40],  # Healthcare
        [0.75, 0.50, 0.35, 1.00, 0.45, 0.52, 0.58, 0.48, 0.40],  # Financials
        [0.65, 0.35, 0.25, 0.45, 1.00, 0.40, 0.50, 0.65, 0.30],  # Energy
        [0.68, 0.45, 0.50, 0.52, 0.40, 1.00, 0.55, 0.45, 0.42],  # Consumer
        [0.72, 0.48, 0.38, 0.58, 0.50, 0.55, 1.00, 0.60, 0.45],  # Industrials
        [0.67, 0.42, 0.32, 0.48, 0.65, 0.45, 0.60, 1.00, 0.38],  # Materials
        [0.55, 0.30, 0.40, 0.40, 0.30, 0.42, 0.45, 0.38, 1.00],  # Utilities
    ])

    # Residual standard deviations (stock-specific risk)
    # Higher for individual stocks, lower for diversified portfolios
    residual_std = np.array([
        # Tech stocks (higher idiosyncratic risk)
        0.025, 0.028, 0.030, 0.027, 0.026,
        # Healthcare stocks
        0.020, 0.022, 0.021,
        # Financials
        0.024, 0.023, 0.025,
        # Energy stocks (high idiosyncratic risk)
        0.030, 0.032, 0.031,
        # Consumer
        0.018, 0.019,
        # Industrials
        0.021, 0.022,
        # Materials
        0.023,
        # Utilities (lowest idiosyncratic risk)
        0.015,
    ])

    # Initial stock prices (different starting prices)
    initial_prices = np.array([
        # Tech stocks
        150.0, 200.0, 80.0, 120.0, 95.0,
        # Healthcare stocks
        110.0, 85.0, 130.0,
        # Financials
        60.0, 45.0, 75.0,
        # Energy stocks
        55.0, 90.0, 70.0,
        # Consumer
        125.0, 140.0,
        # Industrials
        100.0, 95.0,
        # Materials
        65.0,
        # Utilities
        80.0,
    ])

    # Create model inputs
    inputs = FactorModelInputs(
        betas=betas,
        factor_std=factor_std,
        factor_corr=factor_corr,
        residual_std=residual_std,
        n_days=n_days,
        initial_prices=initial_prices
    )

    print("Running simulation...")
    results = simulate_equity_factor_model(inputs, random_seed=123)
    print("Simulation complete!")
    print()

    # Display results by sector
    sector_names = ['Tech', 'Healthcare', 'Financials', 'Energy', 'Consumer',
                   'Industrials', 'Materials', 'Utilities']
    sector_ranges = [(0, 5), (5, 8), (8, 11), (11, 14), (14, 16),
                     (16, 18), (18, 19), (19, 20)]

    print("Results by Sector:")
    print("-" * 80)
    for sector_name, (start, end) in zip(sector_names, sector_ranges):
        sector_accum = results.avg_price_accumulation[start:end]
        print(f"{sector_name:12s}: "
              f"Avg Accumulation = {np.mean(sector_accum):.4f}, "
              f"Range = [{np.min(sector_accum):.4f}, {np.max(sector_accum):.4f}]")
    print()

    # Overall statistics
    print("Overall Statistics:")
    print("-" * 80)
    total_return = (results.stock_prices[-1, :] / initial_prices - 1) * 100
    print(f"Average total return: {np.mean(total_return):.2f}%")
    print(f"Best performing stock: {np.argmax(total_return)}, Return: {np.max(total_return):.2f}%")
    print(f"Worst performing stock: {np.argmin(total_return)}, Return: {np.min(total_return):.2f}%")
    print()

    print("Average Price Accumulation Factors:")
    print(f"  Mean: {np.mean(results.avg_price_accumulation):.4f}")
    print(f"  Std:  {np.std(results.avg_price_accumulation):.4f}")
    print()

    # Calculate and display average prices
    avg_prices = initial_prices * results.avg_price_accumulation
    print("Sample Average Prices:")
    print("-" * 80)
    for i in [0, 5, 8, 11, 14, 16, 18, 19]:  # One from each sector
        print(f"  Stock {i:2d}: "
              f"Initial=${initial_prices[i]:6.2f}, "
              f"Final=${results.stock_prices[-1, i]:6.2f}, "
              f"Average=${avg_prices[i]:6.2f}")
    print()

    print("=" * 80)

    return inputs, results


def example_monte_carlo():
    """
    Example of running multiple simulations (Monte Carlo).
    """
    print("=" * 80)
    print("Monte Carlo Simulation Example")
    print("=" * 80)
    print()

    # Use a simple model
    n_stocks = 10
    n_factors = 9
    n_days = 252
    n_simulations = 100

    print(f"Running {n_simulations} simulations of {n_stocks} stocks over {n_days} days...")
    print()

    # Create simple model inputs
    betas = np.random.randn(n_stocks, n_factors) * 0.3
    betas[:, 0] += 1.0  # Market beta around 1.0

    factor_std = np.full(n_factors, 0.01)
    # Create a valid correlation matrix
    factor_corr = np.eye(n_factors) * 0.7 + np.ones((n_factors, n_factors)) * 0.3 / n_factors
    # Ensure diagonal is exactly 1.0
    np.fill_diagonal(factor_corr, 1.0)
    residual_std = np.full(n_stocks, 0.02)

    inputs = FactorModelInputs(
        betas=betas,
        factor_std=factor_std,
        factor_corr=factor_corr,
        residual_std=residual_std,
        n_days=n_days
    )

    # Run multiple simulations
    all_accumulations = []
    for i in range(n_simulations):
        results = simulate_equity_factor_model(inputs, random_seed=i)
        all_accumulations.append(results.avg_price_accumulation)

    all_accumulations = np.array(all_accumulations)  # Shape: (n_simulations, n_stocks)

    # Analyze results across simulations
    print("Monte Carlo Results:")
    print("-" * 80)
    print(f"Average accumulation factor across all simulations and stocks:")
    print(f"  Mean: {np.mean(all_accumulations):.4f}")
    print(f"  Std:  {np.std(all_accumulations):.4f}")
    print()

    # Show distribution for first stock
    stock_0_accum = all_accumulations[:, 0]
    print(f"Distribution of accumulation factor for Stock 0:")
    print(f"  Mean:   {np.mean(stock_0_accum):.4f}")
    print(f"  Std:    {np.std(stock_0_accum):.4f}")
    print(f"  5th %:  {np.percentile(stock_0_accum, 5):.4f}")
    print(f"  50th %: {np.percentile(stock_0_accum, 50):.4f}")
    print(f"  95th %: {np.percentile(stock_0_accum, 95):.4f}")
    print()

    print("=" * 80)


if __name__ == "__main__":
    # Run custom model example
    example_custom_model()
    print("\n\n")

    # Run Monte Carlo example
    example_monte_carlo()
