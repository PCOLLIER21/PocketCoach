"""
Run equity factor model simulation using parameters from risk_model.json

This script:
1. Loads the risk model from risk_model.json
2. Extracts betas, factor std dev, factor correlations, and residual std dev
3. Runs the equity factor simulation
4. Displays and saves the results
"""

import json
import numpy as np
from equity_factor_simulation import (
    FactorModelInputs,
    simulate_equity_factor_model
)


def load_risk_model(json_path='risk_model.json'):
    """
    Load and parse the risk model from JSON file.

    Args:
        json_path: Path to the risk_model.json file

    Returns:
        Tuple of (betas, factor_names, factor_std, factor_corr, residual_std, stock_tickers)
    """
    print(f"Loading risk model from {json_path}...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    results = data['results']

    # Extract betas
    betas_data = results['betas']['data']
    n_stocks = len(betas_data)

    # Get factor names from the first stock (excluding 'bloomberg_ticker')
    factor_names = [key for key in betas_data[0].keys() if key != 'bloomberg_ticker']
    n_factors = len(factor_names)

    print(f"  Found {n_stocks} stocks")
    print(f"  Found {n_factors} factors")

    # Build betas matrix
    betas = np.zeros((n_stocks, n_factors))
    stock_tickers = []

    for i, stock_data in enumerate(betas_data):
        stock_tickers.append(stock_data['bloomberg_ticker'])
        for j, factor_name in enumerate(factor_names):
            betas[i, j] = stock_data[factor_name]

    # Extract factor standard deviations
    factor_std_data = results['factor_std_dev']['data']
    factor_std = np.zeros(n_factors)
    for item in factor_std_data:
        factor_name = item['index']
        idx = factor_names.index(factor_name)
        factor_std[idx] = item['factor_std_dev']

    # Extract factor correlation matrix
    factor_corr_data = results['factors_corr']['data']
    factor_corr = np.zeros((n_factors, n_factors))

    for row_data in factor_corr_data:
        factor_name_i = row_data['index']
        # Find the index of this factor
        idx_i = factor_names.index(factor_name_i)
        for j, factor_name_j in enumerate(factor_names):
            factor_corr[idx_i, j] = row_data[factor_name_j]

    # Extract residual standard deviations
    residual_std_data = results['residuals_std_dev']['data']
    residual_std = np.zeros(n_stocks)

    for resid_data in residual_std_data:
        ticker = resid_data['bloomberg_ticker']
        # Find the index of this stock
        idx = stock_tickers.index(ticker)
        residual_std[idx] = resid_data['residuals_std_dev']

    print(f"  Loaded betas: {betas.shape}")
    print(f"  Loaded factor std dev: {factor_std.shape}")
    print(f"  Loaded factor correlations: {factor_corr.shape}")
    print(f"  Loaded residual std dev: {residual_std.shape}")
    print()

    return betas, factor_names, factor_std, factor_corr, residual_std, stock_tickers


def run_simulation(n_days=252, random_seed=42):
    """
    Run the equity factor model simulation using the risk model.

    Args:
        n_days: Number of trading days to simulate (default: 252 = 1 year)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (inputs, results, stock_tickers, factor_names)
    """
    # Load the risk model
    betas, factor_names, factor_std, factor_corr, residual_std, stock_tickers = load_risk_model()

    # Create model inputs
    # Note: The risk model provides annualized std devs, but simulation expects daily
    # If they're already daily, use as-is; if annual, divide by sqrt(252)
    # Let's check the magnitude to determine which it is
    print("Factor standard deviations:")
    print(f"  Mean: {np.mean(factor_std):.6f}")
    print(f"  Range: [{np.min(factor_std):.6f}, {np.max(factor_std):.6f}]")
    print()

    # If values are > 0.1, they're likely annual (10%+), so convert to daily
    if np.mean(factor_std) > 0.1:
        print("Converting factor std dev from annual to daily (dividing by sqrt(252))...")
        factor_std = factor_std / np.sqrt(252)

    if np.mean(residual_std) > 0.1:
        print("Converting residual std dev from annual to daily (dividing by sqrt(252))...")
        residual_std = residual_std / np.sqrt(252)

    print()
    print(f"Adjusted factor std dev - Mean: {np.mean(factor_std):.6f}, Range: [{np.min(factor_std):.6f}, {np.max(factor_std):.6f}]")
    print(f"Adjusted residual std dev - Mean: {np.mean(residual_std):.6f}, Range: [{np.min(residual_std):.6f}, {np.max(residual_std):.6f}]")
    print()

    # Create inputs
    inputs = FactorModelInputs(
        betas=betas,
        factor_std=factor_std,
        factor_corr=factor_corr,
        residual_std=residual_std,
        n_days=n_days,
        initial_prices=np.full(len(stock_tickers), 100.0)
    )

    print(f"Running simulation for {n_days} days...")
    results = simulate_equity_factor_model(inputs, random_seed=random_seed)
    print("Simulation complete!")
    print()

    return inputs, results, stock_tickers, factor_names


def display_results(inputs, results, stock_tickers, factor_names):
    """
    Display simulation results.
    """
    print("=" * 80)
    print("SIMULATION RESULTS")
    print("=" * 80)
    print()

    # Overall statistics
    print("Overall Statistics:")
    print("-" * 80)
    print(f"Number of stocks: {inputs.n_stocks}")
    print(f"Number of factors: {inputs.n_factors}")
    print(f"Simulation days: {inputs.n_days}")
    print()

    # Average price accumulation
    print("Average Price Accumulation Factors:")
    print("-" * 80)
    print(f"  Mean: {np.mean(results.avg_price_accumulation):.4f}")
    print(f"  Std:  {np.std(results.avg_price_accumulation):.4f}")
    print(f"  Min:  {np.min(results.avg_price_accumulation):.4f}")
    print(f"  Max:  {np.max(results.avg_price_accumulation):.4f}")
    print()

    # Total returns
    total_returns = (results.stock_prices[-1, :] / inputs.initial_prices - 1) * 100
    print("Total Returns (%):")
    print("-" * 80)
    print(f"  Mean:   {np.mean(total_returns):.2f}%")
    print(f"  Median: {np.median(total_returns):.2f}%")
    print(f"  Std:    {np.std(total_returns):.2f}%")
    print(f"  Min:    {np.min(total_returns):.2f}%")
    print(f"  Max:    {np.max(total_returns):.2f}%")
    print()

    # Top 10 performers
    print("Top 10 Performing Stocks:")
    print("-" * 80)
    top_indices = np.argsort(total_returns)[-10:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        ticker = stock_tickers[idx]
        ret = total_returns[idx]
        accum = results.avg_price_accumulation[idx]
        print(f"  {rank:2d}. {ticker:15s}: Return={ret:7.2f}%, AvgAccum={accum:.4f}")
    print()

    # Bottom 10 performers
    print("Bottom 10 Performing Stocks:")
    print("-" * 80)
    bottom_indices = np.argsort(total_returns)[:10]
    for rank, idx in enumerate(bottom_indices, 1):
        ticker = stock_tickers[idx]
        ret = total_returns[idx]
        accum = results.avg_price_accumulation[idx]
        print(f"  {rank:2d}. {ticker:15s}: Return={ret:7.2f}%, AvgAccum={accum:.4f}")
    print()

    # Factor statistics
    print("Factor Return Statistics (daily):")
    print("-" * 80)
    for i, factor_name in enumerate(factor_names):
        mean_ret = np.mean(results.factor_returns[:, i])
        std_ret = np.std(results.factor_returns[:, i])
        annualized_ret = mean_ret * 252  # Annualize
        annualized_std = std_ret * np.sqrt(252)
        print(f"  {factor_name:30s}: Mean={mean_ret:8.5f} ({annualized_ret:7.2%} ann), "
              f"Std={std_ret:8.5f} ({annualized_std:7.2%} ann)")
    print()

    # Stock return statistics
    print("Stock Return Statistics (daily):")
    print("-" * 80)
    stock_mean = np.mean(results.stock_returns, axis=0)
    stock_std = np.std(results.stock_returns, axis=0)
    print(f"  Mean of means: {np.mean(stock_mean):.5f} ({np.mean(stock_mean)*252:.2%} annualized)")
    print(f"  Mean of stds:  {np.mean(stock_std):.5f} ({np.mean(stock_std)*np.sqrt(252):.2%} annualized)")
    print(f"  Min return:    {np.min(results.stock_returns):.5f}")
    print(f"  Max return:    {np.max(results.stock_returns):.5f}")
    print()

    print("=" * 80)


def save_results(results, stock_tickers, output_file='simulation_results.npz'):
    """
    Save simulation results to file.
    """
    print(f"Saving results to {output_file}...")

    np.savez(
        output_file,
        stock_returns=results.stock_returns,
        factor_returns=results.factor_returns,
        residual_returns=results.residual_returns,
        stock_prices=results.stock_prices,
        avg_price_accumulation=results.avg_price_accumulation,
        stock_tickers=stock_tickers
    )

    print(f"Results saved to {output_file}")
    print()


if __name__ == "__main__":
    # Configuration
    N_DAYS = 252  # One trading year
    RANDOM_SEED = 42

    print("=" * 80)
    print("EQUITY FACTOR MODEL SIMULATION - RISK MODEL")
    print("=" * 80)
    print()

    # Run simulation
    inputs, results, stock_tickers, factor_names = run_simulation(
        n_days=N_DAYS,
        random_seed=RANDOM_SEED
    )

    # Display results
    display_results(inputs, results, stock_tickers, factor_names)

    # Save results
    save_results(results, stock_tickers)

    print("Done!")
