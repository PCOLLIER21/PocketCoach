"""
Monte Carlo simulation runner for risk_model.json

Runs 1000 simulations over 30 trading days for all stocks in the risk model.
"""

import json
import numpy as np
import time
from equity_factor_simulation import FactorModelInputs, simulate_equity_factor_model


def load_risk_model(filepath):
    """Load risk model from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    return {
        'stocks': data['stocks'],
        'betas': np.array(data['betas']),
        'factor_std': np.array(data['factor_std']),
        'factor_corr': np.array(data['factor_corr']),
        'residual_std': np.array(data['residual_std']),
        'initial_prices': np.array(data['initial_prices']),
        'factor_names': data['factor_names']
    }


def run_monte_carlo(risk_model, n_simulations=1000, n_days=30):
    """
    Run Monte Carlo simulations using the risk model.

    Args:
        risk_model: Dictionary containing model parameters
        n_simulations: Number of simulations to run
        n_days: Number of trading days per simulation

    Returns:
        Dictionary containing simulation results and statistics
    """
    print("=" * 80)
    print("Monte Carlo Equity Factor Model Simulation")
    print("=" * 80)
    print()

    # Extract model parameters
    betas = risk_model['betas']
    n_stocks = betas.shape[0]
    n_factors = betas.shape[1]

    print(f"Model Configuration:")
    print(f"  - Number of stocks: {n_stocks}")
    print(f"  - Number of factors: {n_factors}")
    print(f"  - Number of simulations: {n_simulations}")
    print(f"  - Simulation period: {n_days} trading days")
    print()

    # Create model inputs
    inputs = FactorModelInputs(
        betas=betas,
        factor_std=risk_model['factor_std'],
        factor_corr=risk_model['factor_corr'],
        residual_std=risk_model['residual_std'],
        n_days=n_days,
        initial_prices=risk_model['initial_prices']
    )

    # Storage for results across all simulations
    all_final_prices = np.zeros((n_simulations, n_stocks))
    all_avg_accumulation = np.zeros((n_simulations, n_stocks))
    all_total_returns = np.zeros((n_simulations, n_stocks))
    all_max_drawdowns = np.zeros((n_simulations, n_stocks))

    # Run simulations
    print("Running simulations...")
    start_time = time.time()

    for sim in range(n_simulations):
        # Progress indicator
        if (sim + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Completed {sim + 1}/{n_simulations} simulations ({elapsed:.1f}s elapsed)")

        # Run simulation with different random seed for each
        results = simulate_equity_factor_model(inputs, random_seed=sim)

        # Store results
        all_final_prices[sim, :] = results.stock_prices[-1, :]
        all_avg_accumulation[sim, :] = results.avg_price_accumulation
        all_total_returns[sim, :] = (results.stock_prices[-1, :] / inputs.initial_prices - 1) * 100

        # Calculate max drawdown for each stock
        for stock_idx in range(n_stocks):
            prices = results.stock_prices[:, stock_idx]
            cummax = np.maximum.accumulate(prices)
            drawdown = (prices - cummax) / cummax * 100
            all_max_drawdowns[sim, stock_idx] = np.min(drawdown)

    elapsed_time = time.time() - start_time
    print(f"  All simulations complete! ({elapsed_time:.1f}s total)")
    print()

    return {
        'final_prices': all_final_prices,
        'avg_accumulation': all_avg_accumulation,
        'total_returns': all_total_returns,
        'max_drawdowns': all_max_drawdowns,
        'elapsed_time': elapsed_time,
        'n_simulations': n_simulations,
        'n_days': n_days,
        'n_stocks': n_stocks
    }


def analyze_results(mc_results, risk_model):
    """Analyze and display Monte Carlo results."""
    stocks = risk_model['stocks']
    initial_prices = risk_model['initial_prices']

    print("=" * 80)
    print("Monte Carlo Simulation Results")
    print("=" * 80)
    print()

    # Overall statistics
    print("Overall Statistics:")
    print("-" * 80)
    print(f"Total simulations: {mc_results['n_simulations']}")
    print(f"Simulation period: {mc_results['n_days']} trading days")
    print(f"Number of stocks: {mc_results['n_stocks']}")
    print(f"Computation time: {mc_results['elapsed_time']:.2f} seconds")
    print()

    # Average price accumulation statistics
    avg_accum_mean = np.mean(mc_results['avg_accumulation'])
    avg_accum_std = np.std(mc_results['avg_accumulation'])

    print("Average Price Accumulation Factor (across all stocks and simulations):")
    print(f"  Mean: {avg_accum_mean:.4f}")
    print(f"  Std:  {avg_accum_std:.4f}")
    print()

    # Return statistics
    print("Total Returns Statistics (%):")
    print("-" * 80)
    returns_mean = np.mean(mc_results['total_returns'])
    returns_std = np.std(mc_results['total_returns'])
    returns_p5 = np.percentile(mc_results['total_returns'], 5)
    returns_p50 = np.percentile(mc_results['total_returns'], 50)
    returns_p95 = np.percentile(mc_results['total_returns'], 95)

    print(f"  Mean:        {returns_mean:8.2f}%")
    print(f"  Std:         {returns_std:8.2f}%")
    print(f"  5th %ile:    {returns_p5:8.2f}%")
    print(f"  50th %ile:   {returns_p50:8.2f}%")
    print(f"  95th %ile:   {returns_p95:8.2f}%")
    print(f"  Min:         {np.min(mc_results['total_returns']):8.2f}%")
    print(f"  Max:         {np.max(mc_results['total_returns']):8.2f}%")
    print()

    # Per-stock statistics
    print("Per-Stock Statistics:")
    print("-" * 80)
    print(f"{'Ticker':<8} {'Sector':<12} {'Init $':<10} {'Avg Return':<12} {'Std Return':<12} {'Avg Accum':<12} {'Max DD':<10}")
    print("-" * 80)

    for i, stock in enumerate(stocks):
        ticker = stock['ticker']
        sector = risk_model['factor_names'][stock['sector'] + 1]  # +1 because index 0 is Market
        init_price = initial_prices[i]

        avg_return = np.mean(mc_results['total_returns'][:, i])
        std_return = np.std(mc_results['total_returns'][:, i])
        avg_accum = np.mean(mc_results['avg_accumulation'][:, i])
        avg_max_dd = np.mean(mc_results['max_drawdowns'][:, i])

        print(f"{ticker:<8} {sector:<12} ${init_price:<9.2f} {avg_return:>10.2f}%  "
              f"{std_return:>10.2f}%  {avg_accum:>10.4f}  {avg_max_dd:>9.2f}%")

    print()

    # Sector-level analysis
    print("Sector-Level Analysis:")
    print("-" * 80)
    print(f"{'Sector':<15} {'Avg Return':<12} {'Std Return':<12} {'Avg Accum':<12}")
    print("-" * 80)

    sector_stats = {}
    for i, stock in enumerate(stocks):
        sector_idx = stock['sector']
        sector_name = risk_model['factor_names'][sector_idx + 1]

        if sector_name not in sector_stats:
            sector_stats[sector_name] = {
                'returns': [],
                'accumulations': []
            }

        sector_stats[sector_name]['returns'].extend(mc_results['total_returns'][:, i])
        sector_stats[sector_name]['accumulations'].extend(mc_results['avg_accumulation'][:, i])

    for sector_name in sorted(sector_stats.keys()):
        returns = np.array(sector_stats[sector_name]['returns'])
        accums = np.array(sector_stats[sector_name]['accumulations'])

        avg_ret = np.mean(returns)
        std_ret = np.std(returns)
        avg_acc = np.mean(accums)

        print(f"{sector_name:<15} {avg_ret:>10.2f}%  {std_ret:>10.2f}%  {avg_acc:>10.4f}")

    print()

    # Distribution analysis for a sample stock
    sample_stock_idx = 0
    sample_ticker = stocks[sample_stock_idx]['ticker']

    print(f"Distribution Analysis for {sample_ticker}:")
    print("-" * 80)
    stock_returns = mc_results['total_returns'][:, sample_stock_idx]

    print(f"  Mean:        {np.mean(stock_returns):8.2f}%")
    print(f"  Std:         {np.std(stock_returns):8.2f}%")
    print(f"  1st %ile:    {np.percentile(stock_returns, 1):8.2f}%")
    print(f"  5th %ile:    {np.percentile(stock_returns, 5):8.2f}%")
    print(f"  25th %ile:   {np.percentile(stock_returns, 25):8.2f}%")
    print(f"  50th %ile:   {np.percentile(stock_returns, 50):8.2f}%")
    print(f"  75th %ile:   {np.percentile(stock_returns, 75):8.2f}%")
    print(f"  95th %ile:   {np.percentile(stock_returns, 95):8.2f}%")
    print(f"  99th %ile:   {np.percentile(stock_returns, 99):8.2f}%")
    print()

    # Probability of positive/negative returns
    print("Return Probabilities (across all stocks and simulations):")
    print("-" * 80)
    all_returns = mc_results['total_returns'].flatten()
    pct_positive = np.sum(all_returns > 0) / len(all_returns) * 100
    pct_negative = np.sum(all_returns < 0) / len(all_returns) * 100
    pct_neutral = np.sum(all_returns == 0) / len(all_returns) * 100

    print(f"  Probability of positive return: {pct_positive:.2f}%")
    print(f"  Probability of negative return: {pct_negative:.2f}%")
    print(f"  Probability of zero return:     {pct_neutral:.2f}%")
    print()

    # Extreme scenarios
    print("Extreme Scenarios:")
    print("-" * 80)
    best_sim_idx = np.argmax(np.mean(mc_results['total_returns'], axis=1))
    worst_sim_idx = np.argmin(np.mean(mc_results['total_returns'], axis=1))

    print(f"  Best simulation #{best_sim_idx}: Avg return = {np.mean(mc_results['total_returns'][best_sim_idx, :]):.2f}%")
    print(f"  Worst simulation #{worst_sim_idx}: Avg return = {np.mean(mc_results['total_returns'][worst_sim_idx, :]):.2f}%")
    print()

    print("=" * 80)


def save_results(mc_results, risk_model, output_file='monte_carlo_results.json'):
    """Save Monte Carlo results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    output_data = {
        'simulation_config': {
            'n_simulations': int(mc_results['n_simulations']),
            'n_days': int(mc_results['n_days']),
            'n_stocks': int(mc_results['n_stocks']),
            'elapsed_time': float(mc_results['elapsed_time'])
        },
        'summary_statistics': {
            'avg_accumulation': {
                'mean': float(np.mean(mc_results['avg_accumulation'])),
                'std': float(np.std(mc_results['avg_accumulation'])),
                'min': float(np.min(mc_results['avg_accumulation'])),
                'max': float(np.max(mc_results['avg_accumulation']))
            },
            'total_returns_pct': {
                'mean': float(np.mean(mc_results['total_returns'])),
                'std': float(np.std(mc_results['total_returns'])),
                'percentile_5': float(np.percentile(mc_results['total_returns'], 5)),
                'percentile_50': float(np.percentile(mc_results['total_returns'], 50)),
                'percentile_95': float(np.percentile(mc_results['total_returns'], 95)),
                'min': float(np.min(mc_results['total_returns'])),
                'max': float(np.max(mc_results['total_returns']))
            }
        },
        'per_stock_statistics': []
    }

    # Per-stock statistics
    for i, stock in enumerate(risk_model['stocks']):
        stock_data = {
            'ticker': stock['ticker'],
            'name': stock['name'],
            'sector': risk_model['factor_names'][stock['sector'] + 1],
            'initial_price': float(risk_model['initial_prices'][i]),
            'avg_return_pct': float(np.mean(mc_results['total_returns'][:, i])),
            'std_return_pct': float(np.std(mc_results['total_returns'][:, i])),
            'avg_accumulation': float(np.mean(mc_results['avg_accumulation'][:, i])),
            'avg_max_drawdown_pct': float(np.mean(mc_results['max_drawdowns'][:, i])),
            'return_distribution': {
                'percentile_1': float(np.percentile(mc_results['total_returns'][:, i], 1)),
                'percentile_5': float(np.percentile(mc_results['total_returns'][:, i], 5)),
                'percentile_25': float(np.percentile(mc_results['total_returns'][:, i], 25)),
                'percentile_50': float(np.percentile(mc_results['total_returns'][:, i], 50)),
                'percentile_75': float(np.percentile(mc_results['total_returns'][:, i], 75)),
                'percentile_95': float(np.percentile(mc_results['total_returns'][:, i], 95)),
                'percentile_99': float(np.percentile(mc_results['total_returns'][:, i], 99))
            }
        }
        output_data['per_stock_statistics'].append(stock_data)

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_file}")
    print()


if __name__ == "__main__":
    # Load risk model
    print("Loading risk model from risk_model.json...")
    risk_model = load_risk_model('risk_model.json')
    print(f"Loaded model with {len(risk_model['stocks'])} stocks")
    print()

    # Run Monte Carlo simulation
    mc_results = run_monte_carlo(
        risk_model=risk_model,
        n_simulations=1000,
        n_days=30
    )

    # Analyze and display results
    analyze_results(mc_results, risk_model)

    # Save results to file
    save_results(mc_results, risk_model, output_file='monte_carlo_results.json')
