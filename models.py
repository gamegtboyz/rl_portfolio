from utils import date_index, rf_rate, get_rf_rate
import pandas as pd
import numpy as np
import quantstats as qs
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, unwrap_vec_normalize
from portfolio_env import PortfolioEnv
import torch as th
from typing import Tuple
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import BaseCallback
import os

def split_build_normalize_env(price_data:pd.DataFrame, port_initial_date:str, lookback_period:int):
    ''' Split, build, and normalize the environment '''
    # split the data into train and test sets
    port_initial_index = date_index(price_data, port_initial_date)
    train_data = price_data.iloc[:port_initial_index]
    test_data = price_data.iloc[port_initial_index - lookback_period:]

    # build the normalized environments
    train_env = DummyVecEnv([lambda: PortfolioEnv(price_data=train_data,
                                                  lookback_period=lookback_period,
                                                  transaction_cost=0.0005,
                                                  risk_aversion=0.01)])
    train_env = VecNormalize(train_env)

    test_env = DummyVecEnv([lambda: PortfolioEnv(price_data=test_data,
                                                 lookback_period=lookback_period,
                                                 transaction_cost=0.0015)])
    test_env = VecNormalize(test_env, training=False, norm_reward=False)
    
    return train_env, test_env

def rolling_split_build_normalize_env(price_data: pd.DataFrame, 
                                      train_start_date: str, 
                                      train_end_date: str,
                                      test_end_date: str,
                                      lookback_period: int,
                                      step_years: int = 1,
                                      initial_capital: float = 1000000):
    """
    Rolling window train/test split for robustness testing.
    
    This function generates rolling train/test splits that move forward by `step_years` 
    each iteration. For example:
    - Train: 2010-2017, Test: 2018
    - Train: 2011-2018, Test: 2019
    - ... continues until test period reaches test_end_date
    
    Args:
        price_data (pd.DataFrame): Full price data with DatetimeIndex
        train_start_date (str): Initial training start date (e.g., '2010-01-01')
        train_end_date (str): Initial training end date (e.g., '2017-12-31')
        test_end_date (str): Final testing end date (e.g., '2024-12-31')
        lookback_period (int): Lookback period for the environment
        step_years (int): Number of years to roll forward each iteration (default: 1)
        initial_capital (float): Initial capital for first iteration (default: 1000000)
    
    Yields:
        dict: Contains:
            - 'train_env': VecNormalize wrapped training environment
            - 'test_env': VecNormalize wrapped testing environment
            - 'train_start_date': Training start date for this split
            - 'train_end_date': Training end date for this split
            - 'test_start_date': Testing start date for this split
            - 'test_end_date': Testing end date for this split
            - 'iteration': Current iteration number
    """
    from dateutil.relativedelta import relativedelta
    
    # Convert string dates to pandas Timestamps
    train_start = pd.to_datetime(train_start_date)
    train_end = pd.to_datetime(train_end_date)
    test_end = pd.to_datetime(test_end_date)
    
    # Calculate training window size in years
    train_window_years = (train_end.year - train_start.year) + \
                        (train_end.month - train_start.month) / 12
    
    iteration = 0
    current_capital = initial_capital
    
    while True:
        # Get test period start (day after training ends)
        test_start = train_end + pd.Timedelta(days=1)
        
        # Get test period end (1 year after test start by default, or custom period)
        test_period_end = test_start + relativedelta(years=step_years) - pd.Timedelta(days=1)
        
        # Stop if test period exceeds desired end date
        if test_period_end > test_end:
            test_period_end = test_end
        
        # Find indices in price data
        train_start_idx = date_index(price_data, train_start.strftime('%Y-%m-%d'))
        train_end_idx = date_index(price_data, train_end.strftime('%Y-%m-%d'))
        test_start_idx = date_index(price_data, test_start.strftime('%Y-%m-%d'))
        test_end_idx = date_index(price_data, test_period_end.strftime('%Y-%m-%d'))
        
        # Extract train and test data
        train_data = price_data.iloc[train_start_idx:train_end_idx + 1]
        test_data = price_data.iloc[max(0, test_start_idx - lookback_period):test_end_idx + 1]
        
        # Skip if test data is empty or if we have too little test data for meaningful evaluation
        if len(test_data) == 0:
            print(f">>> Iteration {iteration}: No test data available. Stopping rolling test.")
            break
        
        # Check if we have at least lookback_period + 1 trading days of test data
        test_data_length = len(test_data)
        if test_data_length <= 21:  # Less than a month of test data
            print(f">>> Iteration {iteration}: Insufficient test data ({test_data_length} days). Stopping rolling test.")
            break
        
        # Build normalized train environment
        train_env = DummyVecEnv([lambda data=train_data, cap=current_capital: PortfolioEnv(
            price_data=data,
            lookback_period=lookback_period,
            initial_capital=cap,
            transaction_cost=0.0005,
            risk_aversion=0.01)])
        train_env = VecNormalize(train_env)
        
        # Build normalized test environment
        test_env = DummyVecEnv([lambda data=test_data, cap=current_capital: PortfolioEnv(
            price_data=data,
            lookback_period=lookback_period,
            initial_capital=cap,
            transaction_cost=0.0015)])
        test_env = VecNormalize(test_env, training=False, norm_reward=False)
        
        yield {
            'train_env': train_env,
            'test_env': test_env,
            'train_start_date': train_start.strftime('%Y-%m-%d'),
            'train_end_date': train_end.strftime('%Y-%m-%d'),
            'test_start_date': test_start.strftime('%Y-%m-%d'),
            'test_end_date': test_period_end.strftime('%Y-%m-%d'),
            'iteration': iteration,
            'initial_capital': current_capital
        }
        
        # Roll forward by step_years
        train_start = train_start + relativedelta(years=step_years)
        train_end = train_end + relativedelta(years=step_years)
        iteration += 1
        
        # Stop if test period has reached the end date (or beyond available data)
        data_end_date = pd.to_datetime(price_data.index[-1])
        if test_period_end >= test_end or test_period_end >= data_end_date:
            print(f">>> Iteration {iteration}: Reached end of test period or data (test_period_end={test_period_end.strftime('%Y-%m-%d')}, data_end={data_end_date.strftime('%Y-%m-%d')}). Stopping rolling test.")
            break

def conduct_rolling_robustness_test(model_class, 
                                   price_data: pd.DataFrame, 
                                   train_start_date: str,
                                   train_end_date: str, 
                                   test_end_date: str,
                                   lookback_period: int = 21,
                                   step_years: int = 1,
                                   total_timesteps: int = 100000,
                                   eval_episodes: int = 10,
                                   initial_capital: float = 1000000,
                                   **model_kwargs):
    """
    Conduct rolling robustness test across multiple train/test windows.
    
    This function trains a model on rolling windows of historical data and evaluates
    on subsequent test periods. Portfolio values carry forward across iterations:
    - Iteration 0 starts with initial_capital
    - Iteration 1+ starts with the final portfolio value from the previous iteration
    
    Args:
        model_class: SB3 Model class (A2C, PPO, DDPG, SAC)
        price_data (pd.DataFrame): Full price data with DatetimeIndex
        train_start_date (str): Initial training start date
        train_end_date (str): Initial training end date
        test_end_date (str): Final testing end date
        lookback_period (int): Lookback period (default: 21)
        step_years (int): Years to roll forward (default: 1)
        total_timesteps (int): Timesteps per training iteration
        eval_episodes (int): Episodes for evaluation (default: 10)
        initial_capital (float): Initial capital for first iteration (default: 1000000)
        **model_kwargs: Additional arguments for model_class constructor
    
    Returns:
        dict: Aggregated results containing:
            - 'all_results': List of dicts with per-iteration metrics
            - 'summary_stats': Aggregated statistics across all iterations
            - 'hyperparameters': Dict of all hyperparameters used
            - 'portfolio_values_all': Dict mapping iterations to portfolio value arrays
    """
    from copy import deepcopy
    
    all_results = []
    portfolio_values_all = {}
    tracking_data_all = []  # NEW: Collect tracking data from test environments
    current_capital = initial_capital
    
    # Store hyperparameters for logging
    hyperparameters = {
        'model_class': model_class.__name__,
        'train_start_date': train_start_date,
        'train_end_date': train_end_date,
        'test_end_date': test_end_date,
        'lookback_period': lookback_period,
        'step_years': step_years,
        'total_timesteps': total_timesteps,
        'eval_episodes': eval_episodes,
        'initial_capital': initial_capital,
        **model_kwargs  # Include all model-specific hyperparameters
    }
    
    # Iterate through rolling windows
    for split_data in rolling_split_build_normalize_env(
        price_data=price_data,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_end_date=test_end_date,
        lookback_period=lookback_period,
        step_years=step_years,
        initial_capital=current_capital
    ):
        iteration = split_data['iteration']
        train_env = split_data['train_env']
        test_env = split_data['test_env']
        iteration_start_capital = split_data['initial_capital']
        
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"Training: {split_data['train_start_date']} to {split_data['train_end_date']}")
        print(f"Testing: {split_data['test_start_date']} to {split_data['test_end_date']}")
        print(f"Starting Capital: ${iteration_start_capital:,.2f}")
        print(f"{'='*70}")
        
        # Train model
        model = model_class('MlpPolicy', train_env, **model_kwargs)
        model.learn(total_timesteps=total_timesteps)
        
        # Prepare cumulative adjustment if not first iteration
        cumulative_adjustment = None
        if iteration > 0:
            # Carry forward the final value from previous iteration
            cumulative_adjustment = {
                'prev_final_value': current_capital,
                'current_initial_value': iteration_start_capital
            }
        
        # Evaluate on test set with cumulative adjustment
        eval_results = evaluate_model_sb3(
            model, test_env, 
            num_episodes=eval_episodes,
            cumulative_adjustment=cumulative_adjustment
        )
        
        # Get final portfolio value for next iteration (from cumulative-adjusted values)
        final_portfolio_value = eval_results.get('final_portfolio_value', eval_results['portfolio_values'][-1])
        current_capital = final_portfolio_value
        
        # Store results with metadata
        iteration_results = {
            'iteration': iteration,
            'train_start_date': split_data['train_start_date'],
            'train_end_date': split_data['train_end_date'],
            'test_start_date': split_data['test_start_date'],
            'test_end_date': split_data['test_end_date'],
            'starting_capital': iteration_start_capital,
            'final_portfolio_value': final_portfolio_value,
            **eval_results  # Unpack evaluation metrics
        }
        
        all_results.append(iteration_results)
        
        # Get the unwrapped environment for accessing tracking data
        unwrapped_env = test_env.venv.envs[0]  # Get first env from DummyVecEnv
        
        # Store portfolio values and tracking data separately for aggregation
        portfolio_values_all[iteration] = {
            'dates': eval_results['portfolio_df']['date'].tolist(),
            'portfolio_values': eval_results['portfolio_values'].tolist(),
            'weights': np.array(unwrapped_env.weights_history),
            'turnover': np.array(unwrapped_env.turnover_history),
            'transaction_costs': np.array(unwrapped_env.transaction_costs_history)
        }
        
        # Get daily dataframes from eval_results (now have full daily data)
        weights_df = eval_results['weights_df'].copy()
        weights_df['iteration'] = iteration
        weights_df['train_start_date'] = split_data['train_start_date']
        weights_df['train_end_date'] = split_data['train_end_date']
        weights_df['test_start_date'] = split_data['test_start_date']
        weights_df['test_end_date'] = split_data['test_end_date']
        
        transaction_df = eval_results['transaction_df'].copy()
        transaction_df['iteration'] = iteration
        transaction_df['train_start_date'] = split_data['train_start_date']
        transaction_df['train_end_date'] = split_data['train_end_date']
        transaction_df['test_start_date'] = split_data['test_start_date']
        transaction_df['test_end_date'] = split_data['test_end_date']
        
        turnover_df = eval_results['turnover_df'].copy()
        turnover_df['iteration'] = iteration
        turnover_df['train_start_date'] = split_data['train_start_date']
        turnover_df['train_end_date'] = split_data['train_end_date']
        turnover_df['test_start_date'] = split_data['test_start_date']
        turnover_df['test_end_date'] = split_data['test_end_date']
        
        tracking_data_all.append({
            'iteration': iteration,
            'weights_df': weights_df,
            'transaction_df': transaction_df,
            'turnover_df': turnover_df
        })
        
        # Print iteration summary
        print(f"Annualized Return: {eval_results['annualized_return']:.4f}")
        print(f"Sharpe Ratio: {eval_results['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {eval_results['max_drawdown']:.4f}")
        print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
        cumulative_return = ((final_portfolio_value - initial_capital) / initial_capital) * 100
        print(f"Cumulative Return from Initial Capital: {cumulative_return:.2f}%")
    
    # Calculate summary statistics
    returns = [r['annualized_return'] for r in all_results]
    sharpe_ratios = [r['sharpe_ratio'] for r in all_results]
    drawdowns = [r['max_drawdown'] for r in all_results]
    volatilities = [r['annualized_volatility'] for r in all_results]
    sortino_ratios = [r['sortino_ratio'] for r in all_results]
    
    summary_stats = {
        'num_iterations': len(all_results),
        'mean_annualized_return': np.mean(returns),
        'std_annualized_return': np.std(returns),
        'mean_sharpe_ratio': np.mean(sharpe_ratios),
        'std_sharpe_ratio': np.std(sharpe_ratios),
        'mean_max_drawdown': np.mean(drawdowns),
        'std_max_drawdown': np.std(drawdowns),
        'mean_volatility': np.mean(volatilities),
        'std_volatility': np.std(volatilities),
        'mean_sortino_ratio': np.mean(sortino_ratios),
        'std_sortino_ratio': np.std(sortino_ratios),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'min_sharpe': np.min(sharpe_ratios),
        'max_sharpe': np.max(sharpe_ratios),
        'final_portfolio_value': current_capital,
        'total_return': ((current_capital - initial_capital) / initial_capital) * 100,
    }
    
    return {
        'all_results': all_results,
        'summary_stats': summary_stats,
        'hyperparameters': hyperparameters,
        'portfolio_values_all': portfolio_values_all,
        'tracking_data': tracking_data_all  # NEW: Tracking data per iteration
    }

def export_rolling_results_to_csv(results: dict, output_filepath: str):
    """
    Export rolling robustness test results to CSV.
    
    Args:
        results (dict): Output from conduct_rolling_robustness_test()
        output_filepath (str): Path to save results CSV
    """
    # Convert results to DataFrame
    df = pd.DataFrame(results['all_results'])
    
    # Select and reorder key columns
    key_cols = [
        'iteration', 'train_start_date', 'train_end_date', 
        'test_start_date', 'test_end_date', 
        'annualized_return', 'annualized_volatility', 'sharpe_ratio',
        'sortino_ratio', 'max_drawdown'
    ]
    
    # Only include columns that exist in the dataframe
    cols_to_keep = [col for col in key_cols if col in df.columns]
    df = df[cols_to_keep]
    
    # Save to CSV
    df.to_csv(output_filepath, index=False)
    print(f"\nResults exported to {output_filepath}")
    
    return df

def export_hyperparameters_to_csv(results: dict, output_filepath: str):
    """
    Export hyperparameters and summary statistics to CSV.
    
    Args:
        results (dict): Output from conduct_rolling_robustness_test()
        output_filepath (str): Path to save hyperparameters CSV
    """
    hyperparams = results['hyperparameters'].copy()
    summary = results['summary_stats'].copy()
    
    # Combine hyperparameters and summary stats
    combined = {**hyperparams, **summary}
    
    # Create a DataFrame with single row
    df = pd.DataFrame([combined])
    
    # Save to CSV
    df.to_csv(output_filepath, index=False)
    print(f"Hyperparameters and summary stats exported to {output_filepath}")
    
    return df

def export_portfolio_values_to_csv(results: dict, output_filepath: str):
    """
    Export appended portfolio values from all iterations to CSV.
    Each iteration's portfolio values are stored with iteration metadata.
    
    Args:
        results (dict): Output from conduct_rolling_robustness_test()
        output_filepath (str): Path to save portfolio values CSV
    """
    all_portfolio_data = []
    
    # Iterate through all iterations and their portfolio values
    for iteration, pv_data in results['portfolio_values_all'].items():
        dates = pv_data['dates']
        portfolio_values = pv_data['portfolio_values']
        
        # Get iteration metadata from all_results
        iter_metadata = results['all_results'][iteration]
        
        # Create rows for each date/value pair
        for date, pv in zip(dates, portfolio_values):
            row = {
                'iteration': iteration,
                'train_start_date': iter_metadata['train_start_date'],
                'train_end_date': iter_metadata['train_end_date'],
                'test_start_date': iter_metadata['test_start_date'],
                'test_end_date': iter_metadata['test_end_date'],
                'date': date,
                'portfolio_value': pv
            }
            all_portfolio_data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_portfolio_data)
    
    # Save to CSV
    df.to_csv(output_filepath, index=False)
    print(f"Portfolio values from all iterations exported to {output_filepath}")
    
    return df

def export_weights_to_csv(results: dict, output_filepath: str):
    """
    Export daily portfolio weights from all iterations to CSV.
    Each weight entry is stored with iteration metadata and asset columns.
    
    Args:
        results (dict): Output from conduct_rolling_robustness_test()
        output_filepath (str): Path to save weights CSV
    """
    all_weights_dfs = []
    
    # Iterate through tracking data from each iteration
    for tracking_data in results.get('tracking_data', []):
        weights_df = tracking_data.get('weights_df')
        if weights_df is not None:
            all_weights_dfs.append(weights_df)
    
    # Concatenate all iterations into single dataframe
    if all_weights_dfs:
        df = pd.concat(all_weights_dfs, ignore_index=True)
    else:
        df = pd.DataFrame()
    
    # Save to CSV
    df.to_csv(output_filepath, index=False)
    print(f"Daily weights from all iterations exported to {output_filepath}")
    
    return df

def export_turnover_to_csv(results: dict, output_filepath: str):
    """
    Export daily turnover from all iterations to CSV.
    Each turnover entry is stored with iteration metadata.
    
    Args:
        results (dict): Output from conduct_rolling_robustness_test()
        output_filepath (str): Path to save turnover CSV
    """
    all_turnover_dfs = []
    
    # Iterate through tracking data from each iteration
    for tracking_data in results.get('tracking_data', []):
        turnover_df = tracking_data.get('turnover_df')
        if turnover_df is not None:
            all_turnover_dfs.append(turnover_df)
    
    # Concatenate all iterations into single dataframe
    if all_turnover_dfs:
        df = pd.concat(all_turnover_dfs, ignore_index=True)
    else:
        df = pd.DataFrame()
    
    # Save to CSV
    df.to_csv(output_filepath, index=False)
    print(f"Daily turnover from all iterations exported to {output_filepath}")
    
    return df

def export_transaction_costs_to_csv(results: dict, output_filepath: str):
    """
    Export daily transaction costs from all iterations to CSV.
    Each transaction cost entry is stored with iteration metadata.
    
    Args:
        results (dict): Output from conduct_rolling_robustness_test()
        output_filepath (str): Path to save transaction costs CSV
    """
    all_costs_dfs = []
    
    # Iterate through tracking data from each iteration
    for tracking_data in results.get('tracking_data', []):
        transaction_df = tracking_data.get('transaction_df')
        if transaction_df is not None:
            all_costs_dfs.append(transaction_df)
    
    # Concatenate all iterations into single dataframe
    if all_costs_dfs:
        df = pd.concat(all_costs_dfs, ignore_index=True)
    else:
        df = pd.DataFrame()
    
    # Save to CSV
    df.to_csv(output_filepath, index=False)
    print(f"Daily transaction costs from all iterations exported to {output_filepath}")
    
    return df

def evaluate_model_sb3(model, env, num_episodes=10, cumulative_adjustment=None):
    """
    Clean redesign: Evaluate a Stable Baselines3 model on the environment.
    Supports cumulative portfolio value tracking across rolling windows.
    
    Args:
        model: Loaded SB3 model (A2C, PPO, DDPG, SAC)
        env: VecNormalize wrapped environment
        num_episodes (int): Number of evaluation episodes
        cumulative_adjustment (dict, optional): Dict with keys:
            - 'prev_final_value' (float): Final portfolio value from previous iteration
            - 'current_initial_value' (float): Initial capital for current iteration
            Enables cumulative tracking across rolling windows
    
    Returns:
        dict with metrics from LAST episode:
            - annualized_return (float): Return calculated on cumulative-adjusted values
            - annualized_volatility (float)
            - sharpe_ratio (float, risk-free rate ~ 1.2%)
            - max_drawdown (float)
            - sortino_ratio (float)
            - portfolio_values (np.ndarray): Array of portfolio values (cumulative-adjusted)
            - avg_weights (np.ndarray): Average weights across assets
            - final_portfolio_value (float): Final value for next iteration
    """
    episodes_data = []
    
    # ===== RUN MULTIPLE EPISODES =====
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        
        portfolio_values = []
        weights_list = []
        dates = []
        transaction_costs = []
        turnover = []
        initial_capital = 1000000
        
        # ===== COLLECT DATA DURING EPISODE =====
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Extract from info dict (VecEnv returns list of dicts)
            dates.append(info[0]['date'])
            portfolio_values.append(info[0]['portfolio_value'])
            weights_list.append(info[0]['weights'].copy())
            transaction_costs.append(info[0].get('transaction_cost', 0.0))
            turnover.append(info[0].get('turnover', 0.0))
        
        # Convert to arrays
        portfolio_values = np.array(portfolio_values)
        weights_array = np.array(weights_list)
        
        # CUMULATIVE ADJUSTMENT: Adjust portfolio values if carrying forward from previous iteration
        cumulative_offset = 0
        calculation_initial_capital = initial_capital  # For metric calculation
        
        if cumulative_adjustment is not None:
            prev_final = cumulative_adjustment.get('prev_final_value')
            curr_initial = cumulative_adjustment.get('current_initial_value')
            
            if prev_final is not None and curr_initial is not None:
                # Calculate offset to make current iteration start where previous ended
                cumulative_offset = prev_final - curr_initial
                portfolio_values = portfolio_values + cumulative_offset
                calculation_initial_capital = prev_final  # Metrics based on cumulative values

        # build portfolio values dataframe
        portfolio_df = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values
        })
        
        # build weights dataframe
        n_assets = weights_array.shape[1]
        asset_columns = [f'asset_{i}' for i in range(n_assets)]
        weights_df = pd.DataFrame(weights_array, columns=asset_columns)
        weights_df.insert(0, 'date', dates)
        
        # build transaction costs dataframe
        transaction_df = pd.DataFrame({
            'date': dates,
            'transaction_cost': transaction_costs
        })
        
        # build turnover dataframe
        turnover_df = pd.DataFrame({
            'date': dates,
            'turnover': turnover
        })
        
        # ===== CALCULATE METRICS =====
        
        # Daily returns from portfolio values
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Total return over period (calculated from cumulative-adjusted values)
        total_return = (portfolio_values[-1] - calculation_initial_capital) / calculation_initial_capital
        n_days = len(daily_returns)
        
        # Annualized return (convert from daily to annual)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        
        # Annualized volatility
        annualized_volatility = np.std(daily_returns) * np.sqrt(252)
        
        sharpe_ratio = (
            (annualized_return - get_rf_rate(start_date='2019-01-01', end_date='2024-12-01')) / annualized_volatility 
            if annualized_volatility > 0 else 0
        )
        
        # Maximum Drawdown
        cumulative_return = np.cumprod(1 + daily_returns)
        running_max = np.maximum.accumulate(cumulative_return)
        drawdowns = (cumulative_return - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Sortino Ratio (downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0:
            downside_dev = np.std(negative_returns) * np.sqrt(252)
        else:
            downside_dev = annualized_volatility  # Fallback
        
        sortino_ratio = (
            (annualized_return - get_rf_rate(start_date='2019-01-01', end_date='2024-12-01')) / downside_dev 
            if downside_dev > 0 else 0
        )
        
        # Average weights across episode (skip first since it's initial allocation)
        avg_weights = np.mean(weights_array[1:], axis=0)
        
        # Store episode data
        episodes_data.append({
            'portfolio_values': portfolio_values,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'sortino_ratio': sortino_ratio,
            'avg_weights': avg_weights,
            'portfolio_df': portfolio_df,
            'weights_df': weights_df,
            'transaction_df': transaction_df,
            'turnover_df': turnover_df
        })
    
    # ===== COMPUTE AVERAGES ACROSS ALL EPISODES =====
    avg_return = np.mean([ep['annualized_return'] for ep in episodes_data])
    avg_volatility = np.mean([ep['annualized_volatility'] for ep in episodes_data])
    avg_sharpe = np.mean([ep['sharpe_ratio'] for ep in episodes_data])
    avg_drawdown = np.mean([ep['max_drawdown'] for ep in episodes_data])
    avg_sortino = np.mean([ep['sortino_ratio'] for ep in episodes_data])
    avg_weights = np.mean([ep['avg_weights'] for ep in episodes_data], axis=0)
    
    # Use last episode's portfolio values for reference
    last = episodes_data[-1]
    final_portfolio_value = last['portfolio_values'][-1]
    
    return {
        'annualized_return': avg_return,
        'annualized_volatility': avg_volatility,
        'sharpe_ratio': avg_sharpe,
        'max_drawdown': avg_drawdown,
        'sortino_ratio': avg_sortino,
        'portfolio_values': last['portfolio_values'],
        'avg_weights': avg_weights,
        'portfolio_df': last['portfolio_df'],
        'weights_df': last['weights_df'],
        'transaction_df': last['transaction_df'],
        'turnover_df': last['turnover_df'],
        'final_portfolio_value': final_portfolio_value  # For carrying to next iteration
    }

class LoggerCallback(BaseCallback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file
        self.log_data = []
    
    def _on_step(self) -> bool:
        # Log timestep, reward, and other metrics
        self.log_data.append({
            'timestep': self.num_timesteps,
            'reward': self.locals.get('rewards', None),
            'done': self.locals.get('dones', None)
        })
        return True
    
    def _on_training_end(self) -> None:
        # Write to file when training ends
        with open(self.log_file, 'w') as f:
            f.write("Training Log\n")
            f.write("=" * 50 + "\n")
            for entry in self.log_data:
                f.write(str(entry) + "\n")

class OnnxableSB3Policy(th.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # NOTE: Preprocessing is included, but postprocessing
        # (clipping/inscaling actions) is not,
        # If needed, you also need to transpose the images so that they are channel first
        # use deterministic=False if you want to export the stochastic policy
        # policy() returns `actions, values, log_prob` for PPO
        return self.policy(observation, deterministic=True)

def predict_onnx(ort_session, observation, deterministic=True):
    """
    Make predictions using ONNX model
    
    Args:
        ort_session: ONNX Runtime session
        observation: numpy array of shape (n_features,) or (1, n_features)
        deterministic: whether to use deterministic policy (for evaluation)
    
    Returns:
        action: numpy array of predicted actions
    """
    # Ensure observation is 2D: (1, n_features)
    if observation.ndim == 1:
        observation = observation.reshape(1, -1)
    
    # Ensure correct dtype
    observation = observation.astype(np.float32)
    
    # Get input name
    input_name = ort_session.get_inputs()[0].name
    
    # Run inference
    onnx_output = ort_session.run(None, {input_name: observation})
    
    # The output format depends on how you exported the model
    # Typically it's the action (and possibly other outputs like value function)
    action = onnx_output[0]
    
    # If action is 2D (batch, action_dim), squeeze to 1D
    if action.ndim == 2:
        action = action[0]
    
    return action

def evaluate_model_onnx(ort_session, env, num_episodes=10):
    """
    Evaluate ONNX model on the environment
    """
    episode_metrics = []
    
    # Check if environment is wrapped with VecNormalize
    if isinstance(env, VecNormalize):
        normalized_env = env
        is_vectorized = True
    else:
        normalized_env = env
        is_vectorized = False

    for episode in range(num_episodes):
        obs = normalized_env.reset()
        # VecNormalize.reset() returns only obs, not (obs, info)
        if isinstance(obs, tuple):
            obs = obs[0]
        
        # Re-fetch unwrapped_env reference AFTER reset to ensure we have the fresh environment
        if is_vectorized:
            base_env = normalized_env.venv if hasattr(normalized_env, 'venv') else normalized_env.envs
            unwrapped_env = base_env.envs[0] if hasattr(base_env, 'envs') else base_env[0]
        else:
            unwrapped_env = normalized_env.unwrapped
        
        done = False
        
        # Track portfolio values and weights over episode
        portfolio_values = [unwrapped_env.portfolio_value]        
        weights_history = [unwrapped_env.current_weights.copy()]

        
        while not done:
            # Use ONNX model for prediction
            action = predict_onnx(ort_session, obs, deterministic=True)
            
            # Reshape action for vectorized environment (add batch dimension back)
            if isinstance(action, np.ndarray) and action.ndim == 1:
                action = action.reshape(1, -1)

            result = normalized_env.step(action)
            
            # Handle both vectorized and non-vectorized env returns
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result

            # Track values and weights from info dict (more reliable than accessing env directly)
            # For vectorized envs, info is a list of dicts; for non-vectorized, it's a dict
            try:
                current_info = info[0] if len(info) > 0 and isinstance(info[0], dict) else info
            except (TypeError, KeyError):
                # Fallback: use unwrapped_env directly
                current_info = {'portfolio_value': unwrapped_env.portfolio_value}
            
            portfolio_values.append(current_info['portfolio_value'])
            weights_history.append(unwrapped_env.current_weights.copy())            
        
        # converts collections to numpy arrays
        portfolio_values = np.array(portfolio_values)        
        weights_history = np.array(weights_history)

        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        n_periods = len(returns)
        
        annualized_return = (1 + total_return) ** (252 / n_periods) - 1
        annualized_volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = (annualized_return - get_rf_rate(start_date='2019-01-01', end_date='2024-12-01')) / annualized_volatility if annualized_volatility > 0 else 0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        avg_weights = np.mean(weights_history[1:], axis=0)
        
        episode_metrics.append({
            "portfolio_values": portfolio_values,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "avg_weights": avg_weights,
            "final_value": portfolio_values[-1]
        })
    
    # Aggregate statistics
    summary = {
        "portfolio_values": episode_metrics[-1]["portfolio_values"],
        "final_value": episode_metrics[-1]["final_value"],
        "annualized_return": episode_metrics[-1]["annualized_return"],
        "annualized_volatility": episode_metrics[-1]["annualized_volatility"],
        "sharpe_ratio": episode_metrics[-1]["sharpe_ratio"],
        "max_drawdown": episode_metrics[-1]["max_drawdown"],
        "weights": episode_metrics[-1]["avg_weights"],
        "current_weights": episode_metrics[-1]["avg_weights"]
    }
    
    return summary

def conduct_multi_seed_rolling_test(model_class,
                                    price_data: pd.DataFrame,
                                    train_start_date: str,
                                    train_end_date: str,
                                    test_end_date: str,
                                    lookback_period: int = 21,
                                    step_years: int = 1,
                                    total_timesteps: int = 100000,
                                    eval_episodes: int = 10,
                                    initial_capital: float = 1000000,
                                    seeds: list = None,
                                    **model_kwargs):
    """
    Conduct rolling robustness test with multiple random seeds.
    
    This function runs the rolling robustness test multiple times with different seeds,
    collecting results from each run and computing mean/std statistics.
    
    Args:
        model_class: SB3 Model class (A2C, PPO, DDPG, SAC)
        price_data (pd.DataFrame): Full price data with DatetimeIndex
        train_start_date (str): Initial training start date
        train_end_date (str): Initial training end date
        test_end_date (str): Final testing end date
        lookback_period (int): Lookback period (default: 21)
        step_years (int): Years to roll forward (default: 1)
        total_timesteps (int): Timesteps per training iteration
        eval_episodes (int): Episodes for evaluation (default: 10)
        initial_capital (float): Initial capital for first iteration (default: 1000000)
        seeds (list): List of random seeds (default: [0, 1, 2, 3, 4])
        **model_kwargs: Additional arguments for model_class constructor
    
    Returns:
        dict: Contains:
            - 'all_seeds_results': Dict mapping seed -> results from conduct_rolling_robustness_test()
            - 'mean_stats': Mean statistics across all seeds
            - 'std_stats': Standard deviation statistics across all seeds
            - 'hyperparameters': Dict of all hyperparameters used
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    
    all_seeds_results = {}
    
    # Store hyperparameters for logging
    hyperparameters = {
        'model_class': model_class.__name__,
        'train_start_date': train_start_date,
        'train_end_date': train_end_date,
        'test_end_date': test_end_date,
        'lookback_period': lookback_period,
        'step_years': step_years,
        'total_timesteps': total_timesteps,
        'eval_episodes': eval_episodes,
        'initial_capital': initial_capital,
        'seeds': seeds,
        'num_seeds': len(seeds),
        **model_kwargs
    }
    
    print(f"\n{'='*70}")
    print(f"MULTI-SEED ROLLING TEST: {model_class.__name__}")
    print(f"Total Seeds: {len(seeds)}")
    print(f"Seeds: {seeds}")
    print(f"{'='*70}\n")
    
    # Run test for each seed
    for seed in seeds:
        print(f"\n>>> Running test with seed: {seed}")
        
        # Set seed for reproducibility
        np.random.seed(seed)
        th.manual_seed(seed)
        
        # Run rolling test
        results = conduct_rolling_robustness_test(
            model_class=model_class,
            price_data=price_data,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_end_date=test_end_date,
            lookback_period=lookback_period,
            step_years=step_years,
            total_timesteps=total_timesteps,
            eval_episodes=eval_episodes,
            initial_capital=initial_capital,
            **{k: v for k, v in model_kwargs.items() if k not in ['seed']}
        )
        
        all_seeds_results[seed] = results
        print(f"<<< Completed seed: {seed}")
    
    # Calculate aggregate statistics
    mean_stats, std_stats = _compute_seed_statistics(all_seeds_results)
    
    # NEW: Export tracking data organized by seed and iteration
    model_name = model_class.__name__
    for seed in seeds:
        seed_results = all_seeds_results[seed]
        seed_tracking = seed_results.get('tracking_data', [])
        
        if seed_tracking:
            seed_dir = os.path.join('tables', f'{model_name}_seed{seed}')
            os.makedirs(seed_dir, exist_ok=True)
            
            for track in seed_tracking:
                iter_num = track['iteration']
                iter_dir = os.path.join(seed_dir, f'iteration_{iter_num:02d}')
                os.makedirs(iter_dir, exist_ok=True)
                
                # Export weights
                weights_path = os.path.join(iter_dir, 'daily_weights.csv')
                track['weights_df'].to_csv(weights_path, index=False)
                print(f"Exported: {weights_path}")
                
                # Export transaction costs
                transaction_path = os.path.join(iter_dir, 'daily_transaction_costs.csv')
                track['transaction_df'].to_csv(transaction_path, index=False)
                print(f"Exported: {transaction_path}")
                
                # NEW: Export turnover
                turnover_path = os.path.join(iter_dir, 'daily_turnover.csv')
                track['turnover_df'].to_csv(turnover_path, index=False)
                print(f"Exported: {turnover_path}")
    
    return {
        'all_seeds_results': all_seeds_results,
        'mean_stats': mean_stats,
        'std_stats': std_stats,
        'hyperparameters': hyperparameters
    }

def _compute_seed_statistics(all_seeds_results: dict) -> Tuple[dict, dict]:
    """
    Compute mean and standard deviation statistics across all seeds.
    
    Args:
        all_seeds_results (dict): Dict mapping seed -> results from conduct_rolling_robustness_test()
    
    Returns:
        tuple: (mean_stats, std_stats) dicts containing aggregated statistics
    """
    # Extract all summary stats from each seed
    all_summary_stats = [results['summary_stats'] for results in all_seeds_results.values()]
    
    # Extract all iteration-level results
    all_iterations = [results['all_results'] for results in all_seeds_results.values()]
    
    # Compute statistics for summary-level metrics
    metrics_to_aggregate = [
        'mean_annualized_return', 'std_annualized_return',
        'mean_sharpe_ratio', 'std_sharpe_ratio',
        'mean_max_drawdown', 'std_max_drawdown',
        'mean_volatility', 'std_volatility',
        'mean_sortino_ratio', 'std_sortino_ratio',
        'min_return', 'max_return', 'min_sharpe', 'max_sharpe',
        'final_portfolio_value', 'total_return'
    ]
    
    mean_stats = {}
    std_stats = {}
    
    for metric in metrics_to_aggregate:
        values = [stats[metric] for stats in all_summary_stats if metric in stats]
        if values:
            mean_stats[metric] = np.mean(values)
            std_stats[metric] = np.std(values)
    
    # Compute iteration-level statistics
    # Organize by iteration number
    iterations_by_num = {}
    for seed_idx, iterations in enumerate(all_iterations):
        for iteration_data in iterations:
            iter_num = iteration_data['iteration']
            if iter_num not in iterations_by_num:
                iterations_by_num[iter_num] = []
            iterations_by_num[iter_num].append(iteration_data)
    
    mean_stats['iterations'] = {}
    std_stats['iterations'] = {}
    
    for iter_num in sorted(iterations_by_num.keys()):
        iter_data_list = iterations_by_num[iter_num]
        
        mean_stats['iterations'][iter_num] = {
            'mean_annualized_return': np.mean([d['annualized_return'] for d in iter_data_list]),
            'mean_sharpe_ratio': np.mean([d['sharpe_ratio'] for d in iter_data_list]),
            'mean_max_drawdown': np.mean([d['max_drawdown'] for d in iter_data_list]),
            'mean_annualized_volatility': np.mean([d['annualized_volatility'] for d in iter_data_list]),
            'mean_final_portfolio_value': np.mean([d['final_portfolio_value'] for d in iter_data_list]),
        }
        
        std_stats['iterations'][iter_num] = {
            'std_annualized_return': np.std([d['annualized_return'] for d in iter_data_list]),
            'std_sharpe_ratio': np.std([d['sharpe_ratio'] for d in iter_data_list]),
            'std_max_drawdown': np.std([d['max_drawdown'] for d in iter_data_list]),
            'std_annualized_volatility': np.std([d['annualized_volatility'] for d in iter_data_list]),
            'std_final_portfolio_value': np.std([d['final_portfolio_value'] for d in iter_data_list]),
        }
    
    return mean_stats, std_stats

def export_multi_seed_results_to_csv(multi_seed_results: dict, output_dir: str):
    """
    Export multi-seed results to multiple CSV files.
    
    Creates the following files in output_dir:
    - {model}_seeds_summary_stats.csv: Summary statistics with mean/std
    - {model}_seeds_iteration_stats.csv: Per-iteration mean/std statistics
    - {model}_seeds_individual_results_{seed}.csv: Individual results per seed
    
    Args:
        multi_seed_results (dict): Output from conduct_multi_seed_rolling_test()
        output_dir (str): Directory to save output files
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = multi_seed_results['hyperparameters']['model_class']
    
    # 1. Export summary statistics (mean/std across all seeds)
    summary_stats_data = []
    for metric in multi_seed_results['mean_stats'].keys():
        if metric != 'iterations':
            summary_stats_data.append({
                'metric': metric,
                'mean': multi_seed_results['mean_stats'][metric],
                'std': multi_seed_results['std_stats'][metric]
            })
    
    summary_df = pd.DataFrame(summary_stats_data)
    summary_path = os.path.join(output_dir, f'{model_name}_seeds_summary_stats.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics exported to {summary_path}")
    
    # 2. Export iteration-level statistics
    iteration_data = []
    for iter_num in sorted(multi_seed_results['mean_stats']['iterations'].keys()):
        row = {
            'iteration': iter_num,
            'mean_annualized_return': multi_seed_results['mean_stats']['iterations'][iter_num]['mean_annualized_return'],
            'std_annualized_return': multi_seed_results['std_stats']['iterations'][iter_num]['std_annualized_return'],
            'mean_sharpe_ratio': multi_seed_results['mean_stats']['iterations'][iter_num]['mean_sharpe_ratio'],
            'std_sharpe_ratio': multi_seed_results['std_stats']['iterations'][iter_num]['std_sharpe_ratio'],
            'mean_max_drawdown': multi_seed_results['mean_stats']['iterations'][iter_num]['mean_max_drawdown'],
            'std_max_drawdown': multi_seed_results['std_stats']['iterations'][iter_num]['std_max_drawdown'],
            'mean_annualized_volatility': multi_seed_results['mean_stats']['iterations'][iter_num]['mean_annualized_volatility'],
            'std_annualized_volatility': multi_seed_results['std_stats']['iterations'][iter_num]['std_annualized_volatility'],
            'mean_final_portfolio_value': multi_seed_results['mean_stats']['iterations'][iter_num]['mean_final_portfolio_value'],
            'std_final_portfolio_value': multi_seed_results['std_stats']['iterations'][iter_num]['std_final_portfolio_value'],
        }
        iteration_data.append(row)
    
    iteration_df = pd.DataFrame(iteration_data)
    iteration_path = os.path.join(output_dir, f'{model_name}_seeds_iteration_stats.csv')
    iteration_df.to_csv(iteration_path, index=False)
    print(f"Iteration statistics exported to {iteration_path}")
    
    # 3. Export individual results for each seed
    for seed, results in multi_seed_results['all_seeds_results'].items():
        export_rolling_results_to_csv(results, os.path.join(output_dir, f'{model_name}_seed_{seed}_results.csv'))
        export_hyperparameters_to_csv(results, os.path.join(output_dir, f'{model_name}_seed_{seed}_hyperparams.csv'))
        export_portfolio_values_to_csv(results, os.path.join(output_dir, f'{model_name}_seed_{seed}_portfolio.csv'))
        
        # NEW: Export weights, turnover, and transaction costs with full metadata
        export_weights_to_csv(results, os.path.join(output_dir, f'{model_name}_seed_{seed}_weights.csv'))
        export_turnover_to_csv(results, os.path.join(output_dir, f'{model_name}_seed_{seed}_turnover.csv'))
        export_transaction_costs_to_csv(results, os.path.join(output_dir, f'{model_name}_seed_{seed}_transaction_costs.csv'))

def export_multi_seed_hyperparameters_to_csv(multi_seed_results: dict, output_filepath: str):
    """
    Export multi-seed hyperparameters and aggregated statistics to CSV.
    
    Args:
        multi_seed_results (dict): Output from conduct_multi_seed_rolling_test()
        output_filepath (str): Path to save hyperparameters CSV
    """
    hyperparams = multi_seed_results['hyperparameters'].copy()
    mean_stats = multi_seed_results['mean_stats'].copy()
    std_stats = multi_seed_results['std_stats'].copy()
    
    # Remove nested 'iterations' dict for CSV export
    if 'iterations' in mean_stats:
        del mean_stats['iterations']
    if 'iterations' in std_stats:
        del std_stats['iterations']
    
    # Combine all data with mean_ and std_ prefixes
    combined = {**hyperparams}
    for key, value in mean_stats.items():
        combined[f'mean_{key}'] = value
    for key, value in std_stats.items():
        combined[f'std_{key}'] = value
    
    # Create a DataFrame with single row
    df = pd.DataFrame([combined])
    
    # Save to CSV
    df.to_csv(output_filepath, index=False)
    print(f"Multi-seed hyperparameters and stats exported to {output_filepath}")

def export_cumulative_period_results(multi_seed_results: dict, output_dir: str):
    """
    Export cumulative results for the entire test period (all iterations combined).
    
    This calculates performance metrics for the complete test period (e.g., 2019-2024)
    by combining all iteration results for each seed, then exports:
    - {model}_cumulative_period_stats.csv: Per-seed cumulative metrics
    - {model}_cumulative_period_summary.csv: Mean/std across all seeds for cumulative metrics
    
    Args:
        multi_seed_results (dict): Output from conduct_multi_seed_rolling_test()
        output_dir (str): Directory to save output files
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = multi_seed_results['hyperparameters']['model_class']
    
    # Aggregate portfolio values for each seed across all iterations
    cumulative_metrics = []
    
    for seed, seed_results in multi_seed_results['all_seeds_results'].items():
        # Combine all portfolio values from each iteration
        all_portfolio_values = []
        dates = []
        
        for iteration_result in seed_results['all_results']:
            portfolio_df = iteration_result['portfolio_df']
            all_portfolio_values.extend(portfolio_df['portfolio_value'].values)
            dates.extend(portfolio_df['date'].values)
        
        if len(all_portfolio_values) == 0:
            continue
        
        # Create cumulative portfolio series
        portfolio_values = pd.Series(all_portfolio_values)
        initial_capital = seed_results['all_results'][0]['starting_capital']
        
        # Calculate metrics for cumulative period
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        cumulative_days = len(portfolio_values)
        annualized_return = ((1 + total_return) ** (252 / cumulative_days)) - 1
        
        # Volatility
        daily_returns = portfolio_values.pct_change().dropna()
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio (using period-specific risk-free rate)
        # Get the date range for the entire test period
        test_start_date = seed_results['all_results'][0]['test_start_date']
        test_end_date = seed_results['all_results'][-1]['test_end_date']
        
        try:
            # Use the new get_rf_rate function with period-specific dates
            rf_rate = get_rf_rate(start_date='2019-01-01', end_date='2024-12-01')
        except:
            # Fallback to default if period-specific calculation fails
            rf_rate = 0.02  # Assume 2% if not available
        
        if annualized_volatility > 0:
            sharpe_ratio = (annualized_return - rf_rate) / annualized_volatility
        else:
            sharpe_ratio = 0
        
        # Max Drawdown
        cumulative_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        
        # Sortino Ratio (only negative returns for downside volatility)
        downside_returns = daily_returns[daily_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        if downside_volatility > 0:
            sortino_ratio = (annualized_return - rf_rate) / downside_volatility
        else:
            sortino_ratio = 0 if annualized_return <= rf_rate else np.inf
        
        cumulative_metrics.append({
            'seed': seed,
            'test_period': f"{test_start_date} to {test_end_date}",
            'cumulative_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': portfolio_values.iloc[-1],
            'cumulative_trading_days': cumulative_days,
        })
    
    # Export per-seed cumulative results
    cumulative_df = pd.DataFrame(cumulative_metrics)
    cumulative_path = os.path.join(output_dir, f'{model_name}_cumulative_period_stats.csv')
    cumulative_df.to_csv(cumulative_path, index=False)
    print(f"Cumulative period statistics (per seed) exported to {cumulative_path}")
    
    # Export summary statistics (mean/std across seeds)
    if len(cumulative_metrics) > 0:
        summary_metrics = []
        for metric_name in ['cumulative_return', 'annualized_return', 'annualized_volatility', 
                            'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'final_portfolio_value']:
            values = [m[metric_name] for m in cumulative_metrics]
            summary_metrics.append({
                'metric': metric_name,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            })
        
        summary_df = pd.DataFrame(summary_metrics)
        summary_path = os.path.join(output_dir, f'{model_name}_cumulative_period_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Cumulative period summary (mean/std across seeds) exported to {summary_path}")