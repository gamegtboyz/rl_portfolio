from utils import date_index, rf_rate
import pandas as pd
import numpy as np
import quantstats as qs
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, unwrap_vec_normalize
from portfolio_env import PortfolioEnv
import torch as th
from typing import Tuple
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import BaseCallback

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
                                      step_years: int = 1):
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
        
        # Build normalized train environment
        train_env = DummyVecEnv([lambda data=train_data: PortfolioEnv(
            price_data=data,
            lookback_period=lookback_period,
            transaction_cost=0.0005,
            risk_aversion=0.01)])
        train_env = VecNormalize(train_env)
        
        # Build normalized test environment
        test_env = DummyVecEnv([lambda data=test_data: PortfolioEnv(
            price_data=data,
            lookback_period=lookback_period,
            transaction_cost=0.0015)])
        test_env = VecNormalize(test_env, training=False, norm_reward=False)
        
        yield {
            'train_env': train_env,
            'test_env': test_env,
            'train_start_date': train_start.strftime('%Y-%m-%d'),
            'train_end_date': train_end.strftime('%Y-%m-%d'),
            'test_start_date': test_start.strftime('%Y-%m-%d'),
            'test_end_date': test_period_end.strftime('%Y-%m-%d'),
            'iteration': iteration
        }
        
        # Roll forward by step_years
        train_start = train_start + relativedelta(years=step_years)
        train_end = train_end + relativedelta(years=step_years)
        iteration += 1
        
        # Stop if test period has reached the end date
        if test_period_end >= test_end:
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
                                   **model_kwargs):
    """
    Conduct rolling robustness test across multiple train/test windows.
    
    This function trains a model on rolling windows of historical data and evaluates
    on subsequent test periods, aggregating results across all iterations.
    
    Args:
        model_class: SB3 Model class (A2C, PPO, DDPG, SAC)
        price_data (pd.DataFrame): Full price data with DatetimeIndex
        train_start_date (str): Initial training start date
        train_end_date (str): Initial training end date
        test_end_date (str): Final testing end date
        lookback_period (int): Lookback period (default: 21)
        step_years (int): Years to roll forward (default: 1)
        total_timesteps (int): Timesteps per training iteration
        eval_episodes (int): Episodes for evaluation (default: 1)
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
        **model_kwargs  # Include all model-specific hyperparameters
    }
    
    # Iterate through rolling windows
    for split_data in rolling_split_build_normalize_env(
        price_data=price_data,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_end_date=test_end_date,
        lookback_period=lookback_period,
        step_years=step_years
    ):
        iteration = split_data['iteration']
        train_env = split_data['train_env']
        test_env = split_data['test_env']
        
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"Training: {split_data['train_start_date']} to {split_data['train_end_date']}")
        print(f"Testing: {split_data['test_start_date']} to {split_data['test_end_date']}")
        print(f"{'='*70}")
        
        # Train model
        model = model_class('MlpPolicy', train_env, **model_kwargs)
        model.learn(total_timesteps=total_timesteps)
        
        # Evaluate on test set
        eval_results = evaluate_model_sb3(model, test_env, num_episodes=eval_episodes)
        
        # Store results with metadata
        iteration_results = {
            'iteration': iteration,
            'train_start_date': split_data['train_start_date'],
            'train_end_date': split_data['train_end_date'],
            'test_start_date': split_data['test_start_date'],
            'test_end_date': split_data['test_end_date'],
            **eval_results  # Unpack evaluation metrics
        }
        
        all_results.append(iteration_results)
        
        # Store portfolio values separately for aggregation
        portfolio_values_all[iteration] = {
            'dates': eval_results['portfolio_df']['date'].tolist(),
            'portfolio_values': eval_results['portfolio_values'].tolist()
        }
        
        # Print iteration summary
        print(f"Annualized Return: {eval_results['annualized_return']:.4f}")
        print(f"Sharpe Ratio: {eval_results['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {eval_results['max_drawdown']:.4f}")
        print(f"Final Portfolio Value: {eval_results['portfolio_values'][-1]:,.0f}")
    
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
    }
    
    return {
        'all_results': all_results,
        'summary_stats': summary_stats,
        'hyperparameters': hyperparameters,
        'portfolio_values_all': portfolio_values_all
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

def evaluate_model_sb3(model, env, num_episodes=10):
    """
    Clean redesign: Evaluate a Stable Baselines3 model on the environment.
    
    Args:
        model: Loaded SB3 model (A2C, PPO, DDPG, SAC)
        env: VecNormalize wrapped environment
        num_episodes (int): Number of evaluation episodes
    
    Returns:
        dict with metrics from LAST episode:
            - annualized_return (float)
            - annualized_volatility (float)
            - sharpe_ratio (float, risk-free rate ~ 1.2%)
            - max_drawdown (float)
            - sortino_ratio (float)
            - portfolio_values (np.ndarray): Array of portfolio values
            - avg_weights (np.ndarray): Average weights across assets
    """
    episodes_data = []
    
    # ===== RUN MULTIPLE EPISODES =====
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        
        portfolio_values = []
        weights_list = []
        dates = []
        initial_capital = 1000000
        
        # ===== COLLECT DATA DURING EPISODE =====
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Extract from info dict (VecEnv returns list of dicts)
            dates.append(info[0]['date'])
            portfolio_values.append(info[0]['portfolio_value'])
            weights_list.append(info[0]['weights'].copy())
        
        # Convert to arrays
        portfolio_values = np.array(portfolio_values)
        weights_array = np.array(weights_list)

        # build portfolio values dataframe
        portfolio_df = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values
        })
        
        # ===== CALCULATE METRICS =====
        
        # Daily returns from portfolio values
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Total return over period
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        n_days = len(daily_returns)
        
        # Annualized return (convert from daily to annual)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        
        # Annualized volatility
        annualized_volatility = np.std(daily_returns) * np.sqrt(252)
        
        sharpe_ratio = (
            (annualized_return - rf_rate) / annualized_volatility 
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
            (annualized_return - rf_rate) / downside_dev 
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
            'portfolio_df': portfolio_df
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
    
    return {
        'annualized_return': avg_return,
        'annualized_volatility': avg_volatility,
        'sharpe_ratio': avg_sharpe,
        'max_drawdown': avg_drawdown,
        'sortino_ratio': avg_sortino,
        'portfolio_values': last['portfolio_values'],
        'avg_weights': avg_weights,
        'portfolio_df': last['portfolio_df']
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
        sharpe_ratio = (annualized_return - rf_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
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