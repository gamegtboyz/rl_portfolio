from utils import date_index
import pandas as pd
import numpy as np
import quantstats as qs
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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

def evaluate_model_sb3(model, env, num_episodes=10):
    """
    Evaluate a Stable Baselines3 model on the environment
    
    Args:
        model: Loaded SB3 model (A2C, PPO, DDPG, SAC, etc.)
        env: Environment to evaluate on (typically VecNormalize wrapped)
        num_episodes: Number of episodes to evaluate
    
    Returns:
        dict with evaluation metrics:
            - annualized_return: mean return across episodes
            - sharpe_ratio: 0.0 (placeholder)
            - max_drawdown: 0.0 (placeholder)
            - annualized_volatility: 0.0 (placeholder)
            - sortino_ratio: 0.0 (placeholder)
    """
    episode_returns = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        portfolio_values = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Track portfolio value if available
            if hasattr(env.envs[0], 'portfolio_value'):
                portfolio_values.append(env.envs[0].portfolio_value)
        
        # Calculate total return for this episode
        if portfolio_values:
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            episode_returns.append(total_return)
    
    # Return metrics dictionary
    return {
        'annualized_return': np.mean(episode_returns) if episode_returns else 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'annualized_volatility': 0.0,
        'sortino_ratio': 0.0,
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
        risk_free_rate = 0.03
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
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