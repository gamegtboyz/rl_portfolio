import pandas as pd
import numpy as np
import pypfopt as ppo
import quantstats as qs
import csv
from datetime import datetime
import os

def date_index(price_data, date_string: str) -> int:
    '''converts date string to nearest following trading date index'''
    ts = pd.to_datetime(date_string)
    loc = int(price_data.index.get_indexer([ts], method='ffill')[0])

    return loc

def orderdict_nparray(dict):
    '''convert OrderedDict to numpy array'''
    results = []
    for key in dict:
        results.append(dict[key]) 
    return np.array(results)

def rebalance_portfolio(price_data, port_initial_date, lookback_period=252, rebalance_frequency=1, initial_capital=1000000, bounds=(0,1), transaction_cost=0.0015, mode='max_sharpe', gamma=0.00001):
    # safer initialization: equal weights
    n_assets = len(price_data.columns)
    current_weights = np.ones(n_assets) / n_assets

    return_data = price_data.pct_change()
    # pandas Series for robust indexing
    portfolio_values = pd.Series(np.nan, index=price_data.index, dtype=float)

    initial_loc = date_index(price_data, port_initial_date)
    # set starting portfolio value at initial_loc
    portfolio_values.iloc[initial_loc] = initial_capital

    for t in range(initial_loc, len(price_data), rebalance_frequency):
        # build price window for optimization (use up to t, exclusive)
        start_w = max(0, t - lookback_period)
        price_window = price_data.iloc[start_w:t]
        try:
            mu = ppo.expected_returns.mean_historical_return(price_window)
            S = ppo.risk_models.sample_cov(price_window)
            ef = ppo.EfficientFrontier(mu, S, weight_bounds=bounds)
            if mode == 'max_sharpe':
                ef.max_sharpe()
            elif mode == 'min_volatility':
                ef.add_objective(ppo.objective_functions.L2_reg, gamma=gamma)
                ef.min_volatility()
            target_weights = orderdict_nparray(ef.clean_weights())
            # ensure normalized (numerical safety)
            target_weights = np.clip(target_weights, -1.0, 1.0)
            if target_weights.sum() != 0:
                target_weights = target_weights / (np.sum(np.abs(target_weights)) + 1e-12) * np.sum(np.abs(target_weights))
        except Exception:
            target_weights = current_weights.copy()

        last_weights = current_weights.copy()
        current_weights = target_weights.copy()

        # apply to days t .. t+rebalance_frequency-1
        apply_end = min(t + rebalance_frequency, len(price_data))
        for idx in range(t, apply_end):
            daily_ret = return_data.iloc[idx].values  # return for day idx
            portfolio_return = float(np.dot(current_weights, daily_ret))
            turnover = np.sum(np.abs(current_weights - last_weights)) if idx == t and np.any(current_weights != last_weights) else 0.0
            portfolio_return -= turnover * transaction_cost

            # previous value: find last non-nan before idx
            if idx == 0:
                prev_val = initial_capital
            else:
                prev_val = portfolio_values.iloc[idx - 1]
                if pd.isna(prev_val):
                    # fallback to nearest previous valid
                    prev_valid_idx = portfolio_values[:idx].last_valid_index()
                    prev_val = portfolio_values.loc[prev_valid_idx] if prev_valid_idx is not None else initial_capital

            portfolio_values.iloc[idx] = prev_val * (1 + portfolio_return)

    # drop leading NaNs (before initial_loc)
    portfolio_values = portfolio_values.dropna()

    cumulative_return = (portfolio_values.iloc[-1] / initial_capital) - 1
    annualized_return = ((1 + cumulative_return) ** (1 / (len(portfolio_values) / 252))) - 1
    volatility = portfolio_values.pct_change().std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - 0.03) / volatility if volatility != 0 else 0
    max_drawdown = qs.stats.max_drawdown(portfolio_values)

    results = {
        'portfolio_values': portfolio_values,
        'annualized_return': f"{annualized_return:.2%}",
        'annualized_volatility': f"{volatility:.2%}",
        'sharpe_ratio': f"{sharpe_ratio:.2f}",
        'max_drawdown': f"{max_drawdown:.2%}",
        'current_weights': current_weights
    }
    return results

def buy_and_hold(price_data, port_initial_date, initial_capital:float): 
    current_weights = np.array([1/len(price_data.columns)] * len(price_data.columns))       # initial weights is equal weights
    return_data = price_data.pct_change()                                                   # calculate daily returns
    
    port_initial_index = date_index(price_data, port_initial_date)

    portfolio_values = [initial_capital]

    for i in range(1, len(price_data.iloc[port_initial_index:])):
        daily_return = return_data.iloc[port_initial_index + i - 1]
        portfolio_return = np.dot(daily_return, current_weights)
        portfolio_values.append(portfolio_values[i-1] * (1 + portfolio_return))

    portfolio_values = pd.Series(portfolio_values, index=price_data.iloc[port_initial_index:].index)

    # calculate portfolio performance metrics
    cumulative_return = (portfolio_values.iloc[-1]/initial_capital) - 1
    annualized_return = ((cumulative_return+1)**(1/(len(portfolio_values)/252))) - 1
    volatility = portfolio_values.pct_change().std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - 0.03) / volatility if volatility != 0 else 0
    max_drawdown = qs.stats.max_drawdown(portfolio_values)

    results = {
        'portfolio_values': portfolio_values,
        'annualized_return': f"{annualized_return:.2%}",
        'annualized_volatility': f"{volatility:.2%}",
        'sharpe_ratio': f"{sharpe_ratio:.2f}",
        'max_drawdown': max_drawdown,
        'current_weights': current_weights
    }

    return results

def cal_indicator(price_data:pd.DataFrame, port_initial_date:str):
    '''
    Calculates technical indicators and appends them to the price dataframe
    Indicators calculated:
    1. MACD (MACD line and signal line)
    2. RSI (Relative Strength Index)
    3. ATR (Average True Range)
    4. ADX (Average Directional Index)
    
    Returns: DataFrame with original prices + all indicators
    '''
    port_initial_index = date_index(price_data, port_initial_date)
    result_df = price_data.copy()

    # 1. Calculate MACD
    short_ema = price_data.ewm(span=12, adjust=False).mean()
    long_ema = price_data.ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    
    # Append MACD to result
    for col in price_data.columns:
        result_df[f'{col}_MACD'] = macd[col]
        result_df[f'{col}_Signal'] = signal[col]

    # 2. Calculate RSI
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    for col in price_data.columns:
        result_df[f'{col}_RSI'] = calculate_rsi(price_data[col])

    # 3. Calculate ATR
    def calculate_atr(high, low, close, window=14):
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    for col in price_data.columns:
        price_col = price_data[col]
        high_proxy = price_col * 1.005
        low_proxy = price_col * 0.995
        result_df[f'{col}_ATR'] = calculate_atr(high_proxy, low_proxy, price_col, window=14)

    # 4. Calculate ADX
    def calculate_adx(high, low, close, window=14):
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)
        
        plus_dm[(high_diff > 0) & (high_diff > low_diff)] = high_diff[(high_diff > 0) & (high_diff > low_diff)]
        minus_dm[(low_diff > 0) & (low_diff > high_diff)] = low_diff[(low_diff > 0) & (low_diff > high_diff)]
        
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / (atr + 1e-6))
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / (atr + 1e-6))
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-6)
        adx = dx.rolling(window=window).mean()
        
        return adx
    
    for col in price_data.columns:
        price_col = price_data[col]
        high_proxy = price_col * 1.005
        low_proxy = price_col * 0.995
        result_df[f'{col}_ADX'] = calculate_adx(high_proxy, low_proxy, price_col, window=14)
    
    return result_df[port_initial_index:]


hpt_log_file = 'tables/hpt_log.csv'
hpt_columns = [
    'timestamp',
    'model_name',
    'learning_rate',
    'n_steps',
    'batch_size',
    'gamma',
    'gae_lambda',
    'ent_coef',
    'vf_coef',
    'buffer_size',
    'tau',
    'total_timesteps',
    'annualized_return',
    'sharpe_ratio',
    'max_drawdown',
    'annualized_volatility',
    'sortino_ratio'
]

if not os.path.exists(hpt_log_file):
    with open(hpt_log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=hpt_columns)
        writer.writeheader()

def log_hpt_results(model_name, hyperparams, total_timesteps, eval_results):
    """Log hyperparameter tuning results to CSV"""
    row = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'total_timesteps': total_timesteps,
        'annualized_return': eval_results.get('annualized_return'),
        'sharpe_ratio': eval_results.get('sharpe_ratio'),
        'max_drawdown': eval_results.get('max_drawdown'),
        'annualized_volatility': eval_results.get('annualized_volatility'),
        'sortino_ratio': eval_results.get('sortino_ratio')            
    }
    # Add hyperparameters
    row.update(hyperparams)
    
    with open(hpt_log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=hpt_columns)
        writer.writerow(row)
    
    print(f"✓ Logged {model_name} results to {hpt_log_file}")

print("✓ Hyperparameter tracking initialized!")