import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):
    def __init__(self, price_data, lookback_period=252, initial_capital=1000000, bounds=(0,1), transaction_cost=0.0015, risk_aversion=0.05, reward_scale=0.1):
        super().__init__()

        self.feature_tensor = self.build_feature_tensor(price_data)
        self.price_data = price_data
        self.dates = price_data.index
        self.returns = price_data.pct_change().fillna(0).values
        self.n_assets = self.feature_tensor.shape[1]
        self.n_features = self.feature_tensor.shape[2]
        self.initial_capital = initial_capital
        self.bounds = bounds
        self.risk_aversion = risk_aversion
        self.lookback_window = lookback_period
        self.reward_scale = reward_scale
        self.transaction_cost = transaction_cost

        self.action_space = spaces.Box(low=bounds[0], high=bounds[1], shape=(self.n_assets,), dtype=np.float32)
        self.current_step = self.lookback_window
        self.current_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        
        obs_shape = self.build_observation().shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        self.reset()
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.portfolio_value = self.initial_capital
        self.current_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.portfolio_history = [self.initial_capital]
        self.prev_portfolio_value = self.initial_capital        
        obs = self.build_observation()
        info = {"step": self.current_step, "portfolio_value": self.portfolio_value, "date": self.dates[self.current_step]}
        return (obs, info)
    
    def build_feature_tensor(self, price_df=None):
        
        '''
        Receives the closing price dataframe (index=Date, columns=Assets) preprocess by calculating
            1. returns
            2. Normalized RSI(14)
            3. Normalized MACD(12,26,9)
            4. Normalized ATR(14) - Average True Range
            5. Normalized ADX(14) - Average Directional Index
        Then build up the feature_tensor, which is 3D numpy array to store those output
        '''
        if price_df is None:
            price_df = self.price_data

        # 1. calculate returns
        returns_df = price_df.pct_change().fillna(0)

        # 2. calculate RSI(14)
        # calculate relative strength(rs) of the defined window from the change in price
        def calculate_rsi(series, window=14):
            delta = series.diff()
            gain = (delta.where(delta>0,0)).rolling(window=window).mean()
            loss = (delta.where(delta<0,0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        # build rsi from rs
        rsi_df = price_df.apply(lambda x: calculate_rsi(x)).fillna(50)  # when get na, assume as neutral points
        # normailize rsi range [0,100] to the range [0,1]
        rsi_norm = (rsi_df / 100.0)

        # 3. calculate MACD(12,26,9)
        # calculate MACD(12,26,9)
        exp12 = price_df.ewm(span=12, adjust=False).mean()
        exp26 = price_df.ewm(span=26, adjust=False).mean()
        macd_raw = exp12 - exp26
        signal = macd_raw.ewm(span=9, adjust=False).mean()

        # normalize macd by comparing with price
        hist_norm = ((macd_raw - signal) / price_df).fillna(0)*10  # slightly scaling up by multiplication by 10

        # 4. calculate ATR(14) - Average True Range
        def calculate_atr(high, low, close, window=14):
            # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
            tr1 = high - low
            tr2 = np.abs(high - close.shift())
            tr3 = np.abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=window).mean()
            return atr
        
        # For our single price data, estimate high/low from volatility
        # Use price +/- 0.5% as rough high/low proxy
        atr_list = []
        for col in price_df.columns:
            price_col = price_df[col]
            high_proxy = price_col * 1.005
            low_proxy = price_col * 0.995
            atr_col = calculate_atr(high_proxy, low_proxy, price_col, window=14)
            atr_list.append(atr_col)
        
        atr_df = pd.concat(atr_list, axis=1)
        atr_df.columns = price_df.columns
        # normalize ATR by price (ATR / price)
        atr_norm = (atr_df / price_df).fillna(0)

        # 5. calculate ADX(14) - Average Directional Index (trend strength)
        def calculate_adx(high, low, close, window=14):
            # True Range
            tr1 = high - low
            tr2 = np.abs(high - close.shift())
            tr3 = np.abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=window).mean()
            
            # Directional Movement
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
        
        adx_list = []
        for col in price_df.columns:
            price_col = price_df[col]
            high_proxy = price_col * 1.005
            low_proxy = price_col * 0.995
            adx_col = calculate_adx(high_proxy, low_proxy, price_col, window=14)
            adx_list.append(adx_col)
        
        adx_df = pd.concat(adx_list, axis=1)
        adx_df.columns = price_df.columns
        # normalize ADX to [0,1] (original range 0-100)
        adx_norm = (adx_df / 100.0).fillna(0)

        # 6. Final safety check for Infs (sometimes caused by division by tiny numbers)
        # Replace any Infinity with 0
        returns_df.replace([np.inf, -np.inf], 0, inplace=True)
        rsi_norm.replace([np.inf, -np.inf], 0.5, inplace=True)
        hist_norm.replace([np.inf, -np.inf], 0, inplace=True)
        atr_norm.replace([np.inf, -np.inf], 0, inplace=True)
        adx_norm.replace([np.inf, -np.inf], 0, inplace=True)

        # build feature tensor from built data with shape=(time_steps, n_assets, n_features)
        # feature order = [returns, RSI, MACD, ATR, ADX]
        n_time = len(price_df)
        n_assets = len(price_df.columns)
        n_features = 5

        feature_tensor = np.zeros((n_time, n_assets, n_features), dtype=np.float32)

        feature_tensor[:,:,0] = returns_df.values
        feature_tensor[:,:,1] = rsi_norm.values
        feature_tensor[:,:,2] = hist_norm.values
        feature_tensor[:,:,3] = atr_norm.values
        feature_tensor[:,:,4] = adx_norm.values

        # clean the feature tensor
        feature_tensor = np.nan_to_num(
            feature_tensor,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )        
        return feature_tensor
    
    def build_observation(self):
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        market_data = self.feature_tensor[start_idx:end_idx, :, :].flatten().astype(np.float32)
        weights = self.current_weights.flatten().astype(np.float32)
        obs = np.concatenate([market_data, weights]).astype(np.float32)
        return obs
    
    def cal_reward(self, new_value, old_value, turnover):
        '''SIMPLE RAW RETURN REWARD - BACK TO BASICS
        
        Lesson: Volatility adjustment was WRONG in bear markets.
        - Higher vol = suppressed reward signal
        - But in bear markets we NEED strong learning signal!
        
        Fix: Use raw return rewards scaled 100x
        - Daily return of +0.1% → reward +0.01 (after scaling)
        - Daily return of -0.1% → reward -0.01
        - Turnover cost: 1% turnover → -1.0 reward
        - CLEAR GRADIENTS in all market conditions
        '''
        pct_return = (new_value / (old_value + 1e-6)) - 1
        reward = pct_return * 100.0  # Strong scaling for clear signal
        turnover_penalty = turnover * 1.0  # Simple 1:1 penalty
        reward -= turnover_penalty
        return float(reward)
    
    def step(self, action):
        raw = np.array(action, dtype=np.float32).flatten()
        exp = np.exp(raw - np.max(raw))
        target_weights = exp / (exp.sum() + 1e-10)

        daily_returns = self.returns[self.current_step]
        portfolio_return = np.dot(target_weights, daily_returns)

        weight_changes = np.abs(target_weights - self.current_weights)
        turnover = np.sum(weight_changes)
        cost_deduction = turnover * self.transaction_cost

        new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
        new_portfolio_value *= (1 - cost_deduction)

        reward = self.cal_reward(new_portfolio_value, self.portfolio_value, turnover)

        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_value = new_portfolio_value
        self.current_weights = target_weights.copy()
        self.portfolio_history.append(new_portfolio_value)
        self.current_step += 1

        terminated = self.current_step >= len(self.price_data) - 1
        truncated = False
        obs = self.build_observation()
        info = {
            "date": self.dates[self.current_step] if not terminated else self.dates[-1],
            "portfolio_value": float(self.portfolio_value),
            "weights": self.current_weights,
            "turnover": turnover
        }
        return (obs, reward, terminated, truncated, info)