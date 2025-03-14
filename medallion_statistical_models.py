"""
Medallion-Inspired Strategy: Statistical Models

This module implements the statistical models used in the Medallion-inspired
trading strategy, including:
- Stochastic process models
- Statistical arbitrage techniques
- Mean reversion models 
- Market inefficiency detectors
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logger = logging.getLogger("medallion_strategy.statistical_models")

class StochasticProcessModels:
    """
    Implements various stochastic process models for price time series
    based on mathematical formulations from quantitative finance.
    """
    
    @staticmethod
    def fit_geometric_brownian_motion(prices: np.ndarray, dt: float = 1.0) -> Dict[str, float]:
        """
        Fit a Geometric Brownian Motion model to price data
        
        dS/S = μdt + σdW
        
        Args:
            prices: Array of price values
            dt: Time step (1.0 for daily data)
            
        Returns:
            Dict containing mu (drift) and sigma (volatility) parameters
        """
        returns = np.diff(np.log(prices))
        mu = np.mean(returns) / dt
        sigma = np.std(returns) / np.sqrt(dt)
        
        return {
            'mu': mu,
            'sigma': sigma,
            'annualized_mu': mu * 252,  # Assuming daily data
            'annualized_sigma': sigma * np.sqrt(252)  # Assuming daily data
        }
    
    @staticmethod
    def fit_ornstein_uhlenbeck(prices: np.ndarray, dt: float = 1.0) -> Dict[str, float]:
        """
        Fit an Ornstein-Uhlenbeck (mean-reverting) process to price data
        
        dx = θ(μ - x)dt + σdW
        
        Args:
            prices: Array of price values
            dt: Time step (1.0 for daily data)
            
        Returns:
            Dict containing theta (reversion speed), mu (mean), and sigma parameters
        """
        x = np.array(prices)
        x_lag = x[:-1]
        x_diff = np.diff(x)
        
        # Fit the model dx = a + b*x
        model = sm.OLS(x_diff, sm.add_constant(x_lag))
        results = model.fit()
        
        a, b = results.params
        
        # Convert to O-U parameters
        theta = -b  # Mean reversion rate
        mu = a / theta  # Long-term mean
        sigma = np.std(results.resid) / np.sqrt(dt)  # Volatility
        
        half_life = np.log(2) / theta if theta > 0 else float('inf')
        
        return {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'half_life': half_life,
            'is_mean_reverting': theta > 0
        }
    
    @staticmethod
    def fit_heston_model(prices: np.ndarray, dt: float = 1.0) -> Dict[str, float]:
        """
        Simplified fit of Heston stochastic volatility model
        
        dS/S = μdt + √v dW₁
        dv = κ(θ - v)dt + σ√v dW₂
        
        Where v is the variance, κ is mean reversion speed of variance,
        θ is long-term variance, and σ is volatility of variance.
        
        Args:
            prices: Array of price values
            dt: Time step
            
        Returns:
            Dict containing model parameters
        """
        log_returns = np.diff(np.log(prices))
        log_returns_squared = log_returns ** 2
        
        # Estimate variance process parameters
        var_lag = log_returns_squared[:-1]
        var_diff = np.diff(log_returns_squared)
        
        try:
            # Fit the model dv = a + b*v
            model = sm.OLS(var_diff, sm.add_constant(var_lag))
            results = model.fit()
            
            a, b = results.params
            
            # Convert to Heston parameters
            kappa = -b  # Mean reversion speed
            theta = a / kappa if kappa > 0 else np.mean(log_returns_squared)  # Long-term variance
            sigma_v = np.std(results.resid)  # Volatility of variance
            
            # Estimate drift
            mu = np.mean(log_returns) / dt
            
            return {
                'mu': mu,
                'kappa': kappa if kappa > 0 else 0.1,  # Fallback if not mean-reverting
                'theta': theta,
                'sigma_v': sigma_v,
                'annualized_mu': mu * 252,  # Assuming daily data
                'is_mean_reverting': kappa > 0
            }
        except:
            # Fallback to simpler estimation
            mu = np.mean(log_returns) / dt
            theta = np.mean(log_returns_squared)
            
            return {
                'mu': mu,
                'kappa': 0.1,  # Default value
                'theta': theta,
                'sigma_v': np.std(log_returns_squared),
                'annualized_mu': mu * 252  # Assuming daily data
            }
    
    @staticmethod
    def estimate_drift_and_volatility(
        prices: np.ndarray, 
        window: int = 50,
        method: str = 'rolling'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate time-varying drift and volatility components from price data
        
        Args:
            prices: Array of price values
            window: Window size for estimation
            method: 'rolling' or 'ewm' (exponentially weighted)
            
        Returns:
            Tuple of (drift, volatility) arrays
        """
        log_prices = np.log(prices)
        log_returns = np.diff(log_prices)
        
        if method == 'rolling':
            # Rolling window estimation
            drift = np.zeros_like(log_returns)
            volatility = np.zeros_like(log_returns)
            
            for i in range(window, len(log_returns) + 1):
                window_returns = log_returns[i-window:i]
                drift[i-1] = np.mean(window_returns)
                volatility[i-1] = np.std(window_returns)
                
        else:  # 'ewm'
            # Exponentially weighted estimation
            alpha = 2 / (window + 1)
            
            drift = pd.Series(log_returns).ewm(alpha=alpha).mean().values
            volatility = pd.Series(log_returns).ewm(alpha=alpha).std().values
        
        # Pad initial values
        drift[:window-1] = drift[window-1]
        volatility[:window-1] = volatility[window-1]
        
        return drift, volatility
    
    @staticmethod
    def simulate_stochastic_process(
        model_params: Dict[str, float], 
        steps: int = 252, 
        initial_price: float = 100.0,
        process_type: str = 'gbm',
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate a stochastic process using fitted parameters
        
        Args:
            model_params: Dictionary of model parameters
            steps: Number of steps to simulate
            initial_price: Initial price
            process_type: 'gbm', 'ou', or 'heston'
            random_seed: Random seed for reproducibility
            
        Returns:
            Array of simulated prices
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        dt = 1.0  # Time step
        prices = np.zeros(steps + 1)
        prices[0] = initial_price
        
        if process_type == 'gbm':
            # Geometric Brownian Motion
            mu = model_params.get('mu', 0.0)
            sigma = model_params.get('sigma', 0.1)
            
            for t in range(1, steps + 1):
                dW = np.random.normal(0, np.sqrt(dt))
                prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
                
        elif process_type == 'ou':
            # Ornstein-Uhlenbeck process
            theta = model_params.get('theta', 0.1)
            mu = model_params.get('mu', initial_price)
            sigma = model_params.get('sigma', 0.1)
            
            x = np.log(initial_price)
            for t in range(1, steps + 1):
                dW = np.random.normal(0, np.sqrt(dt))
                x = x + theta * (mu - x) * dt + sigma * dW
                prices[t] = np.exp(x)
                
        elif process_type == 'heston':
            # Heston stochastic volatility model
            mu = model_params.get('mu', 0.0)
            kappa = model_params.get('kappa', 0.1)
            theta = model_params.get('theta', 0.04)  # Long-term variance
            sigma_v = model_params.get('sigma_v', 0.2)
            
            # Initial variance
            v = model_params.get('theta', 0.04)
            rho = model_params.get('rho', -0.7)  # Correlation between price and variance
            
            for t in range(1, steps + 1):
                # Correlated Brownian motions
                z1 = np.random.normal(0, 1)
                z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1)
                
                # Update variance (ensure it stays positive)
                v_new = v + kappa * (theta - v) * dt + sigma_v * np.sqrt(max(v, 0)) * np.sqrt(dt) * z2
                v = max(v_new, 0.0001)  # Ensure variance stays positive
                
                # Update price
                prices[t] = prices[t-1] * np.exp((mu - 0.5 * v) * dt + np.sqrt(v) * np.sqrt(dt) * z1)
        
        return prices


class StatisticalArbitrage:
    """
    Implements statistical arbitrage techniques including pairs trading,
    mean reversion strategies, and cointegration-based approaches.
    """
    
    @staticmethod
    def test_stationarity(series, alpha=0.05):
        """
        Test for stationarity using Augmented Dickey-Fuller test
        
        Args:
            series: Price series to test
            alpha: Significance level
            
        Returns:
            Dict with test results including p-value and stationarity boolean
        """
        result = adfuller(series)
        p_value = result[1]
        is_stationary = p_value < alpha
        
        return {
            'p_value': p_value,
            'test_statistic': result[0],
            'is_stationary': is_stationary,
            'critical_values': result[4]
        }
    
    @staticmethod
    def test_cointegration(x, y, alpha=0.05):
        """
        Test for cointegration between two price series
        
        Args:
            x: First price series
            y: Second price series
            alpha: Significance level
            
        Returns:
            Dict with test results including p-value and cointegration boolean
        """
        result = coint(x, y)
        p_value = result[1]
        is_cointegrated = p_value < alpha
        
        return {
            'p_value': p_value,
            'test_statistic': result[0],
            'is_cointegrated': is_cointegrated,
            'critical_values': result[2]
        }
    
    @staticmethod
    def calculate_hedge_ratio(x, y):
        """
        Calculate optimal hedge ratio between two assets
        
        Args:
            x: First price series
            y: Second price series
            
        Returns:
            Hedge ratio (beta)
        """
        # Add constant to X for OLS
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        beta = model.params[1]
        
        return beta
    
    @staticmethod
    def calculate_spread(x, y, beta=None):
        """
        Calculate spread between two price series
        
        Args:
            x: First price series
            y: Second price series
            beta: Hedge ratio (calculated if None)
            
        Returns:
            Spread series
        """
        if beta is None:
            beta = StatisticalArbitrage.calculate_hedge_ratio(x, y)
            
        spread = y - beta * x
        return spread
    
    @staticmethod
    def calculate_zscore(spread, window=20):
        """
        Calculate z-score of spread
        
        Args:
            spread: Spread series
            window: Window for rolling statistics
            
        Returns:
            Z-score series
        """
        mean = spread.rolling(window=window).mean()
        std = spread.rolling(window=window).std()
        zscore = (spread - mean) / std
        
        return zscore
    
    @staticmethod
    def generate_pair_trading_signals(zscore, entry_threshold=2.0, exit_threshold=0.5):
        """
        Generate trading signals based on spread z-score
        
        Args:
            zscore: Z-score of spread
            entry_threshold: Z-score threshold for trade entry
            exit_threshold: Z-score threshold for trade exit
            
        Returns:
            DataFrame with signals
        """
        signals = pd.Series(index=zscore.index, data=np.zeros(len(zscore)))
        
        # 1 for long spread, -1 for short spread, 0 for no position
        long_spread = False
        short_spread = False
        
        for i in range(len(zscore)):
            if not long_spread and not short_spread:
                # No position
                if zscore.iloc[i] < -entry_threshold:
                    # Spread is below lower threshold, go long
                    signals.iloc[i] = 1
                    long_spread = True
                elif zscore.iloc[i] > entry_threshold:
                    # Spread is above upper threshold, go short
                    signals.iloc[i] = -1
                    short_spread = True
            elif long_spread:
                # Currently long spread
                if zscore.iloc[i] > -exit_threshold:
                    # Spread has mean-reverted, exit
                    signals.iloc[i] = 0
                    long_spread = False
                else:
                    # Maintain long position
                    signals.iloc[i] = 1
            elif short_spread:
                # Currently short spread
                if zscore.iloc[i] < exit_threshold:
                    # Spread has mean-reverted, exit
                    signals.iloc[i] = 0
                    short_spread = False
                else:
                    # Maintain short position
                    signals.iloc[i] = -1
        
        return signals
    
    @staticmethod
    def half_life(spread):
        """
        Calculate half-life of mean reversion
        
        Args:
            spread: Spread series
            
        Returns:
            Half-life in periods
        """
        spread_lag = spread.shift(1)
        spread_diff = spread.diff()
        
        # Drop NAs
        spread_lag = spread_lag.dropna()
        spread_diff = spread_diff.dropna()
        
        # Ensure we have matching indices
        spread_lag = spread_lag.loc[spread_diff.index]
        
        # Fit linear regression model
        model = sm.OLS(spread_diff, spread_lag)
        result = model.fit()
        
        # Calculate half-life
        beta = result.params[0]
        half_life = -np.log(2) / beta if beta < 0 else float('inf')
        
        return half_life
    
    def find_cointegrated_pairs(self, price_dict, p_value_threshold=0.05):
        """
        Find cointegrated pairs from a dictionary of price series
        
        Args:
            price_dict: Dictionary of price series with keys as symbols
            p_value_threshold: Threshold for cointegration test
            
        Returns:
            List of tuples (symbol1, symbol2, p_value, hedge_ratio)
        """
        n = len(price_dict)
        symbols = list(price_dict.keys())
        pairs = []
        
        # Initialize progress tracking
        total_tests = n * (n - 1) // 2
        tests_done = 0
        
        logger.info(f"Testing {total_tests} pairs for cointegration")
        
        for i in range(n):
            for j in range(i+1, n):
                symbol1 = symbols[i]
                symbol2 = symbols[j]
                
                series1 = price_dict[symbol1]
                series2 = price_dict[symbol2]
                
                # Ensure series are of the same length
                common_idx = series1.index.intersection(series2.index)
                if len(common_idx) < 100:  # Need enough data
                    logger.warning(f"Not enough common data for {symbol1}/{symbol2}")
                    continue
                    
                series1 = series1.loc[common_idx]
                series2 = series2.loc[common_idx]
                
                # Test for cointegration
                result = self.test_cointegration(series1, series2)
                
                tests_done += 1
                if tests_done % 100 == 0:
                    logger.info(f"Completed {tests_done}/{total_tests} cointegration tests")
                
                if result['is_cointegrated']:
                    hedge_ratio = self.calculate_hedge_ratio(series1, series2)
                    half_life_value = self.half_life(
                        self.calculate_spread(series1, series2, hedge_ratio)
                    )
                    
                    # Only include pairs with reasonable half-life
                    if 1 <= half_life_value <= 100:
                        pairs.append((
                            symbol1, 
                            symbol2, 
                            result['p_value'], 
                            hedge_ratio,
                            half_life_value
                        ))
                        logger.info(f"Found cointegrated pair: {symbol1}/{symbol2} with half-life: {half_life_value:.2f}")
        
        # Sort by p-value (lowest first)
        pairs.sort(key=lambda x: x[2])
        
        logger.info(f"Found {len(pairs)} cointegrated pairs out of {total_tests} tests")
        return pairs
    
    @staticmethod
    def kalman_filter_regression(x, y):
        """
        Use Kalman filter to estimate time-varying hedge ratio
        
        Args:
            x: First price series
            y: Second price series
            
        Returns:
            DataFrame with estimated parameters over time
        """
        # Initial state
        delta = 1e-3
        trans_cov = delta / (1 - delta) * np.eye(2)  # State transition cov
        
        # These are for the observation equation
        obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
        
        # Initialize state and covariance
        state_mean = np.zeros(2)  # Initial state mean [beta, alpha]
        state_cov = np.ones((2, 2))  # Initial state covariance
        
        # Arrays to store states
        beta = np.zeros(len(y))
        alpha = np.zeros(len(y))
        
        for i in range(len(y)):
            # Predict
            state_mean_prior = state_mean
            state_cov_prior = state_cov + trans_cov
            
            # Update
            observation = y[i]
            obs_mat_i = obs_mat[i]
            kalman_gain = state_cov_prior @ obs_mat_i @ np.linalg.inv(
                obs_mat_i.T @ state_cov_prior @ obs_mat_i + 1.0)
            
            state_mean = state_mean_prior + kalman_gain * (
                observation - obs_mat_i.T @ state_mean_prior)
            state_cov = state_cov_prior - kalman_gain @ obs_mat_i.T @ state_cov_prior
            
            # Save states
            beta[i] = state_mean[0]
            alpha[i] = state_mean[1]
        
        return pd.DataFrame({'beta': beta, 'alpha': alpha}, index=x.index)


class MarketInefficiencyDetector:
    """
    Detects market inefficiencies and opportunities for statistical arbitrage
    based on mathematical models and tests.
    """
    
    def __init__(self, config=None):
        """
        Initialize the market inefficiency detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = {
            'mean_reversion_threshold': 2.0,
            'volatility_window': 20,
            'correlation_threshold': 0.7,
            'min_half_life': 1,
            'max_half_life': 50,
            'min_data_points': 100,
            'zscore_window': 20,
            'stationarity_pvalue': 0.05
        }
        
        if config:
            self.config.update(config)
            
        self.stochastic_models = StochasticProcessModels()
        self.stat_arb = StatisticalArbitrage()
        
    def analyze_mean_reversion(self, prices, window=None):
        """
        Analyze a price series for mean-reversion characteristics
        
        Args:
            prices: Price series
            window: Analysis window (default from config)
            
        Returns:
            Dict of analysis results
        """
        if window is None:
            window = self.config['zscore_window']
            
        # Test for stationarity
        stationarity = self.stat_arb.test_stationarity(prices)
        
        # Fit Ornstein-Uhlenbeck process
        ou_params = self.stochastic_models.fit_ornstein_uhlenbeck(prices)
        
        # Calculate z-score
        mean = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        zscore = (prices - mean) / std
        
        # Recent z-score
        recent_zscore = zscore.iloc[-1] if not zscore.empty else 0
        
        # Calculate half-life
        half_life = ou_params['half_life']
        
        # Forecast mean reversion
        current_price = prices.iloc[-1]
        target_price = ou_params['mu']
        expected_reversal_pct = (target_price - current_price) / current_price * 100
        
        return {
            'is_mean_reverting': ou_params['is_mean_reverting'],
            'is_stationary': stationarity['is_stationary'],
            'half_life': half_life,
            'current_zscore': recent_zscore,
            'target_price': target_price,
            'expected_reversal_pct': expected_reversal_pct,
            'ou_params': ou_params,
            'stationarity': stationarity,
            'signal': self._generate_mean_reversion_signal(recent_zscore, half_life)
        }
    
    def _generate_mean_reversion_signal(self, zscore, half_life):
        """
        Generate a mean reversion signal based on z-score and half-life
        
        Args:
            zscore: Current z-score
            half_life: Mean reversion half-life
            
        Returns:
            Dict with signal details
        """
        # Default neutral signal
        signal = {
            'direction': 0,
            'strength': 0,
            'confidence': 0
        }
        
        # Check if mean reversion parameters are reasonable
        if half_life < self.config['min_half_life'] or half_life > self.config['max_half_life']:
            return signal
        
        # Generate signal based on z-score
        threshold = self.config['mean_reversion_threshold']
        
        if zscore < -threshold:
            # Price is below expected range, expect upward reversion
            signal['direction'] = 1  # Buy signal
            signal['strength'] = min(1.0, abs(zscore) / (2 * threshold))
            signal['confidence'] = 1.0 - (half_life / self.config['max_half_life'])
            
        elif zscore > threshold:
            # Price is above expected range, expect downward reversion
            signal['direction'] = -1  # Sell signal
            signal['strength'] = min(1.0, abs(zscore) / (2 * threshold))
            signal['confidence'] = 1.0 - (half_life / self.config['max_half_life'])
        
        return signal
    
    def detect_anomalies(self, prices, window=None):
        """
        Detect price anomalies using statistical methods
        
        Args:
            prices: Price series
            window: Analysis window
            
        Returns:
            Dict with anomaly detection results
        """
        if window is None:
            window = self.config['volatility_window']
            
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Calculate rolling mean and std of returns
        mean = returns.rolling(window=window).mean()
        std = returns.rolling(window=window).std()
        
        # Calculate z-scores
        zscores = (returns - mean) / std
        
        # Identify anomalies (returns that are more than 3 standard deviations from mean)
        anomalies = zscores.abs() > 3
        recent_anomalies = anomalies.iloc[-5:].any() if len(anomalies) >= 5 else False
        
        # Calculate recent volatility
        recent_volatility = std.iloc[-1] if not std.empty else 0
        historical_volatility = returns.std()
        
        # Detect volatility anomalies
        volatility_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1
        volatility_anomaly = volatility_ratio > 2.0  # Volatility is twice the historical average
        
        return {
            'recent_anomalies': recent_anomalies,
            'volatility_anomaly': volatility_anomaly,
            'volatility_ratio': volatility_ratio,
            'recent_volatility': recent_volatility,
            'zscores': zscores.iloc[-5:].tolist() if len(zscores) >= 5 else [],
            'anomaly_dates': prices.index[anomalies].tolist() if any(anomalies) else []
        }
    
    def analyze_drift_regime(self, prices, window=None):
        """
        Analyze the drift component of price movements to detect regime
        
        Args:
            prices: Price series
            window: Analysis window
            
        Returns:
            Dict with drift analysis results
        """
        if window is None:
            window = self.config['volatility_window']
            
        # Estimate drift and volatility components
        drift, volatility = self.stochastic_models.estimate_drift_and_volatility(
            prices.values, window=window
        )
        
        # Convert to annualized terms (assuming daily data)
        ann_factor = np.sqrt(252)
        ann_drift = drift[-1] * 252 if len(drift) > 0 else 0
        ann_volatility = volatility[-1] * ann_factor if len(volatility) > 0 else 0
        
        # Calculate Sharpe ratio of the drift
        sharpe = ann_drift / ann_volatility if ann_volatility > 0 else 0
        
        # Determine regime based on drift strength
        if abs(sharpe) < 0.5:
            regime = 'neutral'
        elif sharpe > 0.5:
            regime = 'bullish'
        else:
            regime = 'bearish'
            
        return {
            'regime': regime,
            'ann_drift': ann_drift,
            'ann_volatility': ann_volatility,
            'sharpe': sharpe,
            'drift_trend': np.mean(np.sign(drift[-min(20, len(drift)):]))
                if len(drift) > 0 else 0
        }
    
    def generate_statistical_signal(self, data):
        """
        Generate a trading signal based on statistical models
        
        Args:
            data: Dict with price data for multiple timeframes
            
        Returns:
            Dict with signal details
        """
        # Initialize signal
        signal = {
            'direction': 0,
            'strength': 0,
            'confidence': 0,
            'timeframes': {}
        }
        
        timeframe_weights = {
            'M15': 0.1,
            'H1': 0.3,
            'H4': 0.3,
            'D1': 0.3
        }
        
        # Generate signals for each timeframe
        for tf, prices in data.items():
            if len(prices) < self.config['min_data_points']:
                continue
                
            # Make sure we're using the close price
            if isinstance(prices, pd.DataFrame) and 'close' in prices.columns:
                price_series = prices['close']
            else:
                price_series = prices
                
            # Analyze mean reversion
            mr_analysis = self.analyze_mean_reversion(price_series)
            
            # Analyze drift regime
            drift_analysis = self.analyze_drift_regime(price_series)
            
            # Combine signals
            tf_signal = mr_analysis['signal']
            
            # Adjust for drift regime
            if drift_analysis['regime'] == 'bullish' and tf_signal['direction'] < 0:
                tf_signal['strength'] *= 0.5  # Reduce strength of sell signals in bullish regime
            elif drift_analysis['regime'] == 'bearish' and tf_signal['direction'] > 0:
                tf_signal['strength'] *= 0.5  # Reduce strength of buy signals in bearish regime
                
            # Store timeframe-specific signal
            signal['timeframes'][tf] = {
                'direction': tf_signal['direction'],
                'strength': tf_signal['strength'],
                'confidence': tf_signal['confidence'],
                'regime': drift_analysis['regime'],
                'zscore': mr_analysis['current_zscore'],
                'half_life': mr_analysis['half_life']
            }
            
            # Weight signal by timeframe
            weight = timeframe_weights.get(tf, 0.1)
            signal['direction'] += tf_signal['direction'] * weight * tf_signal['strength']
            
            if tf_signal['direction'] != 0:
                signal['strength'] += tf_signal['strength'] * weight
                signal['confidence'] += tf_signal['confidence'] * weight
        
        # Normalize strength and confidence
        total_weight = sum(timeframe_weights.get(tf, 0.1) for tf in data.keys() if tf in timeframe_weights)
        if total_weight > 0:
            signal['strength'] /= total_weight
            signal['confidence'] /= total_weight
        
        return signal


class StatisticalModelFactory:
    """
    Factory class for creating various statistical model instances
    based on configuration parameters.
    """
    
    @staticmethod
    def create_model(model_type, config=None):
        """
        Create a statistical model instance
        
        Args:
            model_type: Type of model to create
            config: Configuration parameters
            
        Returns:
            Model instance
        """
        if model_type == 'market_inefficiency':
            return MarketInefficiencyDetector(config)
        elif model_type == 'statistical_arbitrage':
            return StatisticalArbitrage()
        elif model_type == 'stochastic_process':
            return StochasticProcessModels()
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# For testing
if __name__ == "__main__":
    # Create some test data
    np.random.seed(42)
    n = 1000
    t = np.linspace(0, 4, n)
    
    # Create a mean-reverting series
    ou_process = 100 + 10 * np.sin(t) + 2 * np.random.randn(n).cumsum() / np.sqrt(n)
    
    # Create a trending series
    gbm_process = 100 * np.exp(0.0001 * np.arange(n) + 0.01 * np.random.randn(n).cumsum())
    
    # Test stochastic process models
    spm = StochasticProcessModels()
    ou_params = spm.fit_ornstein_uhlenbeck(ou_process)
    gbm_params = spm.fit_geometric_brownian_motion(gbm_process)
    
    print("Ornstein-Uhlenbeck parameters:")
    print(ou_params)
    print("\nGeometric Brownian Motion parameters:")
    print(gbm_params)
    
    # Test statistical arbitrage
    sa = StatisticalArbitrage()
    ou_stationary = sa.test_stationarity(ou_process)
    gbm_stationary = sa.test_stationarity(gbm_process)
    
    print("\nOU process stationarity:")
    print(ou_stationary)
    print("\nGBM process stationarity:")
    print(gbm_stationary)
    
    # Convert to pandas Series
    ou_series = pd.Series(ou_process)
    gbm_series = pd.Series(gbm_process)
    
    # Test market inefficiency detector
    detector = MarketInefficiencyDetector()
    ou_analysis = detector.analyze_mean_reversion(ou_series)
    gbm_analysis = detector.analyze_mean_reversion(gbm_series)
    
    print("\nOU process mean reversion analysis:")
    print(ou_analysis)
    print("\nGBM process mean reversion analysis:")
    print(gbm_analysis) 