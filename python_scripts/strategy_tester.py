#!/usr/bin/env python
"""
Strategy Tester

This module provides functionality to test trading strategies with simulated market data.
It calculates performance metrics similar to the test_optimized_strategy.py script.
"""
import json
import random
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

class StrategyTester:
    """Strategy Tester class for evaluating trading strategies"""
    
    def __init__(self, start_date='2024-01-01', end_date='2025-03-06', 
                 symbols=None, timeframes=None):
        """
        Initialize the strategy tester
        
        Args:
            start_date: Start date for testing
            end_date: End date for testing
            symbols: List of symbols to test on (default: ['EURUSD'])
            timeframes: List of timeframes to test on (default: ['H1'])
        """
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols or ['EURUSD']
        self.timeframes = timeframes or ['H1']
        
        # Cache for performance data to avoid repeatedly generating the same data
        self._performance_cache = {}
    
    def test_strategy(self, strategy_type, parameters):
        """
        Test strategy with given parameters
        
        Args:
            strategy_type: Type of strategy ('ml_strategy' or 'medallion_strategy')
            parameters: Dictionary of parameters for the strategy
            
        Returns:
            dict: Performance metrics
        """
        # For real implementation, we would test on all symbols and timeframes
        # For simplicity, we'll use a random symbol and timeframe from our lists
        symbol = random.choice(self.symbols)
        timeframe = random.choice(self.timeframes)
        
        print(f"Testing {strategy_type} on {symbol} {timeframe}")
        print(f"Period: {self.start_date} to {self.end_date}")
        
        # For demonstration, we'll simulate performance similar to test_optimized_strategy.py
        # In a real implementation, this would run actual backtests
        
        # Create a unique cache key for this configuration
        cache_key = f"{strategy_type}_{symbol}_{timeframe}_{hash(frozenset(parameters.items()))}"
        
        # Check if we have cached results
        if cache_key in self._performance_cache:
            return self._performance_cache[cache_key]
        
        # Simulate a trading equity curve and metrics
        days = (datetime.datetime.strptime(self.end_date, '%Y-%m-%d') - 
                datetime.datetime.strptime(self.start_date, '%Y-%m-%d')).days
        
        # Generate dates for the equity curve
        dates = [datetime.datetime.strptime(self.start_date, '%Y-%m-%d') + 
                datetime.timedelta(days=i) for i in range(days)]
        
        # Parameters influence performance in different ways depending on strategy type
        if strategy_type == 'ml_strategy':
            performance_metrics = self._simulate_ml_strategy(parameters, days, dates)
        else:
            performance_metrics = self._simulate_medallion_strategy(parameters, days, dates)
        
        # Cache the results
        self._performance_cache[cache_key] = performance_metrics
        
        return performance_metrics
    
    def _simulate_ml_strategy(self, parameters, days, dates):
        """
        Simulate ML strategy performance
        
        Args:
            parameters: Strategy parameters
            days: Number of days in the simulation
            dates: List of dates
            
        Returns:
            dict: Performance metrics
        """
        # Extract key parameters that influence performance
        model_type = parameters.get('model_type', 'randomforest')
        lookback = parameters.get('lookback_periods', 20)
        prediction_horizon = parameters.get('prediction_horizon', 5)
        stop_loss = parameters.get('stop_loss_pct', 2.0)
        take_profit = parameters.get('take_profit_pct', 4.0)
        confidence_threshold = parameters.get('confidence_threshold', 0.7)
        
        # Model type factor - different models have different strengths
        model_factor = {
            'xgboost': 1.0 + random.uniform(-0.1, 0.2),
            'randomforest': 0.9 + random.uniform(-0.1, 0.2),
            'linear': 0.7 + random.uniform(-0.1, 0.2)
        }.get(model_type, 1.0)
        
        # Lookback period factor - optimal depends on market conditions
        lookback_factor = 1.0
        if lookback < 30:
            lookback_factor = 0.8 + (lookback / 30) * 0.4  # Better for short-term
        elif lookback > 70:
            lookback_factor = 0.9 + min(1.0, (lookback - 70) / 30) * 0.2  # Better for long-term trends
        else:
            lookback_factor = 1.0 + random.uniform(-0.1, 0.1)  # Middle range is generally good
        
        # Prediction horizon factor
        horizon_factor = 0.9 + min(1.0, prediction_horizon / 10) * 0.2
        
        # Risk-reward ratio
        rr_ratio = take_profit / stop_loss
        rr_factor = 0.5 + min(1.5, rr_ratio) / 3.0
        
        # Confidence threshold
        confidence_factor = 0.8 + min(1.0, confidence_threshold * 0.5)
        
        # Combined performance factor
        base_performance = model_factor * lookback_factor * horizon_factor * rr_factor * confidence_factor
        
        # Adjust for randomness - ML strategies can be inconsistent
        # Higher confidence threshold reduces randomness
        randomness = random.uniform(0.8, 1.2) * (1.1 - confidence_threshold * 0.2)
        
        performance_factor = base_performance * randomness
        
        # Simulate equity curve and calculate metrics
        equity, daily_returns = self._generate_equity_curve(days, performance_factor)
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity, daily_returns, days)
        
        # ML strategies typically have lower win rates but higher R:R
        metrics['win_rate'] = max(30, min(75, metrics['win_rate'] * random.uniform(0.85, 1.15)))
        
        return metrics
    
    def _simulate_medallion_strategy(self, parameters, days, dates):
        """
        Simulate Medallion strategy performance
        
        Args:
            parameters: Strategy parameters
            days: Number of days in the simulation
            dates: List of dates
            
        Returns:
            dict: Performance metrics
        """
        # Extract key parameters that influence performance
        fast_ma = parameters.get('fast_ma_periods', 20)
        slow_ma = parameters.get('slow_ma_periods', 50)
        rsi_periods = parameters.get('rsi_periods', 14)
        rsi_overbought = parameters.get('rsi_overbought', 70)
        rsi_oversold = parameters.get('rsi_oversold', 30)
        volatility_factor = parameters.get('volatility_factor', 1.0)
        stop_loss = parameters.get('stop_loss_pct', 1.0)
        take_profit = parameters.get('take_profit_pct', 2.0)
        
        # Calculate MA ratio - optimal is often between 3-4x
        ma_ratio = slow_ma / max(1, fast_ma)
        if 2.5 <= ma_ratio <= 4.5:
            ma_factor = 1.0 + (1.0 - abs(ma_ratio - 3.5) / 3.5) * 0.3
        else:
            ma_factor = 0.9 - min(0.4, abs(ma_ratio - 3.5) / 10.0)
        
        # RSI settings factor
        rsi_range = rsi_overbought - rsi_oversold
        if 30 <= rsi_range <= 50:
            rsi_factor = 1.0 + (1.0 - abs(rsi_range - 40) / 40) * 0.2
        else:
            rsi_factor = 0.9 - min(0.3, abs(rsi_range - 40) / 60)
        
        # RSI periods factor
        if 10 <= rsi_periods <= 20:
            rsi_periods_factor = 1.0 + (1.0 - abs(rsi_periods - 14) / 14) * 0.1
        else:
            rsi_periods_factor = 0.95
        
        # Volatility factor - optimal depends on market conditions
        volatility_factor_score = 0.9 + min(0.3, volatility_factor / 3.0)
        
        # Risk-reward ratio
        rr_ratio = take_profit / max(0.1, stop_loss)
        if 1.5 <= rr_ratio <= 4.0:
            rr_factor = 0.9 + min(0.3, rr_ratio / 4.0)
        else:
            rr_factor = 0.8
        
        # Golden ratio proximity (1.618)
        golden_ratio = 1.618
        golden_ratio_score = 1.0 - min(0.2, abs(rr_ratio - golden_ratio) / golden_ratio)
        
        # Combined performance factor
        base_performance = (
            ma_factor * 
            rsi_factor * 
            rsi_periods_factor * 
            volatility_factor_score * 
            rr_factor *
            golden_ratio_score
        )
        
        # Medallion strategies are more consistent, less random
        randomness = random.uniform(0.9, 1.1)
        
        performance_factor = base_performance * randomness
        
        # Simulate equity curve and calculate metrics
        equity, daily_returns = self._generate_equity_curve(days, performance_factor)
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity, daily_returns, days)
        
        # Medallion strategies typically have higher win rates but lower R:R
        metrics['win_rate'] = max(40, min(85, metrics['win_rate'] * random.uniform(0.95, 1.05)))
        
        return metrics
    
    def _generate_equity_curve(self, days, performance_factor, initial_equity=10000):
        """
        Generate simulated equity curve based on performance factor
        
        Args:
            days: Number of days
            performance_factor: Factor influencing performance
            initial_equity: Starting equity
            
        Returns:
            tuple: (equity_curve, daily_returns)
        """
        # Start with initial equity
        equity = [initial_equity]
        daily_returns = []
        
        # Define market regimes similar to test_optimized_strategy.py
        market_regimes = [
            # Q1 2024 - Mixed market
            {'start': 0, 'end': 90, 'trend': 0.0002, 'volatility': 1.0},
            # Q2 2024 - Bullish trend
            {'start': 90, 'end': 180, 'trend': 0.001, 'volatility': 0.9},
            # Q3 2024 - High volatility
            {'start': 180, 'end': 270, 'trend': 0.0003, 'volatility': 1.4},
            # Q4 2024 - Year-end rally
            {'start': 270, 'end': 360, 'trend': 0.0008, 'volatility': 0.8},
            # Q1 2025 - Correction
            {'start': 360, 'end': 999, 'trend': -0.0004, 'volatility': 1.2}
        ]
        
        # Create daily returns based on market regimes
        for i in range(1, days):
            # Find current market regime
            regime = next((r for r in market_regimes if r['start'] <= i < r['end']), 
                          {'trend': 0.0, 'volatility': 1.0})
            
            # Base daily return with trend component
            base_return = regime['trend'] * performance_factor
            
            # Add random component scaled by volatility
            random_component = np.random.normal(0, 0.005 * regime['volatility'])
            
            # Combine for final daily return
            daily_return = base_return + random_component
            
            # Occasional larger drawdowns (market shocks)
            if np.random.random() < 0.01:  # 1% chance of a bad day
                daily_return -= 0.02 * regime['volatility']
                
            daily_returns.append(daily_return)
            
            # Update equity
            new_equity = equity[-1] * (1 + daily_return)
            equity.append(new_equity)
        
        return equity, daily_returns
    
    def _calculate_metrics(self, equity, daily_returns, days, num_trades=None):
        """
        Calculate performance metrics from equity curve
        
        Args:
            equity: List of equity values
            daily_returns: List of daily returns
            days: Number of days
            num_trades: Optional number of trades (if None, will be estimated)
            
        Returns:
            dict: Performance metrics
        """
        # Calculate basic metrics
        final_equity = equity[-1]
        initial_equity = equity[0]
        max_equity = max(equity)
        
        # Find the lowest equity after reaching the current max
        max_equity_idx = equity.index(max_equity)
        min_equity_after_max = min(equity[max_equity_idx:])
        
        # Calculate drawdown from peak
        drawdown_from_peak = (max_equity - min_equity_after_max) / max_equity * 100
        
        # Calculate other metrics
        total_return_pct = (final_equity / initial_equity - 1) * 100
        annual_return = total_return_pct * 365 / days
        
        # Calculate Sharpe ratio
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        
        # Estimate number of trades if not provided
        if num_trades is None:
            # Assuming approximately one trade every 3-7 days on average
            avg_trades_per_day = random.uniform(1/7, 1/3)
            num_trades = int(days * avg_trades_per_day)
        
        # Estimate win rate based on performance
        base_win_rate = 40 + total_return_pct / 2
        win_rate = max(20, min(80, base_win_rate))
        
        # Number of winning and losing trades
        num_winners = int(num_trades * (win_rate / 100))
        num_losers = num_trades - num_winners
        
        # Estimate profit factor
        if num_losers > 0:
            avg_win_amount = (total_return_pct / 100 * initial_equity + abs(drawdown_from_peak / 100 * max_equity)) / num_winners
            avg_loss_amount = abs(drawdown_from_peak / 100 * max_equity) / num_losers
            profit_factor = (avg_win_amount * num_winners) / (avg_loss_amount * num_losers) if avg_loss_amount * num_losers != 0 else 999
        else:
            profit_factor = 999  # Arbitrary high number for no losers
        
        # Compile metrics
        metrics = {
            'win_rate': round(win_rate, 2),
            'annual_return': round(annual_return, 2),
            'max_drawdown': round(drawdown_from_peak, 2),
            'total_return': round(total_return_pct, 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'total_trades': num_trades
        }
        
        return metrics

# Test the tester directly if run as script
if __name__ == '__main__':
    # Test ML strategy
    ml_params = {
        'lookback_periods': 44,
        'prediction_horizon': 5,
        'model_type': 'xgboost',
        'feature_selection': 'pca',
        'stop_loss_pct': 2.95,
        'take_profit_pct': 1.46,
        'risk_per_trade_pct': 1.13,
        'confidence_threshold': 0.57
    }
    
    # Test Medallion strategy
    medallion_params = {
        'fast_ma_periods': 26,
        'slow_ma_periods': 142,
        'rsi_periods': 16,
        'rsi_overbought': 79,
        'rsi_oversold': 31,
        'volatility_factor': 1.7099,
        'stop_loss_pct': 0.7367,
        'take_profit_pct': 2.325,
        'risk_per_trade_pct': 0.9979
    }
    
    tester = StrategyTester()
    
    print("Testing ML Strategy:")
    ml_metrics = tester.test_strategy('ml_strategy', ml_params)
    print(json.dumps(ml_metrics, indent=2))
    
    print("\nTesting Medallion Strategy:")
    medallion_metrics = tester.test_strategy('medallion_strategy', medallion_params)
    print(json.dumps(medallion_metrics, indent=2)) 