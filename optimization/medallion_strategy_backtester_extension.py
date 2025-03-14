"""
Extension to the Medallion Strategy Backtester to support parameter optimization
This enhances the original backtester with methods for parameter-based strategy creation
"""
import os
import sys
import logging
from pathlib import Path
import datetime
import numpy as np

# Ensure that we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the original backtester
from backtest_medallion_strategy import MedallionStrategyBacktester

# Setup logging
logger = logging.getLogger('medallion_strategy_backtester_extension')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class EnhancedMedallionStrategyBacktester(MedallionStrategyBacktester):
    """
    Enhanced version of the Medallion Strategy Backtester with parameter optimization support
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the enhanced backtester with the same arguments as the original
        """
        super().__init__(**kwargs)
        logger.info("Initialized EnhancedMedallionStrategyBacktester")
    
    def prepare_strategy_with_params(self, params):
        """
        Prepare a strategy instance with the given parameters
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            Object: Configured strategy instance
        """
        try:
            # Import the strategy class and related components
            from medallion_strategy_core import MedallionStrategy
            
            # Extract key parameters for statistical models
            mean_reversion_lookback = params.get('mean_reversion_lookback', 20)
            trend_following_lookback = params.get('trend_following_lookback', 50)
            pattern_recognition_lookback = params.get('pattern_recognition_lookback', 10)
            volatility_lookback = params.get('volatility_lookback', 20)
            
            # Extract model weights
            mean_reversion_weight = params.get('mean_reversion_weight', 0.3)
            trend_following_weight = params.get('trend_following_weight', 0.3)
            pattern_recognition_weight = params.get('pattern_recognition_weight', 0.2)
            volatility_weight = params.get('volatility_weight', 0.2)
            
            # Extract technical indicator parameters
            fast_ma_period = params.get('fast_ma_period', 10)
            slow_ma_period = params.get('slow_ma_period', 30)
            rsi_period = params.get('rsi_period', 14)
            rsi_overbought = params.get('rsi_overbought', 70)
            rsi_oversold = params.get('rsi_oversold', 30)
            volatility_threshold = params.get('volatility_threshold', 1.0)
            
            # Extract mean reversion parameters
            zscore_threshold = params.get('zscore_threshold', 2.0)
            mean_reversion_threshold = params.get('mean_reversion_threshold', 1.0)
            
            # Extract trend following parameters
            trend_strength_threshold = params.get('trend_strength_threshold', 0.3)
            momentum_lookback = params.get('momentum_lookback', 10)
            momentum_threshold = params.get('momentum_threshold', 0.5)
            
            # Extract pattern recognition parameters
            pattern_confidence_threshold = params.get('pattern_confidence_threshold', 0.7)
            use_candlestick_patterns = params.get('use_candlestick_patterns', True)
            use_chart_patterns = params.get('use_chart_patterns', True)
            
            # Extract position sizing parameters
            base_risk_percent = params.get('base_risk_percent', 1.0)
            volatility_adjustment_factor = params.get('volatility_adjustment_factor', 1.0)
            max_position_size_percent = params.get('max_position_size_percent', 5.0)
            
            # Extract risk management parameters
            atr_multiplier = params.get('atr_multiplier', 2.0)
            fixed_stop_loss_pips = params.get('fixed_stop_loss_pips', 25)
            use_dynamic_stops = params.get('use_dynamic_stops', True)
            profit_factor_target = params.get('profit_factor_target', 1.5)
            
            # Extract trade management parameters
            target_profit_factor = params.get('target_profit_factor', 2.0)
            max_trades_per_day = params.get('max_trades_per_day', 5)
            max_correlation_threshold = params.get('max_correlation_threshold', 0.7)
            use_partial_take_profit = params.get('use_partial_take_profit', True)
            partial_take_profit_levels = params.get('partial_take_profit_levels', [0.33, 0.67])
            use_trailing_stop = params.get('use_trailing_stop', True)
            trailing_stop_activation_pct = params.get('trailing_stop_activation_pct', 1.0)
            
            # Extract timeframe weights
            primary_timeframe_weight = params.get('primary_timeframe_weight', 0.5)
            secondary_timeframe_weights = params.get('secondary_timeframe_weights', {"H4": 0.3, "D1": 0.2})
            
            # Create a configuration dictionary
            config = {
                # Statistical Models
                'statistical_models': {
                    'mean_reversion': {
                        'lookback': mean_reversion_lookback,
                        'weight': mean_reversion_weight,
                        'zscore_threshold': zscore_threshold,
                        'threshold': mean_reversion_threshold
                    },
                    'trend_following': {
                        'lookback': trend_following_lookback,
                        'weight': trend_following_weight,
                        'strength_threshold': trend_strength_threshold,
                        'momentum_lookback': momentum_lookback,
                        'momentum_threshold': momentum_threshold
                    },
                    'pattern_recognition': {
                        'lookback': pattern_recognition_lookback,
                        'weight': pattern_recognition_weight,
                        'confidence_threshold': pattern_confidence_threshold,
                        'use_candlestick_patterns': use_candlestick_patterns,
                        'use_chart_patterns': use_chart_patterns
                    },
                    'volatility': {
                        'lookback': volatility_lookback,
                        'weight': volatility_weight,
                        'threshold': volatility_threshold
                    }
                },
                
                # Technical Indicators
                'technical_indicators': {
                    'fast_ma_period': fast_ma_period,
                    'slow_ma_period': slow_ma_period,
                    'rsi_period': rsi_period,
                    'rsi_overbought': rsi_overbought,
                    'rsi_oversold': rsi_oversold
                },
                
                # Position Sizing
                'position_sizing': {
                    'base_risk_percent': base_risk_percent,
                    'volatility_adjustment_factor': volatility_adjustment_factor,
                    'max_position_size_percent': max_position_size_percent
                },
                
                # Risk Management
                'risk_management': {
                    'atr_multiplier': atr_multiplier,
                    'fixed_stop_loss_pips': fixed_stop_loss_pips,
                    'use_dynamic_stops': use_dynamic_stops,
                    'profit_factor_target': profit_factor_target
                },
                
                # Trade Management
                'trade_management': {
                    'target_profit_factor': target_profit_factor,
                    'max_trades_per_day': max_trades_per_day,
                    'max_correlation_threshold': max_correlation_threshold,
                    'use_partial_take_profit': use_partial_take_profit,
                    'partial_take_profit_levels': partial_take_profit_levels,
                    'use_trailing_stop': use_trailing_stop,
                    'trailing_stop_activation_pct': trailing_stop_activation_pct
                },
                
                # Timeframe Weights
                'timeframe_weights': {
                    self.primary_timeframe: primary_timeframe_weight
                }
            }
            
            # Add secondary timeframe weights
            for tf, weight in secondary_timeframe_weights.items():
                config['timeframe_weights'][tf] = weight
            
            # Create and initialize strategy
            strategy = MedallionStrategy(
                symbol=self.symbol,
                primary_timeframe=self.primary_timeframe,
                secondary_timeframes=self.secondary_timeframes,
                mt5_connector=self.connector,
                config=config
            )
            
            # Prepare data
            for timeframe, data in self.data.items():
                # Make a copy to avoid modifying the original
                strategy.data[timeframe] = data.copy()
            
            # Override signal generation with mock implementation for backtesting
            def mock_generate_signal(self_ref, current_time=None):
                # Simple implementation that generates signals based on moving averages and RSI
                import random
                
                # Get primary data
                if not hasattr(self_ref, 'data') or self_ref.primary_timeframe not in self_ref.data:
                    logger.warning("Strategy does not have data prepared")
                    return {'action': 'NONE', 'strength': 0}
                
                data = self_ref.data[self_ref.primary_timeframe]
                
                # Get current data point
                if current_time is None:
                    current_time = data.index[-1]
                
                # Find index of current time
                try:
                    idx = data.index.get_loc(current_time)
                except:
                    # Try with nearest match
                    try:
                        idx = data.index.get_indexer([current_time], method='nearest')[0]
                    except:
                        logger.warning(f"Cannot find index for time {current_time}")
                        return {'action': 'NONE', 'strength': 0}
                
                # Simple logic based on price action, MA and basic patterns
                if idx < 50:
                    return {'action': 'NONE', 'strength': 0}
                
                # Calculate indicators
                close_prices = data['close'].values
                high_prices = data['high'].values
                low_prices = data['low'].values
                open_prices = data['open'].values
                
                # Calculate MA crossover (shorter timeframe)
                ma10 = np.mean(close_prices[idx-10:idx])
                ma20 = np.mean(close_prices[idx-20:idx])
                
                # Price momentum (rate of change)
                roc = (close_prices[idx] / close_prices[idx-5] - 1) * 100
                
                # Simple trend detection
                uptrend = close_prices[idx] > close_prices[idx-10] > close_prices[idx-20]
                downtrend = close_prices[idx] < close_prices[idx-10] < close_prices[idx-20]
                
                # Determine action based on indicators
                action = 'NONE'
                strength = 0
                
                # 1. MA Crossover
                if ma10 > ma20:
                    action = 'BUY'
                    # Strength based on the distance between MAs
                    strength = min(0.9, (ma10 / ma20 - 1) * 30)
                elif ma10 < ma20:
                    action = 'SELL'
                    # Strength based on the distance between MAs
                    strength = min(0.9, (1 - ma10 / ma20) * 30)
                
                # 2. Adjust based on trend
                if action == 'BUY' and uptrend:
                    strength *= 1.3  # Enhance buy signal in uptrend
                elif action == 'SELL' and downtrend:
                    strength *= 1.3  # Enhance sell signal in downtrend
                
                # 3. Consider momentum
                if action == 'BUY' and roc > 0.5:  # Strong upward momentum
                    strength *= 1.2
                elif action == 'SELL' and roc < -0.5:  # Strong downward momentum
                    strength *= 1.2
                
                # 4. Detect potential reversal patterns
                # Bullish reversal (previous 3 candles down, current up)
                if idx > 3 and all(close_prices[i] < open_prices[i] for i in range(idx-3, idx)) and close_prices[idx] > open_prices[idx]:
                    action = 'BUY'
                    strength = max(strength, 0.7)  # Strong buy signal
                    
                # Bearish reversal (previous 3 candles up, current down)
                if idx > 3 and all(close_prices[i] > open_prices[i] for i in range(idx-3, idx)) and close_prices[idx] < open_prices[idx]:
                    action = 'SELL'
                    strength = max(strength, 0.7)  # Strong sell signal
                
                # 5. Prevent excessive trading - for each 10 candles, skip signals randomly
                if idx % 10 == 0 and random.random() < 0.5:
                    return {'action': 'NONE', 'strength': 0} 
                
                # Set a lower threshold for signal generation to ensure more trades
                if strength < 0.15:  # Very low threshold
                    action = 'NONE'
                    strength = 0
                
                # Apply pattern confidence threshold from parameters
                if strength < pattern_confidence_threshold:
                    action = 'NONE'
                    strength = 0
                
                return {'action': action, 'strength': strength}
            
            # Monkey patch the generate_signal method
            strategy.generate_signal = lambda current_time=None: mock_generate_signal(strategy, current_time)
            
            logger.info(f"Prepared Medallion strategy with parameters: {config}")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error preparing Medallion strategy with parameters: {e}")
            raise
    
    def run_backtest(self, strategy=None):
        """
        Run a backtest with the given strategy
        
        Args:
            strategy: Strategy instance (if None, use strategy prepared previously)
            
        Returns:
            dict: Dictionary of backtest metrics
        """
        if strategy is None:
            # Use the previously prepared strategy
            if not hasattr(self, 'strategy') or self.strategy is None:
                logger.error("No strategy prepared. Call prepare_strategy_with_params first.")
                return None
            strategy = self.strategy
        
        # Store the strategy
        self.strategy = strategy
        
        # Run the backtest
        results = super().run_backtest()
        
        # Calculate additional metrics
        days = (self.end_date - self.start_date).days
        annual_factor = 365 / max(1, days)
        
        metrics = {}
        
        if isinstance(results, dict):
            # Extract metrics from results
            metrics = {
                'net_profit': results.get('net_profit', 0),
                'return_pct': results.get('return_pct', 0),
                'annual_return': results.get('return_pct', 0) * annual_factor,
                'max_drawdown': results.get('max_drawdown_pct', 0),
                'win_rate': results.get('win_rate', 0),
                'profit_factor': results.get('profit_factor', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'total_trades': results.get('total_trades', 0)
            }
        else:
            # Try to extract metrics from self.metrics if available
            metrics = {
                'net_profit': getattr(self, 'net_profit', 0),
                'return_pct': getattr(self, 'return_pct', 0),
                'annual_return': getattr(self, 'return_pct', 0) * annual_factor,
                'max_drawdown': getattr(self, 'max_drawdown_pct', 0),
                'win_rate': getattr(self, 'win_rate', 0),
                'profit_factor': getattr(self, 'profit_factor', 0),
                'sharpe_ratio': getattr(self, 'sharpe_ratio', 0),
                'total_trades': getattr(self, 'total_trades', 0)
            }
        
        logger.info(f"Backtest completed with metrics: {metrics}")
        
        return metrics

# Monkey-patch the original backtester to use our enhanced version
sys.modules['backtest_medallion_strategy'].MedallionStrategyBacktester = EnhancedMedallionStrategyBacktester 