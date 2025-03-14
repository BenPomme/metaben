"""
Extension to the ML Strategy Backtester to support parameter optimization
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
from backtest_ml_strategy import MLStrategyBacktester

# Setup logging
logger = logging.getLogger('ml_strategy_backtester_extension')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class EnhancedMLStrategyBacktester(MLStrategyBacktester):
    """
    Enhanced version of the ML Strategy Backtester with parameter optimization support
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the enhanced backtester with the same arguments as the original
        """
        super().__init__(**kwargs)
        logger.info("Initialized EnhancedMLStrategyBacktester")
    
    def prepare_strategy_with_params(self, params):
        """
        Prepare a strategy instance with the given parameters
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            Object: Configured strategy instance
        """
        try:
            # Import the strategy class
            from simple_ml_strategy import SimpleMLStrategy
            
            # Extract key parameters
            lookback_periods = params.get('lookback_periods', 100)
            prediction_horizon = params.get('prediction_horizon', 5)
            model_type = params.get('model_type', 'ensemble')
            ensemble_voting = params.get('ensemble_voting', 'weighted')
            signal_threshold = params.get('signal_threshold', 0.6)
            min_model_agreement = params.get('min_model_agreement', 0.6)
            risk_percent = params.get('risk_percent', 1.0)
            atr_multiplier = params.get('atr_multiplier', 1.5)
            min_risk_reward = params.get('min_risk_reward', 1.5)
            
            # Create base configuration
            config = {
                'lookback_periods': lookback_periods,
                'prediction_horizon': prediction_horizon,
                'model_type': model_type,
                'ensemble_voting': ensemble_voting,
                'signal_threshold': signal_threshold,
                'min_model_agreement': min_model_agreement,
                'risk_percent': risk_percent,
                'atr_multiplier': atr_multiplier,
                'min_risk_reward': min_risk_reward,
                'max_trades_per_day': params.get('max_trades_per_day', 3),
                'partial_take_profit': params.get('partial_take_profit', False),
                'use_trailing_stop': params.get('use_trailing_stop', False)
            }
            
            # Add model-specific parameters
            if model_type == 'rf' or model_type == 'ensemble':
                config['rf_params'] = {
                    'n_estimators': params.get('rf_n_estimators', 100),
                    'max_depth': params.get('rf_max_depth', 10),
                    'min_samples_split': params.get('rf_min_samples_split', 2)
                }
            
            if model_type == 'gbm' or model_type == 'ensemble':
                config['gbm_params'] = {
                    'n_estimators': params.get('gbm_n_estimators', 100),
                    'learning_rate': params.get('gbm_learning_rate', 0.1),
                    'max_depth': params.get('gbm_max_depth', 3)
                }
            
            if model_type == 'nn' or model_type == 'ensemble':
                config['nn_params'] = {
                    'hidden_layers': params.get('nn_hidden_layers', [64, 32]),
                    'dropout_rate': params.get('nn_dropout_rate', 0.2),
                    'learning_rate': params.get('nn_learning_rate', 0.001)
                }
            
            if model_type == 'ensemble':
                config['ensemble_models'] = params.get('ensemble_models', ['rf', 'gbm', 'nn'])
            
            # Create and initialize strategy
            strategy = SimpleMLStrategy(
                symbol=self.symbol,
                primary_timeframe=self.primary_timeframe,
                secondary_timeframes=self.secondary_timeframes,
                mt5_connector=self.connector,
                config=config
            )
            
            # Prepare data and features
            for timeframe, data in self.data.items():
                # Make a copy to avoid modifying the original
                strategy.data[timeframe] = data.copy()
            
            # Feature generation based on parameters
            fast_ma = params.get('fast_ma_period', 12)
            slow_ma = params.get('slow_ma_period', 26)
            rsi_period = params.get('rsi_period', 14)
            bb_period = params.get('bb_period', 20)
            bb_std = params.get('bb_std', 2.0)
            
            # Generate technical indicators
            use_price_features = params.get('use_price_features', True)
            use_volume_features = params.get('use_volume_features', True)
            use_volatility_features = params.get('use_volatility_features', True)
            
            strategy.prepare_data(
                use_price_features=use_price_features,
                use_volume_features=use_volume_features,
                use_volatility_features=use_volatility_features,
                fast_ma=fast_ma,
                slow_ma=slow_ma,
                rsi_period=rsi_period,
                bb_period=bb_period,
                bb_std=bb_std
            )
            
            logger.info(f"Prepared ML strategy with parameters: {config}")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error preparing ML strategy with parameters: {e}")
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
        
        # Override the default run_backtest to return a metrics dictionary
        super().run_backtest(strategy)
        
        # Calculate additional metrics
        days = (self.end_date - self.start_date).days
        annual_factor = 365 / max(1, days)
        
        metrics = {
            'net_profit': self.metrics.get('net_profit', 0),
            'return_pct': self.metrics.get('return_pct', 0),
            'annual_return': self.metrics.get('return_pct', 0) * annual_factor,
            'max_drawdown': self.metrics.get('max_drawdown_pct', 0),
            'win_rate': self.metrics.get('win_rate', 0),
            'profit_factor': self.metrics.get('profit_factor', 0),
            'sharpe_ratio': self.metrics.get('sharpe_ratio', 0),
            'total_trades': self.metrics.get('total_trades', 0),
            'avg_trade_duration': self.metrics.get('avg_trade_duration', 0)
        }
        
        logger.info(f"Backtest completed with metrics: {metrics}")
        
        return metrics

# Monkey-patch the original backtester to use our enhanced version
sys.modules['backtest_ml_strategy'].MLStrategyBacktester = EnhancedMLStrategyBacktester 