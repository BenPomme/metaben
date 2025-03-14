"""
Medallion-Inspired Trading Strategy: Core Module

This module implements the core components of a trading strategy inspired by the
mathematical principles of Renaissance Technologies' Medallion Fund, including:
- Statistical arbitrage
- Advanced ML prediction models
- Portfolio optimization
- Risk management
- Execution optimization
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/medallion_strategy.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("medallion_strategy")

# Create required directories
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

class MedallionStrategy:
    """
    Core implementation of a Medallion-inspired trading strategy
    
    This class integrates multiple advanced components:
    1. Statistical models for market inefficiency detection
    2. Multi-model ensemble prediction system
    3. Portfolio optimization with dynamic risk allocation
    4. Execution optimization with market impact modeling
    5. Continuous adaptive learning system
    """
    
    def __init__(
        self, 
        symbol: str,
        primary_timeframe: str = 'H1',
        secondary_timeframes: List[str] = None,
        mt5_connector = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the Medallion-inspired strategy
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            primary_timeframe: Primary trading timeframe
            secondary_timeframes: List of additional timeframes for multi-timeframe analysis
            mt5_connector: MetaTrader 5 connector for data retrieval and execution
            config: Configuration parameters
        """
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframes = secondary_timeframes or ['M15', 'H4', 'D1']
        self.mt5_connector = mt5_connector
        
        # Default configuration
        self.default_config = {
            # Statistical model parameters
            'stat_model': {
                'mean_reversion_threshold': 2.0,  # Z-score threshold for mean reversion signals
                'pairs_correlation_threshold': 0.7,  # Min correlation for pair trading
                'volatility_window': 20,  # Window for volatility calculation
                'drift_estimation_window': 50,  # Window for drift component estimation
                'stochastic_process': 'ornstein_uhlenbeck'  # Type of stochastic process model
            },
            
            # Machine learning parameters
            'ml_model': {
                'lookback_periods': 20,  # Historical data points for feature generation
                'prediction_horizon': 5,  # Forecast horizon
                'retraining_frequency': 168,  # Hours between model retraining (1 week)
                'min_training_samples': 5000,  # Minimum samples required for training
                'performance_threshold': 0.55,  # Minimum accuracy to keep model in ensemble
                'feature_importance_threshold': 0.01,  # Min importance to keep feature
                'model_weight_method': 'dynamic_sharpe'  # Method to weight models in ensemble
            },
            
            # Reinforcement learning parameters
            'rl_model': {
                'enabled': True,
                'state_variables': ['price', 'volume', 'indicators', 'position'],
                'reward_function': 'sharpe',  # Alternative: 'pnl', 'sortino', 'calmar'
                'learning_rate': 0.001,
                'exploration_rate': 0.1,
                'discount_factor': 0.95
            },
            
            # Portfolio optimization
            'portfolio': {
                'max_leverage': 5.0,  # Maximum leverage
                'target_volatility': 0.10,  # Annualized portfolio volatility target
                'optimization_method': 'mean_variance',  # Alternative: 'min_variance', 'max_sharpe', 'kelly'
                'position_sizing_method': 'kelly',  # Alternative: 'fixed', 'volatility', 'optimal_f'
                'rebalancing_frequency': 24,  # Hours between portfolio rebalancing
                'correlation_window': 500,  # Data points for correlation calculation
                'risk_aversion_parameter': 0.5  # Lambda in mean-variance optimization
            },
            
            # Risk management
            'risk': {
                'max_drawdown_limit': 0.15,  # Maximum allowable drawdown before reducing exposure
                'var_confidence_level': 0.99,  # VaR confidence level
                'var_window': 250,  # Window for VaR calculation
                'stress_test_scenarios': ['vol_spike', 'trend_reversal', 'liquidity_crisis'],
                'tail_risk_measure': 'cvar',  # Conditional Value at Risk
                'max_position_size': 0.05,  # Max position size as fraction of capital
                'risk_allocation_method': 'equal',  # Alternative: 'volatility_parity', 'equal_risk_contribution'
                'correlation_regime_adjustment': True  # Adjust for changing correlation regimes
            },
            
            # Execution optimization
            'execution': {
                'market_impact_model': 'square_root',  # Model for market impact
                'execution_style': 'adaptive',  # Alternative: 'aggressive', 'passive', 'twap', 'vwap'
                'min_trade_interval': 300,  # Minimum seconds between trades
                'iceberg_order_threshold': 5.0,  # Volume multiple for iceberg orders
                'smart_order_routing': True,  # Use smart order routing
                'transaction_cost_model': 'fixed_plus_slippage'  # Model for transaction costs
            },
            
            # Signal generation and filtering
            'signal': {
                'min_signal_strength': 0.6,  # Minimum signal strength (0-1)
                'confirmation_required': True,  # Require confirmation from multiple sub-models
                'filter_regime': True,  # Filter signals based on detected market regime
                'signal_decay_factor': 0.5,  # Weight decay factor for older signals
                'signal_smoothing_window': 3,  # Window for signal smoothing
                'multi_timeframe_agreement_threshold': 0.7  # Required agreement across timeframes
            },
            
            # Anomaly detection
            'anomaly': {
                'enabled': True,
                'detection_method': 'isolation_forest',  # Alternative: 'autoencoder', 'local_outlier_factor'
                'threshold': 0.95,  # Anomaly score threshold
                'reaction_policy': 'reduce_exposure'  # Alternative: 'exit_positions', 'hedge'
            },
            
            # Continuous learning and adaptation
            'adaptation': {
                'performance_evaluation_window': 500,  # Bars for performance evaluation
                'regime_detection_enabled': True,  # Detect and adapt to market regimes
                'feature_selection_frequency': 168,  # Hours between feature re-selection
                'hyperparameter_optimization_frequency': 720,  # Hours between hyperparameter tuning
                'model_ensemble_weights_update_frequency': 24,  # Hours between ensemble weight updates
                'dynamic_timeframe_weighting': True  # Dynamically adjust weights of different timeframes
            }
        }
        
        # Override defaults with provided config
        self.config = self.default_config.copy()
        if config:
            self._update_nested_dict(self.config, config)
            
        # Runtime state variables
        self.data = {}  # Data for each timeframe
        self.features = {}  # Calculated features for each timeframe
        self.signals = {}  # Generated signals
        self.positions = []  # Current positions
        self.performance_metrics = {}  # Performance tracking
        self.market_state = {
            'regime': 'unknown',
            'volatility': 0.0,
            'trend_strength': 0.0,
            'liquidity': 0.0
        }
        
        # Initialize components (will be set up in separate methods)
        self.statistical_models = {}
        self.ml_models = {}
        self.rl_model = None
        self.portfolio_optimizer = None
        self.risk_manager = None
        self.execution_optimizer = None
        self.anomaly_detector = None
        
        # Training and execution state
        self.last_model_training = None
        self.last_feature_selection = None
        self.last_weight_update = None
        self.last_hyperparameter_optimization = None
        self.last_signal_time = None
        self.last_portfolio_rebalance = None
        
        logger.info(f"Initialized Medallion-inspired strategy for {symbol}")
        
    def _update_nested_dict(self, d, u):
        """Helper method to update nested dictionaries"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def initialize_components(self):
        """Initialize all strategy components"""
        try:
            # Import necessary component modules
            from medallion_statistical_models import (
                StatisticalModelFactory, StochasticProcessModels, StatisticalArbitrage
            )
            from medallion_ml_models import (
                AdvancedMLModelEnsemble, MarketRegimeDetector, AnomalyDetectionModel
            )
            from medallion_risk_management import (
                RiskManager, PortfolioOptimizer, StressTestEngine
            )
            from medallion_execution import (
                ExecutionOptimizer, MarketImpactModel, TransactionCostModel
            )
            
            logger.info("Initializing strategy components...")
            
            # Initialize statistical models
            self._init_statistical_models()
            
            # Initialize ML models
            self._init_ml_models()
            
            # Initialize risk management
            self._init_risk_management()
            
            # Initialize execution optimization
            self._init_execution_optimization()
            
            # Initialize market regime detection
            self._init_market_regime_detection()
            
            # Initialize anomaly detection if enabled
            if self.config['anomaly']['enabled']:
                self._init_anomaly_detection()
                
            logger.info("All strategy components initialized successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Component initialization failed: {str(e)}")
            logger.info("You need to implement the component modules first")
            return False
        except Exception as e:
            logger.error(f"Component initialization failed: {str(e)}")
            logger.exception("Component initialization error")
            return False
    
    def _init_statistical_models(self):
        """Initialize statistical models"""
        # This is a placeholder - will be implemented in medallion_statistical_models.py
        logger.info("Statistical models initialization placeholder")
        
    def _init_ml_models(self):
        """Initialize machine learning models"""
        # This is a placeholder - will be implemented in medallion_ml_models.py
        logger.info("ML models initialization placeholder")
        
    def _init_risk_management(self):
        """Initialize risk management components"""
        # This is a placeholder - will be implemented in medallion_risk_management.py
        logger.info("Risk management initialization placeholder")
        
    def _init_execution_optimization(self):
        """Initialize execution optimization components"""
        # This is a placeholder - will be implemented in medallion_execution.py
        logger.info("Execution optimization initialization placeholder")
        
    def _init_market_regime_detection(self):
        """Initialize market regime detection"""
        # This is a placeholder - will be implemented in medallion_ml_models.py
        logger.info("Market regime detection initialization placeholder")
        
    def _init_anomaly_detection(self):
        """Initialize anomaly detection"""
        # This is a placeholder - will be implemented in medallion_ml_models.py
        logger.info("Anomaly detection initialization placeholder")
    
    def prepare_data(self, data):
        """
        Prepare and validate input data for strategy execution
        
        Args:
            data: Dictionary of DataFrames for each timeframe
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.data = {}
            
            # Validate and store data for each timeframe
            for tf in [self.primary_timeframe] + self.secondary_timeframes:
                if tf in data and not data[tf].empty:
                    # Validate required columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume', 'time']
                    missing_cols = [col for col in required_cols if col not in data[tf].columns]
                    
                    if missing_cols:
                        logger.warning(f"Missing columns for {tf}: {missing_cols}")
                        continue
                    
                    # Copy the data
                    self.data[tf] = data[tf].copy()
                    
                    # Ensure datetime index
                    if not isinstance(self.data[tf].index, pd.DatetimeIndex):
                        if 'time' in self.data[tf].columns:
                            self.data[tf]['time'] = pd.to_datetime(self.data[tf]['time'])
                            self.data[tf].set_index('time', inplace=True)
                        else:
                            logger.warning(f"No datetime column found for {tf}, using default index")
                    
                    logger.info(f"Prepared data for {tf}: {len(self.data[tf])} rows")
                else:
                    logger.warning(f"No data available for timeframe {tf}")
            
            if not self.data:
                logger.error("No valid data available for any timeframe")
                return False
                
            if self.primary_timeframe not in self.data:
                logger.error(f"No data available for primary timeframe {self.primary_timeframe}")
                return False
                
            # Calculate features for all timeframes
            self._calculate_features()
            
            # Detect market regime
            self._detect_market_regime()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            logger.exception("Data preparation error")
            return False
    
    def _calculate_features(self):
        """Calculate features for all timeframes"""
        self.features = {}
        
        for tf, df in self.data.items():
            try:
                # This will be expanded in the implementation with actual feature calculation
                logger.info(f"Calculating features for {tf}")
                self.features[tf] = df.copy()  # Placeholder
            except Exception as e:
                logger.error(f"Error calculating features for {tf}: {str(e)}")
    
    def _detect_market_regime(self):
        """Detect current market regime"""
        try:
            if not self.data or self.primary_timeframe not in self.data:
                return
                
            # This is a placeholder for market regime detection
            # Will be implemented in the MarketRegimeDetector class
            logger.info("Detecting market regime")
            
            # Placeholder detection
            df = self.data[self.primary_timeframe]
            
            # Simple volatility calculation as example
            returns = df['close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)  # Annualized
            
            # Simple trend strength calculation as example
            sma_50 = df['close'].rolling(window=50).mean()
            sma_200 = df['close'].rolling(window=200).mean()
            trend_strength = abs(sma_50.iloc[-1] - sma_200.iloc[-1]) / df['close'].iloc[-1]
            
            # Update market state
            self.market_state['volatility'] = volatility
            self.market_state['trend_strength'] = trend_strength
            
            # Simplified regime classification
            if volatility > 0.2:  # High volatility
                if trend_strength > 0.01:
                    self.market_state['regime'] = 'volatile_trending'
                else:
                    self.market_state['regime'] = 'volatile_ranging'
            else:  # Low volatility
                if trend_strength > 0.01:
                    self.market_state['regime'] = 'stable_trending'
                else:
                    self.market_state['regime'] = 'stable_ranging'
                    
            logger.info(f"Detected market regime: {self.market_state['regime']}")
            logger.info(f"Market volatility: {self.market_state['volatility']:.4f}")
            logger.info(f"Trend strength: {self.market_state['trend_strength']:.4f}")
            
        except Exception as e:
            logger.error(f"Error in market regime detection: {str(e)}")
            self.market_state['regime'] = 'unknown'
    
    def generate_signal(self, current_time=None):
        """
        Generate trading signal based on all implemented models
        
        Args:
            current_time: Current timestamp (default: use latest from data)
            
        Returns:
            dict: Trading signal with action, parameters and diagnostics
        """
        if current_time is None and self.primary_timeframe in self.data:
            current_time = self.data[self.primary_timeframe].index[-1]
            
        logger.info(f"Generating signal for {self.symbol} at {current_time}")
        
        # Check if enough time has passed since last signal
        if self.last_signal_time and (current_time - self.last_signal_time).total_seconds() < self.config['execution']['min_trade_interval']:
            logger.info(f"Signal generation skipped, minimum interval not reached")
            return {'action': 'NONE', 'reason': 'Signal throttling'}
        
        # Generate signal components from each model type
        stat_signal = self._generate_statistical_signal()
        ml_signal = self._generate_ml_signal()
        
        # Check for market anomalies if enabled
        anomaly_detected = False
        if self.config['anomaly']['enabled']:
            anomaly_detected = self._detect_anomalies()
            if anomaly_detected:
                logger.warning(f"Market anomaly detected, adjusting signal generation")
        
        # Combine signals (placeholder for actual signal combination logic)
        combined_signal = self._combine_signals(stat_signal, ml_signal, anomaly_detected)
        
        # Apply risk management constraints
        final_signal = self._apply_risk_constraints(combined_signal)
        
        # Update last signal time
        self.last_signal_time = current_time
        
        # Log signal generation
        signal_strength = final_signal.get('signal_strength', 0)
        action = final_signal.get('action', 'NONE')
        
        logger.info(f"Generated {action} signal with strength {signal_strength:.4f}")
        
        return final_signal
    
    def _generate_statistical_signal(self):
        """Generate signal from statistical models"""
        # Placeholder for statistical model signal generation
        return {'direction': 0, 'strength': 0, 'confidence': 0}
    
    def _generate_ml_signal(self):
        """Generate signal from ML models"""
        # Placeholder for ML model signal generation
        return {'direction': 0, 'strength': 0, 'confidence': 0}
    
    def _detect_anomalies(self):
        """Detect market anomalies"""
        # Placeholder for anomaly detection
        return False
    
    def _combine_signals(self, stat_signal, ml_signal, anomaly_detected):
        """
        Combine signals from different models
        
        Args:
            stat_signal: Signal from statistical models
            ml_signal: Signal from ML models
            anomaly_detected: Whether an anomaly was detected
            
        Returns:
            dict: Combined signal
        """
        # Placeholder for signal combination logic
        combined_signal = {
            'action': 'NONE',
            'entry': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'risk_percent': 0,
            'signal_strength': 0,
        }
        
        # Placeholder logic - will be replaced with actual signal combination
        if anomaly_detected:
            return combined_signal
            
        # Example simple combination
        combined_direction = 0.7 * ml_signal['direction'] + 0.3 * stat_signal['direction']
        combined_strength = 0.7 * ml_signal['strength'] + 0.3 * stat_signal['strength']
        
        # Convert continuous signal to discrete action
        if combined_direction > 0.2 and combined_strength > self.config['signal']['min_signal_strength']:
            action = 'BUY'
        elif combined_direction < -0.2 and combined_strength > self.config['signal']['min_signal_strength']:
            action = 'SELL'
        else:
            action = 'NONE'
            
        if action != 'NONE' and self.primary_timeframe in self.data:
            # Get current price
            current_price = self.data[self.primary_timeframe]['close'].iloc[-1]
            
            # Calculate ATR for stop loss and take profit
            atr = self._calculate_atr(self.data[self.primary_timeframe])
            
            # Set entry, stop loss and take profit based on action
            if action == 'BUY':
                entry = current_price
                stop_loss = entry - 2 * atr
                take_profit = entry + 3 * atr
            else:  # SELL
                entry = current_price
                stop_loss = entry + 2 * atr
                take_profit = entry - 3 * atr
                
            # Calculate risk percentage based on signal strength and model confidence
            risk_percent = self.config['portfolio']['max_leverage'] * combined_strength
            
            combined_signal = {
                'action': action,
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_percent': risk_percent,
                'signal_strength': combined_strength,
                'ml_confidence': ml_signal['confidence'],
                'stat_confidence': stat_signal['confidence'],
                'market_regime': self.market_state['regime'],
                'volatility': self.market_state['volatility']
            }
        
        return combined_signal
    
    def _calculate_atr(self, df, window=14):
        """
        Calculate Average True Range
        
        Args:
            df: DataFrame with OHLC data
            window: ATR calculation window
            
        Returns:
            float: ATR value
        """
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=window).mean().iloc[-1]
        
        return atr
    
    def _apply_risk_constraints(self, signal):
        """
        Apply risk management constraints to the signal
        
        Args:
            signal: Combined trading signal
            
        Returns:
            dict: Risk-adjusted signal
        """
        # This is a placeholder for risk management logic
        return signal
    
    def optimize_portfolio(self):
        """Optimize portfolio allocation based on current positions and signals"""
        # Placeholder for portfolio optimization logic
        logger.info("Portfolio optimization placeholder")
        return True
    
    def optimize_execution(self, signal):
        """
        Optimize trade execution for a given signal
        
        Args:
            signal: Trading signal
            
        Returns:
            dict: Execution plan
        """
        # Placeholder for execution optimization logic
        logger.info("Execution optimization placeholder")
        return signal
    
    def evaluate_performance(self):
        """Evaluate strategy performance"""
        # Placeholder for performance evaluation
        logger.info("Performance evaluation placeholder")
        return {}
    
    def update_models(self):
        """Update models based on recent performance"""
        # Placeholder for model update logic
        logger.info("Model update placeholder")
        return True
    
    def save_strategy_state(self):
        """Save the current state of the strategy"""
        # Placeholder for state saving logic
        logger.info("Strategy state saving placeholder")
        return True
    
    def load_strategy_state(self):
        """Load the saved state of the strategy"""
        # Placeholder for state loading logic
        logger.info("Strategy state loading placeholder")
        return True
    
    def visualize_strategy(self):
        """Generate visualization of strategy performance and decisions"""
        # Placeholder for visualization logic
        logger.info("Strategy visualization placeholder")
        return True


# For testing/development
if __name__ == "__main__":
    # Example usage
    strategy = MedallionStrategy(symbol="EURUSD")
    strategy.initialize_components()
    
    # This would be expanded in actual implementation
    print("Medallion strategy initialized for testing") 