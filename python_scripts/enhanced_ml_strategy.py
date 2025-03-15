"""
Enhanced ML Strategy for Forex Trading

This module provides an advanced trading strategy that leverages machine learning models,
ensemble methods, and multi-timeframe analysis to make trading decisions.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import asyncio
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_ml_strategy.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("enhanced_ml_strategy")

# Import our modules
from ml_models import (
    EnsembleModel, RandomForestModel, GradientBoostingModel, 
    NeuralNetworkModel, SVMModel, TechnicalFeatureGenerator
)

class EnhancedMLStrategy:
    """
    Advanced ML strategy that uses ensemble learning and multi-timeframe analysis
    """
    
    def __init__(
        self, 
        symbol, 
        primary_timeframe='H1', 
        secondary_timeframes=None, 
        mt5_connector=None,
        config=None
    ):
        """
        Initialize the enhanced ML strategy
        
        Args:
            symbol: Trading symbol
            primary_timeframe: Primary timeframe for analysis
            secondary_timeframes: List of secondary timeframes for multi-timeframe analysis
            mt5_connector: MT5 connector instance
            config: Strategy configuration parameters
        """
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframes = secondary_timeframes if secondary_timeframes else ['H4', 'D1']
        self.mt5_connector = mt5_connector
        
        # Default configuration
        self.config = {
            # ML model parameters
            'lookback_periods': 10,
            'prediction_horizon': 5,
            'ensemble_voting': 'soft',
            'model_weight_threshold': 0.6,  # Minimum accuracy for model inclusion
            
            # Signal parameters
            'signal_threshold': 0.6,
            'min_model_agreement': 0.7,  # Minimum proportion of models that must agree
            
            # Risk management
            'risk_percent': 1.0,
            'atr_multiplier': 2.0,
            'min_risk_reward': 1.5,
            'max_trades_per_day': 3,
            
            # Multi-timeframe parameters
            'timeframe_agreement_threshold': 0.7,
            
            # Trade management
            'trailing_stop_activation': 1.0,  # ATR multiplier for trailing stop activation
            'trailing_stop_distance': 1.5,  # ATR multiplier for trailing stop distance
            'partial_take_profit': True,
            'partial_take_profit_level': 1.0,  # ATR multiplier for first take profit
            'partial_take_profit_size': 0.5  # Portion of position to close at first take profit
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # Data containers
        self.data = {}
        self.indicators = {}
        self.ml_models = {}
        self.ensemble_model = None
        
        # Trade tracking
        self.last_trade_time = None
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        # Create required directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        logger.info(f"Initialized Enhanced ML Strategy for {symbol} on {primary_timeframe}")
        logger.info(f"Secondary timeframes: {self.secondary_timeframes}")
        logger.info(f"Config: {json.dumps(self.config, indent=2)}")
        
    def prepare_data(self, data):
        """
        Prepare data for strategy execution
        
        Args:
            data (dict): Dictionary of dataframes for each timeframe
        """
        self.data = data
        
        # Calculate indicators for each timeframe
        for tf in [self.primary_timeframe] + self.secondary_timeframes:
            if tf in self.data and not self.data[tf].empty:
                self._calculate_indicators(tf)
            else:
                logger.warning(f"No data available for timeframe {tf}")
                
        logger.info("Data preparation completed")
        
    def _calculate_indicators(self, timeframe):
        """
        Calculate technical indicators for a specific timeframe
        
        Args:
            timeframe (str): Timeframe to calculate indicators for
        """
        if timeframe not in self.data or self.data[timeframe].empty:
            logger.warning(f"No data available for {timeframe}")
            return
            
        # Get dataframe for this timeframe
        df = self.data[timeframe].copy()
        
        # Use TechnicalFeatureGenerator to calculate all indicators
        df_with_indicators = TechnicalFeatureGenerator.add_technical_indicators(df)
        df_with_patterns = TechnicalFeatureGenerator.add_pattern_recognition(df_with_indicators)
        
        # Store indicators
        self.indicators[timeframe] = df_with_patterns
        
        logger.debug(f"Calculated indicators for {timeframe}")
        
    def _build_ml_models(self):
        """
        Build ML models for all timeframes
        """
        logger.info("Building ML models")
        
        # Create models for primary timeframe
        self.ml_models[self.primary_timeframe] = {
            'rf': RandomForestModel(
                self.symbol, 
                self.primary_timeframe,
                lookback_periods=self.config['lookback_periods'],
                prediction_horizon=self.config['prediction_horizon']
            ),
            'gb': GradientBoostingModel(
                self.symbol, 
                self.primary_timeframe,
                lookback_periods=self.config['lookback_periods'],
                prediction_horizon=self.config['prediction_horizon']
            ),
            'nn': NeuralNetworkModel(
                self.symbol, 
                self.primary_timeframe,
                lookback_periods=self.config['lookback_periods'],
                prediction_horizon=self.config['prediction_horizon']
            ),
            'svm': SVMModel(
                self.symbol, 
                self.primary_timeframe,
                lookback_periods=self.config['lookback_periods'],
                prediction_horizon=self.config['prediction_horizon']
            )
        }
        
        # Create ensemble model for primary timeframe
        self.ensemble_model = EnsembleModel(
            self.symbol,
            self.primary_timeframe,
            models=list(self.ml_models[self.primary_timeframe].values()),
            voting=self.config['ensemble_voting']
        )
        
        # Create simpler models for secondary timeframes
        for tf in self.secondary_timeframes:
            self.ml_models[tf] = {
                'rf': RandomForestModel(
                    self.symbol, 
                    tf,
                    lookback_periods=self.config['lookback_periods'],
                    prediction_horizon=self.config['prediction_horizon']
                )
            }
            
        logger.info(f"Created models for {len(self.ml_models)} timeframes")
        
    def train_models(self, force_retrain=False):
        """
        Train all ML models
        
        Args:
            force_retrain (bool): Whether to force retraining even if models exist
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Build models if they don't exist
            if not self.ml_models:
                self._build_ml_models()
                
            # Check if models are already trained
            if not force_retrain:
                primary_models_trained = all(
                    model.load() 
                    for model in self.ml_models[self.primary_timeframe].values()
                )
                
                if primary_models_trained:
                    logger.info("Models already trained, skipping training")
                    return True
                    
            # Train models for primary timeframe
            logger.info(f"Training models for {self.primary_timeframe}")
            
            primary_df = self.indicators[self.primary_timeframe]
            for name, model in self.ml_models[self.primary_timeframe].items():
                logger.info(f"Training {name} model")
                result = model.train(primary_df)
                logger.info(f"{name} model trained with accuracy: {result.get('accuracy', 0):.4f}")
                
            # Train ensemble model
            logger.info("Training ensemble model")
            ensemble_result = self.ensemble_model.train(primary_df)
            logger.info(f"Ensemble model trained with {len(ensemble_result.get('results', []))} models")
            
            # Train models for secondary timeframes
            for tf in self.secondary_timeframes:
                if tf in self.indicators:
                    logger.info(f"Training models for {tf}")
                    tf_df = self.indicators[tf]
                    
                    for name, model in self.ml_models[tf].items():
                        logger.info(f"Training {name} model for {tf}")
                        result = model.train(tf_df)
                        logger.info(f"{name} model for {tf} trained with accuracy: {result.get('accuracy', 0):.4f}")
                else:
                    logger.warning(f"No indicators available for {tf}, skipping model training")
                    
            logger.info("All models trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            logger.exception("Training error")
            return False
            
    def _get_prediction(self, timeframe=None):
        """
        Get prediction for a specific timeframe
        
        Args:
            timeframe (str): Timeframe to get prediction for (default: primary timeframe)
            
        Returns:
            tuple: (direction, confidence) where direction is -1, 0, or 1 and confidence is 0-1
        """
        if timeframe is None:
            timeframe = self.primary_timeframe
            
        if timeframe not in self.ml_models:
            logger.warning(f"No models available for {timeframe}")
            return 0, 0.0
            
        if timeframe not in self.indicators:
            logger.warning(f"No indicators available for {timeframe}")
            return 0, 0.0
            
        current_data = self.indicators[timeframe]
        
        if timeframe == self.primary_timeframe and self.ensemble_model is not None:
            # Use ensemble model for primary timeframe
            preds, probs, _ = self.ensemble_model.predict(current_data)
            
            if preds is not None and len(preds) > 0:
                # Use the latest prediction
                latest_pred = preds[-1]
                latest_prob = probs[-1] if probs is not None else 0.5
                
                # Convert to -1, 0, 1
                if latest_pred == 1:
                    direction = 1
                    confidence = latest_prob
                else:
                    direction = -1
                    confidence = 1 - latest_prob
                    
                # Apply threshold
                if confidence < self.config['signal_threshold']:
                    direction = 0
                    
                return direction, confidence
            else:
                logger.warning("No predictions available from ensemble model")
                return 0, 0.0
        else:
            # Use individual model for secondary timeframe
            model = next(iter(self.ml_models[timeframe].values()))
            preds, probs = model.predict(current_data)
            
            if preds is not None and len(preds) > 0:
                # Use the latest prediction
                latest_pred = preds[-1]
                
                # Convert to -1, 0, 1
                if latest_pred == 1:
                    direction = 1
                    confidence = 0.7  # Default confidence for secondary timeframes
                else:
                    direction = -1
                    confidence = 0.7
                    
                return direction, confidence
            else:
                logger.warning(f"No predictions available for {timeframe}")
                return 0, 0.0
                
    def _check_trade_limits(self, current_time):
        """
        Check if we've exceeded trade limits
        
        Args:
            current_time (datetime): Current bar time
            
        Returns:
            bool: True if trade limits are not exceeded, False otherwise
        """
        current_date = current_time.date()
        
        # Reset daily trade count if it's a new day
        if self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date
        
        # Check if we've exceeded the maximum trades per day
        if self.daily_trade_count >= self.config['max_trades_per_day']:
            logger.info(f"Maximum trades per day ({self.config['max_trades_per_day']}) reached for {current_date}")
            return False
        
        # Check if enough time has passed since the last trade (at least 4 hours)
        if self.last_trade_time and (current_time - self.last_trade_time).total_seconds() < 14400:  # 4 hours in seconds
            logger.info(f"Not enough time passed since last trade at {self.last_trade_time}")
            return False
            
        return True
    
    def _analyze_multi_timeframe(self):
        """
        Analyze all timeframes and check for agreement
        
        Returns:
            tuple: (overall_direction, overall_confidence, timeframe_agreement)
        """
        predictions = {}
        
        # Get predictions for all timeframes
        for tf in [self.primary_timeframe] + self.secondary_timeframes:
            if tf in self.ml_models:
                direction, confidence = self._get_prediction(tf)
                predictions[tf] = (direction, confidence)
                logger.info(f"Prediction for {tf}: direction={direction}, confidence={confidence:.2f}")
            
        if not predictions:
            logger.warning("No predictions available")
            return 0, 0.0, 0.0
            
        # Calculate agreement between timeframes
        non_zero_directions = [d for d, c in predictions.values() if d != 0]
        
        if not non_zero_directions:
            logger.info("No clear signals from any timeframe")
            return 0, 0.0, 0.0
            
        agreement_score = 0.0
        if len(non_zero_directions) >= 2:
            # Calculate proportion of timeframes that agree with the most common direction
            from collections import Counter
            direction_counts = Counter(non_zero_directions)
            most_common_direction, count = direction_counts.most_common(1)[0]
            agreement_score = count / len(non_zero_directions)
            
            logger.info(f"Agreement score: {agreement_score:.2f} for direction {most_common_direction}")
            
            # If agreement is strong enough, use the common direction
            if agreement_score >= self.config['timeframe_agreement_threshold']:
                # Get average confidence from agreeing timeframes
                avg_confidence = np.mean([
                    c for tf, (d, c) in predictions.items() if d == most_common_direction
                ])
                
                return most_common_direction, avg_confidence, agreement_score
        
        # If no strong agreement, use primary timeframe
        primary_direction, primary_confidence = predictions[self.primary_timeframe]
        
        # Adjust confidence based on agreement
        adjusted_confidence = primary_confidence * (0.5 + 0.5 * agreement_score)
        
        return primary_direction, adjusted_confidence, agreement_score
    
    def _calculate_risk_reward(self, entry_price, stop_loss, take_profit):
        """
        Calculate risk-reward ratio for a potential trade
        
        Args:
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            take_profit (float): Take profit price
            
        Returns:
            float: Risk-reward ratio
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return 0
            
        return reward / risk
    
    def generate_signal(self, current_time):
        """
        Generate trading signal based on ML models and multi-timeframe analysis
        
        Args:
            current_time (datetime): Current bar time
            
        Returns:
            dict: Trading signal with action, entry, stop loss, take profit, etc.
        """
        # Check if we have enough data and models
        if not self.indicators or self.primary_timeframe not in self.indicators:
            logger.warning("Not enough data to generate signal")
            return {
                'action': 'NONE',
                'entry': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'risk_percent': 0,
                'signal_strength': 0,
                'reason': 'Insufficient data'
            }
            
        if not self.ml_models:
            logger.warning("ML models not initialized")
            return {
                'action': 'NONE',
                'entry': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'risk_percent': 0,
                'signal_strength': 0,
                'reason': 'Models not initialized'
            }
        
        # Check trade limits
        if not self._check_trade_limits(current_time):
            return {
                'action': 'NONE',
                'entry': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'risk_percent': 0,
                'signal_strength': 0,
                'reason': 'Trade limits exceeded'
            }
        
        # Get current price data
        df = self.indicators[self.primary_timeframe]
        if df.empty:
            logger.warning("No data available")
            return {
                'action': 'NONE',
                'entry': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'risk_percent': 0,
                'signal_strength': 0,
                'reason': 'No data available'
            }
            
        current_price = df.iloc[-1]['close']
        
        # Get ATR for stop loss and take profit calculation
        if 'atr_14' in df.columns:
            current_atr = df.iloc[-1]['atr_14']
        else:
            # Calculate ATR if not already available
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            current_atr = true_range.rolling(14).mean().iloc[-1]
        
        # Analyze multi-timeframe signals
        signal, signal_strength, agreement = self._analyze_multi_timeframe()
        
        # If signal is not strong enough, don't trade
        if abs(signal) < 0.5 or signal_strength < self.config['signal_threshold']:
            logger.info(f"Signal not strong enough: {signal} with strength {signal_strength:.2f}")
            return {
                'action': 'NONE',
                'entry': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'risk_percent': 0,
                'signal_strength': signal_strength,
                'reason': 'Signal not strong enough'
            }
        
        # Calculate stop loss and take profit based on ATR
        atr_multiplier = self.config['atr_multiplier']
        
        if signal > 0:  # Buy signal
            entry = current_price
            stop_loss = entry - (current_atr * atr_multiplier)
            take_profit = entry + (current_atr * atr_multiplier * self.config['min_risk_reward'])
            action = 'BUY'
        else:  # Sell signal
            entry = current_price
            stop_loss = entry + (current_atr * atr_multiplier)
            take_profit = entry - (current_atr * atr_multiplier * self.config['min_risk_reward'])
            action = 'SELL'
        
        # Calculate risk-reward ratio
        risk_reward = self._calculate_risk_reward(entry, stop_loss, take_profit)
        
        # Check if risk-reward ratio is acceptable
        if risk_reward < self.config['min_risk_reward']:
            logger.info(f"Risk-reward ratio too low: {risk_reward:.2f} < {self.config['min_risk_reward']}")
            return {
                'action': 'NONE',
                'entry': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'risk_percent': 0,
                'signal_strength': signal_strength,
                'reason': 'Risk-reward ratio too low'
            }
        
        # Adjust risk percent based on signal strength and agreement
        adjusted_risk = self.config['risk_percent'] * signal_strength * (0.5 + 0.5 * agreement)
        
        # Update trade tracking
        self.last_trade_time = current_time
        self.daily_trade_count += 1
        
        # Set up partial take profit levels if configured
        partial_take_profits = []
        if self.config['partial_take_profit']:
            first_tp_distance = current_atr * self.config['partial_take_profit_level']
            if action == 'BUY':
                first_tp = entry + first_tp_distance
                partial_take_profits.append({
                    'price': first_tp,
                    'size': self.config['partial_take_profit_size']
                })
            else:
                first_tp = entry - first_tp_distance
                partial_take_profits.append({
                    'price': first_tp,
                    'size': self.config['partial_take_profit_size']
                })
        
        # Trailing stop settings if configured
        trailing_stop_settings = None
        if self.config['trailing_stop_activation'] > 0:
            trailing_stop_settings = {
                'activation_level': current_atr * self.config['trailing_stop_activation'],
                'distance': current_atr * self.config['trailing_stop_distance']
            }
        
        logger.info(f"Generated {action} signal with strength {signal_strength:.2f}")
        logger.info(f"Entry: {entry:.5f}, Stop Loss: {stop_loss:.5f}, Take Profit: {take_profit:.5f}")
        logger.info(f"Risk: {adjusted_risk:.2f}%, Risk-Reward: {risk_reward:.2f}")
        
        return {
            'action': action,
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_percent': adjusted_risk,
            'signal_strength': signal_strength,
            'agreement': agreement,
            'risk_reward': risk_reward,
            'partial_take_profits': partial_take_profits,
            'trailing_stop': trailing_stop_settings
        }
    
    def visualize_model_performance(self):
        """
        Visualize model performance
        
        Generates plots showing model accuracy, feature importance, and prediction results
        """
        try:
            if not self.ml_models or self.primary_timeframe not in self.ml_models:
                logger.warning("No models available to visualize")
                return
                
            # Collect accuracy scores
            model_names = []
            accuracies = []
            
            for name, model in self.ml_models[self.primary_timeframe].items():
                if hasattr(model, 'model') and model.model is not None:
                    model_names.append(name)
                    # Try to get accuracy from model metadata
                    model_path = Path(f"models/{model.model_name}_info.json")
                    if model_path.exists():
                        with open(model_path, 'r') as f:
                            info = json.load(f)
                            accuracies.append(info.get('accuracy', 0.5))
                    else:
                        # Default to 0.5 if not available
                        accuracies.append(0.5)
            
            # Create accuracy plot
            if model_names and accuracies:
                plt.figure(figsize=(10, 6))
                plt.bar(model_names, accuracies)
                plt.title(f'Model Accuracy for {self.symbol} on {self.primary_timeframe}')
                plt.ylabel('Accuracy')
                plt.ylim(0, 1)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.savefig(f"results/model_accuracy_{self.symbol}_{self.primary_timeframe}.png")
                plt.close()
                
            # Plot feature importances if available
            for name, model in self.ml_models[self.primary_timeframe].items():
                if hasattr(model.model, 'feature_importances_'):
                    # Model has feature importances
                    feature_importances = model.model.feature_importances_
                    feature_names = model.feature_columns
                    
                    # Sort by importance
                    indices = np.argsort(feature_importances)[::-1]
                    
                    # Plot top 15 features
                    plt.figure(figsize=(12, 8))
                    plt.title(f'Feature Importance for {name.upper()} on {self.symbol}')
                    plt.bar(range(min(15, len(indices))), 
                            feature_importances[indices[:15]], 
                            align='center')
                    plt.xticks(range(min(15, len(indices))), 
                               [feature_names[i] for i in indices[:15]], 
                               rotation=90)
                    plt.tight_layout()
                    plt.savefig(f"results/feature_importance_{name}_{self.symbol}.png")
                    plt.close()
            
            # Plot latest predictions
            if self.primary_timeframe in self.indicators:
                df = self.indicators[self.primary_timeframe].copy()
                
                # Get predictions from each model
                for name, model in self.ml_models[self.primary_timeframe].items():
                    preds, _ = model.predict(df)
                    
                    if preds is not None and len(preds) > 0:
                        # Add predictions to dataframe
                        pred_index = df.index[-len(preds):]
                        df.loc[pred_index, f'{name}_pred'] = preds
                
                # Plot price and predictions for the last 100 bars
                plt.figure(figsize=(15, 10))
                
                # Plot price
                ax1 = plt.subplot(2, 1, 1)
                df['close'].iloc[-100:].plot(ax=ax1, label='Close Price')
                plt.title(f'{self.symbol} Price and Model Predictions')
                plt.legend()
                
                # Plot predictions
                ax2 = plt.subplot(2, 1, 2)
                for name in self.ml_models[self.primary_timeframe].keys():
                    if f'{name}_pred' in df.columns:
                        df[f'{name}_pred'].iloc[-100:].plot(ax=ax2, label=f'{name.upper()} Prediction')
                
                plt.legend()
                plt.savefig(f"results/predictions_{self.symbol}_{self.primary_timeframe}.png")
                plt.close()
                
            logger.info("Model visualization complete")
            
        except Exception as e:
            logger.error(f"Error visualizing model performance: {str(e)}")
            logger.exception("Visualization error")
    
    def save_strategy_state(self):
        """
        Save the current state of the strategy
        
        Saves ML models, indicators, and configuration
        """
        try:
            # Create a state dictionary
            state = {
                'symbol': self.symbol,
                'primary_timeframe': self.primary_timeframe,
                'secondary_timeframes': self.secondary_timeframes,
                'config': self.config,
                'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
                'daily_trade_count': self.daily_trade_count,
                'last_trade_date': self.last_trade_date.isoformat() if self.last_trade_date else None
            }
            
            # Save state to JSON
            state_file = f"data/strategy_state_{self.symbol}.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=4)
                
            logger.info(f"Strategy state saved to {state_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving strategy state: {str(e)}")
            return False
    
    def load_strategy_state(self):
        """
        Load the saved state of the strategy
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            state_file = f"data/strategy_state_{self.symbol}.json"
            
            if not os.path.exists(state_file):
                logger.warning(f"No saved state found for {self.symbol}")
                return False
                
            with open(state_file, 'r') as f:
                state = json.load(f)
                
            # Update attributes from state
            if 'config' in state:
                self.config.update(state['config'])
                
            if 'last_trade_time' in state and state['last_trade_time']:
                self.last_trade_time = datetime.fromisoformat(state['last_trade_time'])
                
            if 'daily_trade_count' in state:
                self.daily_trade_count = state['daily_trade_count']
                
            if 'last_trade_date' in state and state['last_trade_date']:
                self.last_trade_date = datetime.fromisoformat(state['last_trade_date']).date()
                
            logger.info(f"Strategy state loaded from {state_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading strategy state: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # This is implemented in a separate backtesting script
    pass 