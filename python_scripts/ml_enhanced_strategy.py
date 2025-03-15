import os
import json
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from adaptive_ma_strategy import AdaptiveMAStrategy
from utils.logging_config import setup_logging
from utils.config_manager import load_config, MLStrategyConfig
from utils.technical_indicators import calculate_all_indicators
from utils.openai_service import OpenAIService

# Set up logger
logger = setup_logging(__name__)

class MLEnhancedStrategy(AdaptiveMAStrategy):
    """
    Enhanced trading strategy that combines traditional technical analysis with 
    machine learning predictions and OpenAI-powered market analysis.
    """
    
    def __init__(
        self, 
        symbol: str, 
        primary_timeframe: str = 'H1', 
        secondary_timeframes: Optional[List[str]] = None, 
        mt5_connector: Optional[Any] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize ML-enhanced strategy
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            primary_timeframe: Primary timeframe for analysis
            secondary_timeframes: List of secondary timeframes for multi-timeframe analysis
            mt5_connector: MT5 connector instance
            config_path: Path to configuration file (optional)
        """
        # Initialize base strategy
        super().__init__(symbol, primary_timeframe, secondary_timeframes, mt5_connector)
        
        # Load configuration
        self.config = load_config(config_path)
        logger.info(f"Loaded configuration for {symbol} on {primary_timeframe}")
        
        # Initialize ML models
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # Initialize OpenAI service
        self.openai_service = OpenAIService(
            model=self.config.openai_model,
            temperature=self.config.openai_temperature,
            max_retries=self.config.openai_max_retries,
            timeout=self.config.openai_timeout
        )
        
        # Set parameters from config
        self.feature_window = self.config.feature_window
        self.prediction_threshold = self.config.prediction_threshold
        
        # Initialize data storage
        self._data = {}
        self.processed_data = {}
        self.feature_columns = None
        
        logger.info(f"ML-enhanced strategy initialized for {symbol}")
        
    @property
    def data(self) -> Dict[str, pd.DataFrame]:
        """Get the raw data"""
        return self._data
        
    @data.setter
    def data(self, value: Dict[str, pd.DataFrame]) -> None:
        """
        Set the raw data and process it
        
        Args:
            value: Dictionary of DataFrames with timeframes as keys
        """
        self._data = value
        # Process data when it's set
        if value:
            self.processed_data = {}
            for timeframe, df in value.items():
                try:
                    logger.debug(f"Processing data for {timeframe}")
                    self.processed_data[timeframe] = calculate_all_indicators(df)
                    logger.debug(f"Processed {len(df)} rows for {timeframe}")
                except Exception as e:
                    logger.error(f"Error processing data for {timeframe}: {str(e)}")
                    self.processed_data[timeframe] = df.copy()
        
    def prepare_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model
        
        Args:
            data: DataFrame with price data and technical indicators
            
        Returns:
            DataFrame with features for ML model
        """
        if data is None or data.empty:
            logger.warning("Empty data provided to prepare_ml_features")
            return pd.DataFrame()
            
        try:
            # Make sure data is processed
            if 'rsi' not in data.columns:
                logger.debug("Data needs processing for ML features")
                df = calculate_all_indicators(data)
            else:
                df = data.copy()
            
            # Base feature columns
            base_features = [
                'returns', 'volatility', 'relative_volume',
                'price_range', 'body_size', 'upper_shadow', 'lower_shadow',
                'rsi', 'momentum', 'macd', 'macd_signal', 'macd_hist', 'atr'
            ]
            
            # Ensure all base features exist
            for col in base_features:
                if col not in df.columns:
                    logger.warning(f"Feature column '{col}' missing, filling with zeros")
                    df[col] = 0
            
            # Add lagged features
            for col in base_features:
                for i in range(1, 5):  # Add 4 lagged values
                    lag_col = f'{col}_lag_{i}'
                    df[lag_col] = df[col].shift(i)
            
            # Fill NaN values with forward fill then zeros
            df = df.fillna(method='ffill').fillna(0)
            
            # Determine feature columns (exclude price and volume data)
            exclude_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            features = [col for col in df.columns if col not in exclude_cols]
            
            # Store feature column names for consistency
            if self.feature_columns is None or len(self.feature_columns) == 0:
                self.feature_columns = features
                logger.debug(f"Set {len(features)} feature columns for ML model")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {str(e)}")
            return pd.DataFrame()
        
    def train_ml_model(self, training_data: pd.DataFrame, lookforward: int = 20) -> bool:
        """
        Train the ML model on historical data
        
        Args:
            training_data: DataFrame with historical price data
            lookforward: Number of periods to look forward for return calculation
            
        Returns:
            True if training was successful, False otherwise
        """
        if training_data is None or training_data.empty:
            logger.error("No training data available")
            return False
            
        try:
            logger.info(f"Training ML model with {len(training_data)} data points")
            
            # Prepare features
            df = self.prepare_ml_features(training_data)
            if df.empty:
                logger.error("Failed to prepare features for training")
                return False
            
            # Create labels (1 for profitable trades, 0 for unprofitable)
            future_returns = training_data['close'].shift(-lookforward) / training_data['close'] - 1
            labels = (future_returns > 0).astype(int)
            
            # Ensure we have enough valid labels
            valid_indices = ~labels.isna()
            if valid_indices.sum() < 10:
                logger.error(f"Not enough valid labels for training: {valid_indices.sum()}")
                return False
            
            # Get feature columns
            feature_cols = self.feature_columns
            
            # Prepare training data
            X = df[feature_cols].values
            y = labels[df.index].fillna(0)  # Fill NaN values with 0
            
            # Scale features
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train model
            self.rf_model.fit(X_scaled, y)
            
            logger.info("ML model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training ML model: {str(e)}")
            return False
        
    async def get_market_analysis(self) -> Dict[str, Any]:
        """
        Get market analysis from OpenAI
        
        Returns:
            Dictionary with market analysis
        """
        if not self.processed_data or self.primary_timeframe not in self.processed_data:
            logger.warning("No processed data available for market analysis")
            return None
            
        try:
            # Get processed data
            current_data = self.processed_data[self.primary_timeframe].iloc[-1]
            daily_change = (current_data['close'] / current_data['open'] - 1) * 100
            
            # Get market analysis
            analysis = self.openai_service.get_market_analysis(
                symbol=self.symbol,
                price=current_data['close'],
                daily_change=daily_change,
                atr=current_data['atr'],
                rsi=current_data['rsi'],
                macd=current_data['macd']
            )
            
            logger.debug(f"Got market analysis for {self.symbol}: {analysis['sentiment']}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting market analysis: {str(e)}")
            return {
                "sentiment": "neutral",
                "support_levels": [],
                "resistance_levels": [],
                "entry_points": [],
                "risks": [f"Error: {str(e)}"]
            }
            
    def calculate_ml_signal(self) -> Tuple[int, float]:
        """
        Calculate trading signal using ML model
        
        Returns:
            Tuple of (signal direction, signal strength)
            Direction: 1 for buy, -1 for sell, 0 for neutral
            Strength: Value between 0.0 and 1.0
        """
        if not self.processed_data or self.primary_timeframe not in self.processed_data:
            logger.warning("No processed data available for ML signal calculation")
            return 0, 0.0
            
        try:
            # Prepare features
            df = self.prepare_ml_features(self.processed_data[self.primary_timeframe])
            
            # Check if we have feature columns and data
            if df.empty or not self.feature_columns:
                logger.warning("No features available for ML signal calculation")
                return 0, 0.0
            
            # Get latest features
            X = df[self.feature_columns].iloc[-1:].values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get prediction probability
            prob = self.rf_model.predict_proba(X_scaled)[0]
            
            # Determine signal
            if prob[1] > self.prediction_threshold:
                signal = 1  # Buy signal
                strength = prob[1]
            elif prob[0] > self.prediction_threshold:
                signal = -1  # Sell signal
                strength = prob[0]
            else:
                signal = 0  # No signal
                strength = max(prob)
                
            logger.debug(f"ML signal: {signal} (strength: {strength:.2f})")
            return signal, strength
            
        except Exception as e:
            logger.error(f"Error calculating ML signal: {str(e)}")
            return 0, 0.0
            
    async def calculate_multi_timeframe_signal(self) -> Tuple[int, float]:
        """
        Enhanced signal calculation using ML and OpenAI
        
        Returns:
            Tuple of (signal direction, signal strength)
            Direction: 1 for buy, -1 for sell, 0 for neutral
            Strength: Value between 0.0 and 1.0
        """
        try:
            # Get traditional signal
            traditional_signal, traditional_strength = super().calculate_multi_timeframe_signal()
            logger.debug(f"Traditional signal: {traditional_signal} (strength: {traditional_strength:.2f})")
            
            # Get ML signal
            ml_signal, ml_strength = self.calculate_ml_signal()
            logger.debug(f"ML signal: {ml_signal} (strength: {ml_strength:.2f})")
            
            # Get market analysis
            try:
                market_analysis = await self.get_market_analysis()
                logger.debug(f"Market analysis sentiment: {market_analysis['sentiment']}")
            except Exception as e:
                logger.error(f"Error getting market analysis: {str(e)}")
                market_analysis = None
            
            # Combine signals
            if traditional_signal == ml_signal and market_analysis:
                # Both signals agree, enhance strength
                final_signal = traditional_signal
                final_strength = max(traditional_strength, ml_strength)
                logger.info(f"Signals agree: {final_signal} (strength: {final_strength:.2f})")
            elif market_analysis:
                # Signals disagree, use the stronger one
                if traditional_strength > ml_strength:
                    final_signal = traditional_signal
                    final_strength = traditional_strength * 0.8  # Reduce confidence
                else:
                    final_signal = ml_signal
                    final_strength = ml_strength * 0.8  # Reduce confidence
                logger.info(f"Signals disagree, using stronger: {final_signal} (strength: {final_strength:.2f})")
            else:
                # No market analysis, use weighted average
                final_signal = traditional_signal
                final_strength = (traditional_strength + ml_strength) / 2
                logger.info(f"Using weighted average: {final_signal} (strength: {final_strength:.2f})")
                
            return final_signal, final_strength
            
        except Exception as e:
            logger.error(f"Error calculating multi-timeframe signal: {str(e)}")
            return 0, 0.0
        
    def generate_trade_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Enhanced trade parameter generation
        
        Returns:
            Dictionary with trade parameters or None if no trade should be executed
        """
        try:
            # Get basic trade parameters from parent class
            trade_params = super().generate_trade_parameters()
            
            if trade_params is None:
                logger.debug("No trade parameters generated by base strategy")
                return None
                
            # Adjust position size based on ML confidence
            _, ml_strength = self.calculate_ml_signal()
            trade_params['position_size'] *= ml_strength
            
            # Adjust stop loss and take profit based on volatility
            current_data = self.processed_data[self.primary_timeframe].iloc[-1]
            current_volatility = current_data['volatility']
            
            # Make stops wider in high volatility
            volatility_multiplier = 1 + current_volatility
            
            # Apply multiplier to stop loss and take profit
            entry_price = trade_params['entry_price']
            original_stop = trade_params['stop_loss']
            original_tp = trade_params['take_profit']
            
            # Adjust stop loss
            stop_distance = original_stop - entry_price
            trade_params['stop_loss'] = entry_price + (stop_distance * volatility_multiplier)
            
            # Adjust take profit
            tp_distance = original_tp - entry_price
            trade_params['take_profit'] = entry_price + (tp_distance * volatility_multiplier)
            
            logger.info(f"Trade parameters generated: entry={trade_params['entry_price']:.5f}, "
                       f"stop={trade_params['stop_loss']:.5f}, tp={trade_params['take_profit']:.5f}, "
                       f"size={trade_params['position_size']:.2f}")
            
            return trade_params
            
        except Exception as e:
            logger.error(f"Error generating trade parameters: {str(e)}")
            return None 