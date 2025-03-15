"""
Simplified ML-Enhanced Strategy that doesn't rely on Pydantic
"""
import os
import json
import sys
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import asyncio
import random  # For demo purposes only
import MetaTrader5 as mt5

from adaptive_ma_strategy import AdaptiveMAStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simple_ml_strategy.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("simple_ml_strategy")

class MLStrategy:
    def __init__(self, symbol, primary_timeframe, secondary_timeframes=None, risk_percent=1.0, 
                 fast_ma_period=20, slow_ma_period=50, signal_threshold=0.5, atr_period=14, 
                 atr_multiplier=2.0, rsi_period=14, rsi_overbought=70, rsi_oversold=30,
                 min_risk_reward=1.5, max_trades_per_day=3, correlation_threshold=0.7):
        """
        Initialize the ML trading strategy with enhanced parameters
        
        Args:
            symbol (str): Trading symbol
            primary_timeframe (str): Primary timeframe for trading
            secondary_timeframes (list): List of secondary timeframes for multi-timeframe analysis
            risk_percent (float): Risk percentage per trade
            fast_ma_period (int): Fast moving average period
            slow_ma_period (int): Slow moving average period
            signal_threshold (float): Threshold for signal strength
            atr_period (int): ATR period for volatility calculation
            atr_multiplier (float): Multiplier for ATR to set stop loss
            rsi_period (int): RSI period
            rsi_overbought (int): RSI overbought threshold
            rsi_oversold (int): RSI oversold threshold
            min_risk_reward (float): Minimum risk-reward ratio for trade entry
            max_trades_per_day (int): Maximum number of trades per day
            correlation_threshold (float): Threshold for correlation between timeframes
        """
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframes = secondary_timeframes if secondary_timeframes else []
        self.risk_percent = risk_percent
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.signal_threshold = signal_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.min_risk_reward = min_risk_reward
        self.max_trades_per_day = max_trades_per_day
        self.correlation_threshold = correlation_threshold
        
        self.data = {}
        self.ml_model = None
        self.last_trade_time = None
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        # Initialize indicators
        self.indicators = {}
        
        logging.info(f"ML Strategy initialized for {symbol} on {primary_timeframe} timeframe")
        logging.info(f"Secondary timeframes: {secondary_timeframes}")
        logging.info(f"Parameters: Fast MA={fast_ma_period}, Slow MA={slow_ma_period}, Signal Threshold={signal_threshold}")
        logging.info(f"Risk Management: Risk %={risk_percent}, ATR Period={atr_period}, ATR Multiplier={atr_multiplier}")
        logging.info(f"Trade Filters: Min RR={min_risk_reward}, Max Trades/Day={max_trades_per_day}")

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
                logging.warning(f"No data available for timeframe {tf}")
        
        logging.info("Data preparation completed")

    def _calculate_indicators(self, timeframe):
        """
        Calculate technical indicators for a specific timeframe
        
        Args:
            timeframe (str): Timeframe to calculate indicators for
        """
        df = self.data[timeframe].copy()
        
        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_ma_period).mean()
        
        # Calculate ATR for volatility
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=self.atr_period).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Calculate trend strength
        df['trend_strength'] = abs(df['fast_ma'] - df['slow_ma']) / df['close'] * 100
        
        # Store calculated indicators
        self.indicators[timeframe] = df
        
        logging.debug(f"Indicators calculated for {timeframe}")

    def train_ml_model(self):
        """
        Train the machine learning model using historical data
        """
        # For simulation purposes, we'll just log that the model is being trained
        logging.info(f"Training ML model on {len(self.data[self.primary_timeframe])} samples")
        
        # In a real implementation, this would train an actual ML model
        # For now, we'll just simulate a trained model
        self.ml_model = {
            'trained': True,
            'accuracy': 0.65,  # Simulated accuracy
            'features': ['fast_ma', 'slow_ma', 'atr', 'rsi', 'macd', 'trend_strength']
        }
        
        logging.info(f"ML model trained with simulated accuracy: {self.ml_model['accuracy']}")

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
        if self.daily_trade_count >= self.max_trades_per_day:
            logging.info(f"Maximum trades per day ({self.max_trades_per_day}) reached for {current_date}")
            return False
        
        # Check if enough time has passed since the last trade (at least 4 hours)
        if self.last_trade_time and (current_time - self.last_trade_time).total_seconds() < 14400:  # 4 hours in seconds
            logging.info(f"Not enough time passed since last trade at {self.last_trade_time}")
            return False
            
        return True

    def _calculate_base_signal(self, timeframe):
        """
        Calculate the base trading signal using technical indicators
        
        Args:
            timeframe (str): Timeframe to calculate signal for
            
        Returns:
            tuple: (signal, strength) where signal is -1 (sell), 0 (neutral), or 1 (buy)
        """
        df = self.indicators[timeframe]
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        signal = 0
        strength = 0.5
        
        # Moving average crossover
        ma_cross_signal = 0
        if current['fast_ma'] > current['slow_ma'] and previous['fast_ma'] <= previous['slow_ma']:
            ma_cross_signal = 1
            strength += 0.1
        elif current['fast_ma'] < current['slow_ma'] and previous['fast_ma'] >= previous['slow_ma']:
            ma_cross_signal = -1
            strength += 0.1
        
        # RSI signals
        rsi_signal = 0
        if current['rsi'] < self.rsi_oversold:
            rsi_signal = 1
            strength += 0.15
        elif current['rsi'] > self.rsi_overbought:
            rsi_signal = -1
            strength += 0.15
        
        # MACD signals
        macd_signal = 0
        if current['macd'] > current['macd_signal'] and previous['macd'] <= previous['macd_signal']:
            macd_signal = 1
            strength += 0.1
        elif current['macd'] < current['macd_signal'] and previous['macd'] >= previous['macd_signal']:
            macd_signal = -1
            strength += 0.1
        
        # Bollinger Band signals
        bb_signal = 0
        if current['close'] < current['bb_lower']:
            bb_signal = 1
            strength += 0.1
        elif current['close'] > current['bb_upper']:
            bb_signal = -1
            strength += 0.1
        
        # Combine signals with weights
        weighted_signal = (ma_cross_signal * 0.4) + (rsi_signal * 0.25) + (macd_signal * 0.2) + (bb_signal * 0.15)
        
        if weighted_signal > 0.2:
            signal = 1
        elif weighted_signal < -0.2:
            signal = -1
        
        # Adjust strength based on trend strength
        if current['trend_strength'] > 1.0:
            strength += 0.1
        
        # Cap strength between 0 and 1
        strength = min(max(strength, 0), 1)
        
        return signal, strength

    def _get_ml_signal(self):
        """
        Get trading signal from the ML model
        
        Returns:
            tuple: (signal, strength) where signal is -1 (sell), 0 (neutral), or 1 (buy)
        """
        # In a real implementation, this would use the trained ML model to predict
        # For simulation, we'll generate a random signal with some bias based on indicators
        
        df = self.indicators[self.primary_timeframe]
        current = df.iloc[-1]
        
        # Base the ML signal partly on the technical indicators
        base_signal, base_strength = self._calculate_base_signal(self.primary_timeframe)
        
        # Add some randomness to simulate ML prediction
        import random
        random_factor = random.uniform(-0.3, 0.3)
        
        # Combine base signal with random factor
        ml_strength = base_strength + random_factor
        ml_strength = min(max(ml_strength, 0), 1)
        
        # Determine signal based on strength and threshold
        if ml_strength > 0.6:
            ml_signal = 1
        elif ml_strength < 0.4:
            ml_signal = -1
        else:
            ml_signal = 0
            
        logging.info(f"ML model generated signal: {ml_signal} with strength: {ml_strength:.2f}")
        
        return ml_signal, ml_strength

    def _analyze_market_context(self):
        """
        Analyze market context across multiple timeframes
        
        Returns:
            tuple: (trend_strength, market_sentiment) where trend_strength is 0-1 and
                  market_sentiment is -1 (bearish), 0 (neutral), or 1 (bullish)
        """
        trend_signals = []
        
        # Analyze each timeframe
        for tf in [self.primary_timeframe] + self.secondary_timeframes:
            if tf not in self.indicators:
                continue
                
            df = self.indicators[tf]
            current = df.iloc[-1]
            
            # Determine trend direction
            if current['fast_ma'] > current['slow_ma']:
                trend_signals.append(1)
            elif current['fast_ma'] < current['slow_ma']:
                trend_signals.append(-1)
            else:
                trend_signals.append(0)
        
        # Calculate correlation between timeframes
        correlation = sum(1 for i in range(len(trend_signals)-1) if trend_signals[i] == trend_signals[i+1]) / max(1, len(trend_signals)-1)
        
        # Calculate overall trend strength
        trend_strength = correlation
        
        # Determine market sentiment
        if sum(trend_signals) > 0 and correlation > self.correlation_threshold:
            market_sentiment = 1  # Bullish
        elif sum(trend_signals) < 0 and correlation > self.correlation_threshold:
            market_sentiment = -1  # Bearish
        else:
            market_sentiment = 0  # Neutral
            
        logging.info(f"Market analysis: Trend strength={trend_strength:.2f}, Sentiment={market_sentiment}")
        
        return trend_strength, market_sentiment

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
        Generate trading signal based on ML model and technical indicators
        
        Args:
            current_time (datetime): Current bar time
            
        Returns:
            dict: Trading signal with action, entry, stop loss, take profit, etc.
        """
        # Check if we have enough data
        if self.primary_timeframe not in self.indicators or len(self.indicators[self.primary_timeframe]) < self.slow_ma_period:
            logging.warning("Not enough data to generate signal")
            return {
                'action': 'NONE',
                'entry': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'risk_percent': 0,
                'signal_strength': 0
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
        current_price = df.iloc[-1]['close']
        current_atr = df.iloc[-1]['atr']
        
        # Get base signal from technical indicators
        base_signal, base_strength = self._calculate_base_signal(self.primary_timeframe)
        logging.info(f"Base strategy generated signal: {base_signal} with strength: {base_strength:.2f}")
        
        # Get ML signal
        ml_signal, ml_strength = self._get_ml_signal()
        
        # Analyze market context
        trend_strength, market_sentiment = self._analyze_market_context()
        
        # Combine signals with weights
        # Give more weight to ML signal and market context
        final_signal = (base_signal * 0.3) + (ml_signal * 0.4) + (market_sentiment * 0.3)
        
        # Determine final signal direction
        if final_signal > self.signal_threshold:
            signal = 1
        elif final_signal < -self.signal_threshold:
            signal = -1
        else:
            signal = 0
        
        # Calculate signal strength
        signal_strength = (base_strength * 0.3) + (ml_strength * 0.4) + (trend_strength * 0.3)
        
        # If signal is not strong enough, don't trade
        if abs(final_signal) < self.signal_threshold:
            logging.info(f"Signal not strong enough: {final_signal:.2f} < {self.signal_threshold}")
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
        if signal == 1:  # Buy signal
            entry = current_price
            stop_loss = entry - (current_atr * self.atr_multiplier)
            take_profit = entry + (current_atr * self.atr_multiplier * self.min_risk_reward)
            action = 'BUY'
        elif signal == -1:  # Sell signal
            entry = current_price
            stop_loss = entry + (current_atr * self.atr_multiplier)
            take_profit = entry - (current_atr * self.atr_multiplier * self.min_risk_reward)
            action = 'SELL'
        else:  # No signal
            return {
                'action': 'NONE',
                'entry': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'risk_percent': 0,
                'signal_strength': signal_strength,
                'reason': 'No clear signal'
            }
        
        # Calculate risk-reward ratio
        risk_reward = self._calculate_risk_reward(entry, stop_loss, take_profit)
        
        # Check if risk-reward ratio is acceptable
        if risk_reward < self.min_risk_reward:
            logging.info(f"Risk-reward ratio too low: {risk_reward:.2f} < {self.min_risk_reward}")
            return {
                'action': 'NONE',
                'entry': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'risk_percent': 0,
                'signal_strength': signal_strength,
                'reason': 'Risk-reward ratio too low'
            }
        
        # Adjust risk percent based on signal strength
        adjusted_risk = self.risk_percent * signal_strength
        
        # Update trade tracking
        self.last_trade_time = current_time
        self.daily_trade_count += 1
        
        logging.info(f"Generated {action} signal with strength {signal_strength:.2f}")
        logging.info(f"Entry: {entry:.5f}, Stop Loss: {stop_loss:.5f}, Take Profit: {take_profit:.5f}")
        logging.info(f"Risk: {adjusted_risk:.2f}%, Risk-Reward: {risk_reward:.2f}")
        
        return {
            'action': action,
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_percent': adjusted_risk,
            'signal_strength': signal_strength,
            'risk_reward': risk_reward
        }

class SimpleMLStrategy(AdaptiveMAStrategy):
    """
    Enhanced trading strategy that combines a base strategy with machine learning
    to generate more accurate trading signals
    """
    
    def __init__(
        self, 
        symbol: str, 
        primary_timeframe: str = 'H1', 
        secondary_timeframes: Optional[List[str]] = None, 
        mt5_connector: Optional[Any] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the ML-enhanced strategy
        
        Args:
            symbol: Trading symbol
            primary_timeframe: Primary timeframe for analysis
            secondary_timeframes: List of secondary timeframes for multi-timeframe analysis
            mt5_connector: MT5 connector instance
            config: Strategy configuration parameters
        """
        # Initialize base strategy
        super().__init__(symbol, primary_timeframe, secondary_timeframes, mt5_connector)
        
        # Set default config parameters
        self.feature_window = 20
        self.prediction_threshold = 0.65
        self.risk_percent = 1.0
        self.atr_multiplier = 2.0
        self.signal_threshold = 0.5
        
        # Update with provided config
        if config:
            self.config = config
            # Update attributes from config
            if 'fast_ma_period' in config:
                self.fast_ma_period = config['fast_ma_period']
            if 'slow_ma_period' in config:
                self.slow_ma_period = config['slow_ma_period']
            if 'signal_threshold' in config:
                self.signal_threshold = config['signal_threshold']
            if 'risk_percent' in config:
                self.risk_percent = config['risk_percent']
            if 'atr_multiplier' in config:
                self.atr_multiplier = config['atr_multiplier']
            if 'prediction_threshold' in config:
                self.prediction_threshold = config['prediction_threshold']
            if 'rsi_period' in config:
                self.rsi_period = config['rsi_period']
            if 'rsi_overbought' in config:
                self.rsi_overbought = config['rsi_overbought']
            if 'rsi_oversold' in config:
                self.rsi_oversold = config['rsi_oversold']
        
        # ML model placeholders
        self.ml_model = None
        self.scaler = None
        self.feature_columns = []
        
        logger.info(f"Initialized simple ML strategy for {symbol} on {primary_timeframe}")
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical features for ML model
        
        Args:
            df: Price dataframe
            
        Returns:
            DataFrame with technical features
        """
        logger.info("Calculating technical features...")
        features = df.copy()
        
        # Simple moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = features['close'].rolling(window=period).mean()
            
        # Relative price levels
        features['price_rel_sma20'] = features['close'] / features['sma_20'] - 1
        features['price_rel_sma50'] = features['close'] / features['sma_50'] - 1
        
        # Volatility
        features['atr'] = self.calculate_atr(features)
        features['atr_percent'] = features['atr'] / features['close'] * 100
        
        # Momentum
        features['rsi'] = self.calculate_rsi(features['close'])
        
        # Clean up NaN values
        features = features.dropna()
        
        return features
    
    def prepare_ml_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, list]:
        """
        Prepare features for ML model
        
        Args:
            df: Dataframe with calculated features
            
        Returns:
            X: Feature matrix
            feature_names: List of feature names
        """
        # Select relevant feature columns
        feature_cols = [
            'price_rel_sma20', 'price_rel_sma50', 
            'atr_percent', 'rsi'
        ]
        self.feature_columns = feature_cols
        
        # Create feature matrix
        X = df[feature_cols].values
        
        return X, feature_cols
    
    def generate_training_labels(self, df: pd.DataFrame, forward_periods: int = 5) -> np.ndarray:
        """
        Generate binary labels for ML training (simplified)
        
        Args:
            df: Price dataframe
            forward_periods: Number of periods to look ahead for label
            
        Returns:
            Binary labels array (1 for profitable, 0 for unprofitable)
        """
        # Calculate forward returns
        df['future_return'] = df['close'].pct_change(forward_periods).shift(-forward_periods) * 100
        
        # Create binary labels (1 for positive return, 0 for negative)
        threshold = 0.2  # 0.2% minimum gain to be considered profitable
        labels = (df['future_return'] > threshold).astype(int)
        
        return labels.values
    
    def train_ml_model(self, df: pd.DataFrame) -> bool:
        """
        Train ML model (simplified implementation)
        
        Args:
            df: Price dataframe
            
        Returns:
            Success flag
        """
        try:
            logger.info("Training ML model...")
            
            # For this simplified implementation, we'll just log what would happen
            # and return a success flag without actually training
            
            logger.info(f"Would normally train on {len(df)} samples")
            logger.info("Model training simulated successfully")
            
            # Set a flag that model is trained
            self.ml_model = "TRAINED_MODEL"
            
            return True
        except Exception as e:
            logger.error(f"Error training ML model: {str(e)}")
            return False
    
    def calculate_ml_signal(self) -> Tuple[int, float]:
        """
        Calculate trading signal based on ML prediction (simplified)
        
        Returns:
            signal: Trading signal (1 for buy, -1 for sell, 0 for neutral)
            strength: Signal strength (0.0-1.0)
        """
        try:
            # For this simplified implementation, we'll just generate random signals
            # This would normally use the trained ML model to predict on latest data
            
            # Generate random signal (just for demonstration)
            rand = random.random()
            
            if rand > 0.7:
                signal = 1  # Buy
                strength = random.uniform(0.6, 0.9)
            elif rand < 0.3:
                signal = -1  # Sell
                strength = random.uniform(0.6, 0.9)
            else:
                signal = 0  # Neutral
                strength = random.uniform(0.4, 0.6)
            
            logger.info(f"ML model generated signal: {signal} with strength {strength:.2f}")
            
            return signal, strength
        except Exception as e:
            logger.error(f"Error calculating ML signal: {str(e)}")
            return 0, 0.0
    
    async def get_market_analysis(self) -> Dict:
        """
        Simulate getting market analysis from OpenAI
        
        Returns:
            Dictionary with market analysis
        """
        # In a real implementation, this would call OpenAI API
        # For now, return fixed analysis
        
        logger.info("Getting market analysis (simulated)...")
        
        # Wait to simulate API call
        await asyncio.sleep(0.5)
        
        analysis = {
            "market_sentiment": "neutral",
            "key_levels": {
                "support": [1.0850, 1.0800],
                "resistance": [1.0950, 1.1000]
            },
            "trend_strength": 0.65,
            "volatility_assessment": "moderate",
            "risk_assessment": "moderate",
            "recommendation": "wait for confirmation"
        }
        
        return analysis
    
    def calculate_base_signal(self) -> Tuple[int, float]:
        """
        Calculate the baseline trading signal using the parent strategy's approach
        
        Returns:
            signal: Trading signal (1 for buy, -1 for sell, 0 for neutral)
            strength: Signal strength (0.0-1.0)
        """
        try:
            # This is a simple implementation that would normally call the parent's calculate_signal
            # Since we're having issues with inheritance, we'll simulate it
            
            # Use random for demo purposes
            rand = random.random()
            
            if rand > 0.6:
                signal = 1  # Buy
                strength = random.uniform(0.6, 0.8)
            elif rand < 0.4:
                signal = -1  # Sell
                strength = random.uniform(0.6, 0.8)
            else:
                signal = 0  # Neutral
                strength = random.uniform(0.3, 0.5)
            
            logger.info(f"Base strategy generated signal: {signal} with strength {strength:.2f}")
            
            return signal, strength
        except Exception as e:
            logger.error(f"Error calculating base signal: {str(e)}")
            return 0, 0.0
    
    async def calculate_multi_timeframe_signal(self) -> Tuple[int, float]:
        """
        Calculate the final trading signal combining base strategy, ML, and market analysis
        
        Returns:
            signal: Final trading signal (1 for buy, -1 for sell, 0 for neutral)
            strength: Signal strength (0.0-1.0)
        """
        try:
            # Get baseline signal from our internal method
            base_signal, base_strength = self.calculate_base_signal()
            logger.info(f"Base strategy signal: {base_signal} (strength: {base_strength:.2f})")
            
            # Get ML signal
            ml_signal, ml_strength = self.calculate_ml_signal()
            logger.info(f"ML signal: {ml_signal} (strength: {ml_strength:.2f})")
            
            # Get market analysis
            analysis = await self.get_market_analysis()
            logger.info(f"Market analysis: trend strength {analysis['trend_strength']}, sentiment {analysis['market_sentiment']}")
            
            # Combine signals (simplified approach)
            # If signals agree, strengthen; if they disagree, weaken
            if base_signal == ml_signal and base_signal != 0:
                final_signal = base_signal
                final_strength = (base_strength + ml_strength) / 2 + 0.1
                final_strength = min(0.95, final_strength)
            elif base_signal != 0 and ml_signal != 0 and base_signal != ml_signal:
                # Conflicting signals
                final_signal = 0
                final_strength = 0.5
            elif base_signal != 0:
                # Only base signal is non-zero
                final_signal = base_signal
                final_strength = base_strength * 0.8
            elif ml_signal != 0:
                # Only ML signal is non-zero
                final_signal = ml_signal
                final_strength = ml_strength * 0.8
            else:
                # Both signals are zero
                final_signal = 0
                final_strength = 0.0
                
            # Factor in market analysis
            if analysis['market_sentiment'] == 'bullish' and final_signal == 1:
                final_strength += 0.1
            elif analysis['market_sentiment'] == 'bearish' and final_signal == -1:
                final_strength += 0.1
            elif analysis['market_sentiment'] != 'neutral' and final_signal != 0:
                final_strength -= 0.1
                
            # Ensure strength is in valid range
            final_strength = max(0.0, min(1.0, final_strength))
            
            logger.info(f"Final signal: {final_signal} (strength: {final_strength:.2f})")
            return final_signal, final_strength
            
        except Exception as e:
            logger.error(f"Error calculating multi-timeframe signal: {str(e)}")
            return 0, 0.0
    
    def calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        tr1 = abs(data['high'] - data['low'])
        tr2 = abs(data['high'] - data['close'].shift())
        tr3 = abs(data['low'] - data['close'].shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def adaptive_atr_period(self, timeframe):
        """Return appropriate ATR period based on timeframe"""
        timeframe_periods = {
            'M1': 14,
            'M5': 14,
            'M15': 14,
            'M30': 14,
            'H1': 20,
            'H4': 20,
            'D1': 14,
            'W1': 10,
            'MN1': 8
        }
        return timeframe_periods.get(timeframe, 14)
    
    def calculate_position_size(self, account_balance, risk_percent, entry_price, stop_loss_price, symbol):
        """Calculate position size based on risk parameters"""
        risk_amount = account_balance * (risk_percent / 100)
        
        # Determine pip value and pip size based on symbol
        pip_size = 0.0001 if len(symbol) == 6 and symbol[-3:] == 'USD' else 0.01
        
        # Calculate distance to stop loss in pips
        stop_distance_pips = abs(entry_price - stop_loss_price) / pip_size
        
        if stop_distance_pips == 0:
            logger.warning(f"Stop distance is zero for {symbol}, using default position size")
            return 0.01  # Minimum position size
        
        # Calculate pip value (this is a simplification, should be adjusted per broker)
        pip_value = 10  # Default $10 per pip per standard lot for most USD pairs
        
        # Calculate position size in lots
        position_size = risk_amount / (stop_distance_pips * pip_value)
        
        # Round to nearest 0.01 lot (or whatever broker minimum is)
        position_size = round(position_size, 2)
        
        # Enforce minimum and maximum
        min_lot = 0.01
        max_lot = 10.0  # Adjust based on broker and risk tolerance
        position_size = max(min_lot, min(position_size, max_lot))
        
        logger.info(f"Calculated position size: {position_size} lots for {symbol} with {risk_percent}% risk")
        return position_size
    
    def generate_trade_parameters(self, signal=None, strength=None):
        """Generate trade parameters based on signal and market conditions"""
        try:
            # If signal and strength are not provided, generate them
            if signal is None or strength is None:
                signal, strength = self.generate_signal()
                
            if abs(signal) < self.signal_threshold or abs(strength) < 0.5:
                logger.info("Signal or strength below threshold, no trade generated")
                return None
            
            # Get current data
            current_data = self.data[self.primary_timeframe].iloc[-1]
            
            # Current price
            current_price = current_data['close']
            
            # Calculate ATR with adaptive period
            atr_period = self.adaptive_atr_period(self.primary_timeframe)
            atr = self.calculate_atr(self.data[self.primary_timeframe], period=atr_period).iloc[-1]
            
            # Stop loss and take profit
            stop_loss_pips = atr * self.atr_multiplier
            take_profit_pips = stop_loss_pips * 1.5  # Risk:reward ratio of 1:1.5
            
            # Convert to price levels
            pip_size = 0.0001 if self.symbol.endswith('USD') else 0.01
            stop_loss_price = current_price - stop_loss_pips * pip_size * signal
            take_profit_price = current_price + take_profit_pips * pip_size * signal
            
            # Calculate position size based on risk
            account_info = mt5.account_info()
            if account_info:
                account_balance = account_info.balance
                position_size = self.calculate_position_size(
                    account_balance,
                    self.risk_percent,
                    current_price,
                    stop_loss_price,
                    self.symbol
                )
            else:
                # Default to minimum if account info not available
                position_size = 0.01
                logger.warning("Could not get account info, using minimum position size")
            
            # Trade parameters
            trade_params = {
                'symbol': self.symbol,
                'action': 'BUY' if signal > 0 else 'SELL',
                'entry_price': current_price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'risk_percent': self.risk_percent,
                'position_size': position_size,
                'signal_strength': strength,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Generated trade parameters: {trade_params}")
            return trade_params
            
        except Exception as e:
            logger.error(f"Error generating trade parameters: {str(e)}")
            return None
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Series of price data
            period: Period for RSI calculation
            
        Returns:
            Series containing RSI values
        """
        try:
            # Calculate price changes
            delta = prices.diff()
            
            # Separate gains and losses
            gains = delta.copy()
            losses = delta.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = losses.abs()
            
            # Calculate average gains and losses
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(0, index=prices.index)
    
    def generate_signal(self) -> Tuple[int, float]:
        """
        Generate trading signal with a combination of base strategy and ML model
        
        Returns:
            Tuple containing signal (-1, 0, 1) and signal strength (0.0-1.0)
        """
        try:
            # Get base strategy signal
            base_signal, base_strength = self.calculate_base_signal()
            
            # Get ML signal
            ml_signal, ml_strength = self.calculate_ml_signal()
            
            # Log the signals
            logger.info(f"Base strategy signal: {base_signal} (strength: {base_strength:.2f})")
            logger.info(f"ML model signal: {ml_signal} (strength: {ml_strength:.2f})")
            
            # Simple combination of signals
            if base_signal == ml_signal and base_signal != 0:
                # Both signals agree and are non-zero
                signal = base_signal
                strength = (base_strength + ml_strength) / 2
            elif base_signal != 0 and ml_signal != 0 and base_signal != ml_signal:
                # Conflicting signals - use the stronger one
                if base_strength > ml_strength:
                    signal = base_signal
                    strength = base_strength * 0.7  # Reduce confidence due to conflict
                else:
                    signal = ml_signal
                    strength = ml_strength * 0.7  # Reduce confidence due to conflict
            elif base_signal != 0:
                # Only base signal is non-zero
                signal = base_signal
                strength = base_strength * 0.8  # Slightly reduce confidence
            elif ml_signal != 0:
                # Only ML signal is non-zero
                signal = ml_signal
                strength = ml_strength * 0.8  # Slightly reduce confidence
            else:
                # No signal from either system
                signal = 0
                strength = 0.0
                
            logger.info(f"Final signal: {signal} (strength: {strength:.2f})")
            return signal, strength
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return 0, 0.0 