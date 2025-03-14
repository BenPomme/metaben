"""
Adaptive Moving Average Strategy - Main strategy implementation

This strategy uses multi-timeframe analysis with adaptive parameters based on 
market volatility to generate trading signals.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class AdaptiveMAStrategy:
    """
    Adaptive Moving Average Strategy with multi-timeframe analysis
    """
    
    def __init__(self, symbol, primary_timeframe='1h', 
                 secondary_timeframes=None, mt5_connector=None):
        """
        Initialize the Adaptive MA Strategy
        
        Args:
            symbol: Trading symbol
            primary_timeframe: Main trading timeframe
            secondary_timeframes: List of additional timeframes to analyze
            mt5_connector: MT5Connector instance for live trading
        """
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframes = secondary_timeframes or ['4h', '1d']
        self.mt5_connector = mt5_connector
        
        # Default strategy parameters
        self.params = {
            # Base MA parameters
            'fast_ma_period': 12,
            'slow_ma_period': 26,
            'signal_ma_period': 9,
            
            # ATR parameters
            'atr_period': 14,
            'atr_multiplier': 2.0,
            
            # Volatility adaptation
            'volatility_lookback': 20,
            'volatility_adjustment': True,
            
            # Risk parameters
            'risk_percent': 1.0,
            'target_profit_multiplier': 2.0,  # Risk:Reward ratio
            
            # Multi-timeframe weights
            'primary_weight': 0.6,
            'secondary_weights': [0.25, 0.15],
            
            # Trend filter
            'trend_ma_period': 50,
            'trend_filter_enabled': True,
            
            # Confirmation
            'confirmation_threshold': 0.7  # Minimum signal strength [0-1]
        }
        
        # Store data for each timeframe
        self.data = {}
        
        # Current signals
        self.current_signal = 0  # -1: Sell, 0: Neutral, 1: Buy
        self.signal_strength = 0.0  # Signal strength [0-1]
        
    def load_data(self, start_date=None, end_date=None, count=1000):
        """
        Load data for all timeframes
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            count: Number of bars to retrieve
            
        Returns:
            dict: Dictionary of dataframes for each timeframe
        """
        if self.mt5_connector is None or not self.mt5_connector.connected:
            print("MT5 connector not available. Cannot load data.")
            return None
        
        # Load data for primary timeframe
        self.data[self.primary_timeframe] = self.mt5_connector.get_historical_data(
            self.symbol, self.primary_timeframe, 
            start_time=start_date, end_time=end_date, count=count
        )
        
        # Load data for secondary timeframes
        for tf in self.secondary_timeframes:
            self.data[tf] = self.mt5_connector.get_historical_data(
                self.symbol, tf, 
                start_time=start_date, end_time=end_date, count=count
            )
        
        return self.data
    
    def calculate_indicators(self, data=None, timeframe=None):
        """
        Calculate indicators for a specific timeframe
        
        Args:
            data: Data to calculate indicators on (optional, uses stored data if None)
            timeframe: Timeframe to calculate for (optional, uses primary if None)
            
        Returns:
            DataFrame: Data with indicators
        """
        if data is None:
            if timeframe is None:
                timeframe = self.primary_timeframe
            
            if timeframe not in self.data or self.data[timeframe] is None:
                print(f"No data available for {timeframe}")
                return None
            
            data = self.data[timeframe].copy()
        
        # Copy data to avoid modifying original
        df = data.copy()
        
        # Get parameters for calculations
        fast_period = self.params['fast_ma_period']
        slow_period = self.params['slow_ma_period']
        signal_period = self.params['signal_ma_period']
        atr_period = self.params['atr_period']
        trend_period = self.params['trend_ma_period']
        
        # Adjust periods based on volatility if enabled
        if self.params['volatility_adjustment']:
            # Calculate price volatility (standard deviation of returns)
            returns = df['close'].pct_change(1).dropna()
            volatility = returns.rolling(self.params['volatility_lookback']).std().iloc[-1]
            
            # Adjust periods - shorter during high volatility, longer during low volatility
            # Define volatility ranges (these would need to be tuned for the specific instrument)
            avg_volatility = 0.01  # 1% daily volatility is considered average
            
            # If volatility is higher than average, decrease periods
            # If volatility is lower than average, increase periods
            volatility_ratio = volatility / avg_volatility
            
            # Adjust with a maximum change of ±40%
            adjustment_factor = max(0.6, min(1.4, volatility_ratio))
            
            # Adjust periods (ensure they remain integers)
            fast_period = max(5, int(fast_period / adjustment_factor))
            slow_period = max(10, int(slow_period / adjustment_factor))
            signal_period = max(5, int(signal_period / adjustment_factor))
        
        # Calculate Exponential Moving Averages
        df['fast_ema'] = df['close'].ewm(span=fast_period, adjust=False).mean()
        df['slow_ema'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line and signal line
        df['macd_line'] = df['fast_ema'] - df['slow_ema']
        df['signal_line'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['signal_line']
        
        # Calculate trend direction
        df['trend_ma'] = df['close'].rolling(window=trend_period).mean()
        df['trend_direction'] = np.where(df['close'] > df['trend_ma'], 1, -1)
        
        # Calculate ATR for stop loss and take profit
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=atr_period).mean()
        
        # Generate raw signals
        df['raw_signal'] = 0.0
        
        # Determine signal direction
        df.loc[(df['macd_line'] > df['signal_line']), 'raw_signal'] = 1.0
        df.loc[(df['macd_line'] < df['signal_line']), 'raw_signal'] = -1.0
        
        # Determine signal strength based on histogram magnitude
        # Normalize histogram by using a rolling max to scale between 0-1
        df['hist_abs'] = np.abs(df['macd_histogram'])
        df['hist_max'] = df['hist_abs'].rolling(50).max()
        df['signal_strength'] = df['hist_abs'] / df['hist_max'].replace(0, 1)
        
        # Apply trend filter if enabled
        if self.params['trend_filter_enabled']:
            df['filtered_signal'] = df['raw_signal'] * np.where(
                df['raw_signal'] == df['trend_direction'], 1, 0)
        else:
            df['filtered_signal'] = df['raw_signal']
        
        # Calculate stop loss and take profit levels
        df['stop_loss_long'] = df['close'] - (df['atr'] * self.params['atr_multiplier'])
        df['stop_loss_short'] = df['close'] + (df['atr'] * self.params['atr_multiplier'])
        df['take_profit_long'] = df['close'] + (df['atr'] * self.params['atr_multiplier'] * 
                                            self.params['target_profit_multiplier'])
        df['take_profit_short'] = df['close'] - (df['atr'] * self.params['atr_multiplier'] * 
                                             self.params['target_profit_multiplier'])
        
        return df
    
    def calculate_multi_timeframe_signal(self):
        """
        Calculate a consolidated signal from all timeframes
        
        Returns:
            tuple: (signal, strength) where signal is -1 (sell), 0 (neutral), or 1 (buy)
                  and strength is 0-1.
        """
        signals = {}
        strengths = {}
        
        # Calculate signals for each timeframe
        for i, tf in enumerate([self.primary_timeframe] + self.secondary_timeframes):
            if tf in self.data and self.data[tf] is not None:
                data_with_indicators = self.calculate_indicators(timeframe=tf)
                
                if data_with_indicators is not None and len(data_with_indicators) > 0:
                    # Get the most recent signal
                    signals[tf] = data_with_indicators['filtered_signal'].iloc[-1]
                    strengths[tf] = data_with_indicators['signal_strength'].iloc[-1]
        
        # If we don't have signals for all timeframes, return neutral
        if len(signals) != len([self.primary_timeframe] + self.secondary_timeframes):
            return 0, 0.0
        
        # Get weights for each timeframe
        weights = [self.params['primary_weight']] + self.params['secondary_weights']
        
        # Calculate weighted signal
        weighted_signal = 0
        for i, tf in enumerate([self.primary_timeframe] + self.secondary_timeframes):
            weighted_signal += signals[tf] * strengths[tf] * weights[i]
        
        # Determine final signal type
        if weighted_signal > self.params['confirmation_threshold']:
            signal = 1
        elif weighted_signal < -self.params['confirmation_threshold']:
            signal = -1
        else:
            signal = 0
        
        # Calculate signal strength [0-1]
        strength = abs(weighted_signal)
        
        return signal, strength
    
    def generate_trade_parameters(self):
        """
        Generate trade parameters including entry, stop loss, and take profit
        
        Returns:
            dict: Trade parameters
        """
        # Make sure we have data for the primary timeframe
        if self.primary_timeframe not in self.data or self.data[self.primary_timeframe] is None:
            print("No data available for the primary timeframe")
            return None
        
        # Calculate signal
        self.current_signal, self.signal_strength = self.calculate_multi_timeframe_signal()
        
        # If no signal, return None
        if self.current_signal == 0:
            return None
        
        # Get the indicator data
        primary_data = self.calculate_indicators(timeframe=self.primary_timeframe)
        
        # Get current price and ATR
        current_price = primary_data['close'].iloc[-1]
        current_atr = primary_data['atr'].iloc[-1]
        
        # Calculate stop loss and take profit levels
        if self.current_signal == 1:  # Buy signal
            entry_price = current_price
            stop_loss = primary_data['stop_loss_long'].iloc[-1]
            take_profit = primary_data['take_profit_long'].iloc[-1]
        else:  # Sell signal
            entry_price = current_price
            stop_loss = primary_data['stop_loss_short'].iloc[-1]
            take_profit = primary_data['take_profit_short'].iloc[-1]
        
        # Calculate stop loss distance in pips
        pip_size = 0.0001 if len(str(int(current_price)).split('.')[0]) <= 2 else 0.01
        stop_distance_pips = abs(entry_price - stop_loss) / pip_size
        
        # Calculate position size based on risk
        position_size = 0.01  # Default minimum
        if self.mt5_connector is not None and self.mt5_connector.connected:
            position_size = self.mt5_connector.calculate_lot_size(
                self.symbol, 
                self.params['risk_percent'], 
                stop_distance_pips
            )
        
        # Generate trade parameters
        trade_params = {
            'symbol': self.symbol,
            'order_type': 'buy' if self.current_signal == 1 else 'sell',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'signal_strength': self.signal_strength,
            'risk_percent': self.params['risk_percent'],
            'risk_reward_ratio': self.params['target_profit_multiplier'],
        }
        
        return trade_params
    
    def get_trade_recommendation(self):
        """
        Get human-readable trade recommendation
        
        Returns:
            str: Trade recommendation
        """
        # Generate trade parameters
        trade_params = self.generate_trade_parameters()
        
        if trade_params is None:
            return "No trade signal at this time. Wait for better conditions."
        
        # Format the recommendation
        signal_type = "BUY" if trade_params['order_type'] == 'buy' else "SELL"
        
        recommendation = f"""
{signal_type} SIGNAL - {self.symbol} - {datetime.now().strftime('%Y-%m-%d %H:%M')}
----------------------------------------------------------------------------------------
Signal Strength: {trade_params['signal_strength']:.2f} ({trade_params['signal_strength']*100:.1f}%)
Current Price: {trade_params['entry_price']:.5f}

TRADE SETUP:
- Entry Price: {trade_params['entry_price']:.5f}
- Stop Loss: {trade_params['stop_loss']:.5f}
- Take Profit: {trade_params['take_profit']:.5f}

POSITION SIZING:
- Risk: {trade_params['risk_percent']:.1f}% of account
- Position Size: {trade_params['position_size']:.2f} lots
- Risk/Reward Ratio: 1:{trade_params['risk_reward_ratio']:.1f}

TIMEFRAME ANALYSIS:
- Primary ({self.primary_timeframe}): {'Bullish' if self.current_signal == 1 else 'Bearish'}
"""
        
        # Add additional information about secondary timeframes if available
        for tf in self.secondary_timeframes:
            if tf in self.data and self.data[tf] is not None:
                data_with_indicators = self.calculate_indicators(timeframe=tf)
                if data_with_indicators is not None and len(data_with_indicators) > 0:
                    tf_signal = data_with_indicators['filtered_signal'].iloc[-1]
                    tf_direction = 'Bullish' if tf_signal > 0 else 'Bearish' if tf_signal < 0 else 'Neutral'
                    recommendation += f"- {tf}: {tf_direction}\n"
        
        return recommendation
    
    def execute_trade(self):
        """
        Execute a trade based on the current signal
        
        Returns:
            int: Trade ticket if successful, None otherwise
        """
        if self.mt5_connector is None or not self.mt5_connector.connected:
            print("MT5 connector not available. Cannot execute trade.")
            return None
        
        # Generate trade parameters
        trade_params = self.generate_trade_parameters()
        
        if trade_params is None:
            print("No trade signal. Not executing any trade.")
            return None
        
        # Execute the trade
        ticket = self.mt5_connector.open_position(
            symbol=trade_params['symbol'],
            order_type=trade_params['order_type'],
            volume=trade_params['position_size'],
            sl=trade_params['stop_loss'],
            tp=trade_params['take_profit'],
            comment=f"Signal: {self.current_signal}, Strength: {self.signal_strength:.2f}"
        )
        
        return ticket 