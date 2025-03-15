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
    
    def __init__(self, symbol, primary_timeframe='H1', 
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
        self.secondary_timeframes = secondary_timeframes or ['H4', 'D1']
        self.mt5_connector = mt5_connector
        
        # Initialize parameters with default values
        self.fast_ma_period = 12
        self.slow_ma_period = 26
        self.signal_ma_period = 9
        self.atr_period = 14
        self.atr_multiplier = 2.0
        self.volatility_lookback = 20
        self.volatility_adjustment = True
        self.risk_percent = 0.5
        self.target_profit_multiplier = 2.0
        self.primary_weight = 0.6
        self.secondary_weights = [0.25, 0.15]
        self.trend_ma_period = 50
        self.trend_filter_enabled = True
        self.confirmation_threshold = 0.7
        
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
        self.data[self.primary_timeframe] = self.mt5_connector.get_data(
            self.symbol, self.primary_timeframe, 
            start_date=start_date, end_date=end_date
        )
        
        # Load data for secondary timeframes
        for tf in self.secondary_timeframes:
            self.data[tf] = self.mt5_connector.get_data(
                self.symbol, tf, 
                start_date=start_date, end_date=end_date
            )
        
        return self.data
    
    def calculate_indicators(self, data):
        """Calculate technical indicators for the given data"""
        df = data.copy()
        
        # Calculate EMAs
        df['fast_ema'] = df['close'].ewm(span=self.fast_ma_period, adjust=False).mean()
        df['slow_ema'] = df['close'].ewm(span=self.slow_ma_period, adjust=False).mean()
        df['trend_ema'] = df['close'].ewm(span=self.trend_ma_period, adjust=False).mean()
        
        # Calculate MACD
        df['macd'] = df['fast_ema'] - df['slow_ema']
        df['signal'] = df['macd'].ewm(span=self.signal_ma_period, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['signal']
        
        # Calculate ATR
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].ewm(span=self.atr_period, adjust=False).mean()
        
        return df
    
    def get_signal(self, data):
        """Get trading signal for the given timeframe data"""
        if data is None or len(data) < self.slow_ma_period:
            return 0, 0.0
            
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        # Get the latest values
        current = df.iloc[-1]
        
        # Determine trend
        trend = 1 if current['close'] > current['trend_ema'] else -1
        
        # Calculate signal
        if current['macd'] > current['signal']:
            signal = 1
        elif current['macd'] < current['signal']:
            signal = -1
        else:
            signal = 0
            
        # Apply trend filter if enabled
        if self.trend_filter_enabled and signal != 0:
            signal = signal if signal == trend else 0
            
        # Calculate signal strength [0-1]
        if signal != 0:
            # Normalize MACD histogram
            hist_max = df['macd_hist'].abs().rolling(window=20).max().iloc[-1]
            strength = min(1.0, abs(current['macd_hist']) / hist_max if hist_max > 0 else 0)
        else:
            strength = 0.0
            
        return signal, strength
    
    def calculate_multi_timeframe_signal(self):
        """Calculate consolidated signal from all timeframes"""
        if not self.data:
            return 0, 0.0
            
        signals = {}
        strengths = {}
        
        # Get signals for each timeframe
        signals[self.primary_timeframe], strengths[self.primary_timeframe] = self.get_signal(self.data[self.primary_timeframe])
        
        for tf in self.secondary_timeframes:
            if tf in self.data:
                signals[tf], strengths[tf] = self.get_signal(self.data[tf])
                
        # Calculate weighted signal
        weights = [self.primary_weight] + self.secondary_weights
        timeframes = [self.primary_timeframe] + self.secondary_timeframes
        
        weighted_signal = 0
        for tf, w in zip(timeframes, weights):
            if tf in signals:
                weighted_signal += signals[tf] * strengths[tf] * w
                
        # Determine final signal
        if abs(weighted_signal) >= self.confirmation_threshold:
            final_signal = 1 if weighted_signal > 0 else -1
        else:
            final_signal = 0
            
        return final_signal, abs(weighted_signal)
    
    def generate_trade_parameters(self):
        """Generate trade parameters if there's a valid signal"""
        if not self.data or self.primary_timeframe not in self.data:
            return None
            
        signal, strength = self.calculate_multi_timeframe_signal()
        if signal == 0:
            return None
            
        # Calculate indicators for current data
        df = self.calculate_indicators(self.data[self.primary_timeframe])
        current_data = df.iloc[-1]
        
        # Get current price and ATR
        entry_price = current_data['close']
        atr = current_data['atr']  # Using ATR instead of TR
        
        if signal == 1:  # Buy
            stop_loss = entry_price - (atr * self.atr_multiplier)
            take_profit = entry_price + (atr * self.atr_multiplier * self.target_profit_multiplier)
        else:  # Sell
            stop_loss = entry_price + (atr * self.atr_multiplier)
            take_profit = entry_price - (atr * self.atr_multiplier * self.target_profit_multiplier)
            
        # Calculate position size based on risk
        account_info = self.mt5_connector.get_account_info()
        if account_info is None:
            return None
            
        risk_amount = account_info['balance'] * (self.risk_percent / 100)
        pip_value = 0.0001  # For EURUSD
        
        # Calculate stop loss in pips
        sl_pips = abs(entry_price - stop_loss) / pip_value
        
        # Calculate lot size
        lot_size = risk_amount / (sl_pips * 10)  # $10 per pip per lot for EURUSD
        lot_size = round(lot_size, 2)  # Round to 2 decimal places
        
        return {
            'signal': signal,
            'strength': strength,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': lot_size,
            'risk_amount': risk_amount,
            'potential_profit': risk_amount * self.target_profit_multiplier
        }
    
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
        signal_type = "BUY" if trade_params['signal'] == 1 else "SELL"
        
        recommendation = f"""
{signal_type} SIGNAL - {self.symbol} - {datetime.now().strftime('%Y-%m-%d %H:%M')}
----------------------------------------------------------------------------------------
Signal Strength: {trade_params['strength']:.2f} ({trade_params['strength']*100:.1f}%)
Current Price: {trade_params['entry_price']:.5f}

TRADE SETUP:
- Entry Price: {trade_params['entry_price']:.5f}
- Stop Loss: {trade_params['stop_loss']:.5f}
- Take Profit: {trade_params['take_profit']:.5f}

POSITION SIZING:
- Risk: {self.risk_percent:.1f}% of account
- Position Size: {trade_params['position_size']:.2f} lots
- Potential Profit: {trade_params['potential_profit']:.2f}

TIMEFRAME ANALYSIS:
- Primary ({self.primary_timeframe}): {'Bullish' if trade_params['signal'] == 1 else 'Bearish'}
"""
        
        # Add additional information about secondary timeframes if available
        for tf in self.secondary_timeframes:
            if tf in self.data and self.data[tf] is not None:
                data_with_indicators = self.calculate_indicators(self.data[tf])
                if data_with_indicators is not None and len(data_with_indicators) > 0:
                    tf_signal = data_with_indicators['signal'].iloc[-1]
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
            symbol=self.symbol,
            order_type='buy' if trade_params['signal'] == 1 else 'sell',
            volume=trade_params['position_size'],
            sl=trade_params['stop_loss'],
            tp=trade_params['take_profit'],
            comment=f"Signal: {trade_params['signal']}, Strength: {trade_params['strength']:.2f}"
        )
        
        return ticket 