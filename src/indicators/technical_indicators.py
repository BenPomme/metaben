import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

from utils.logging_config import setup_logging

# Set up logger
logger = setup_logging(__name__)

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        prices: Series of price data
        period: Period for RSI calculation
        
    Returns:
        Series containing RSI values
    """
    try:
        # Ensure we have enough data
        if len(prices) < period + 1:
            logger.warning(f"Not enough data for RSI calculation. Need at least {period + 1} data points.")
            return pd.Series(index=prices.index, data=50.0)  # Default neutral RSI
        
        # Calculate price changes
        delta = prices.diff()
        
        # Create gain and loss series
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Handle division by zero
        avg_loss = avg_loss.replace(0, 1e-10)  # Small non-zero value to prevent division by zero
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return pd.Series(index=prices.index, data=50.0)  # Return neutral RSI on error


def calculate_macd(
    prices: pd.Series, 
    fast_period: int = 12, 
    slow_period: int = 26, 
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD)
    
    Args:
        prices: Series of price data
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
        signal_period: Period for signal line
        
    Returns:
        Tuple of (MACD line, signal line, histogram)
    """
    try:
        # Ensure we have enough data
        min_data_points = max(fast_period, slow_period, signal_period) + 10
        if len(prices) < min_data_points:
            logger.warning(f"Not enough data for MACD calculation. Need at least {min_data_points} data points.")
            zeros = pd.Series(0, index=prices.index)
            return zeros, zeros, zeros
        
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
        
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        zeros = pd.Series(0, index=prices.index)
        return zeros, zeros, zeros


def calculate_atr(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        period: Period for ATR calculation
        
    Returns:
        Series containing ATR values
    """
    try:
        # Ensure we have enough data
        if len(high) < period + 1 or len(low) < period + 1 or len(close) < period + 1:
            logger.warning(f"Not enough data for ATR calculation. Need at least {period + 1} data points.")
            return pd.Series(0, index=high.index)
        
        # Calculate true range
        close_prev = close.shift(1)
        true_range = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=period).mean()
        
        return atr
        
    except Exception as e:
        logger.error(f"Error calculating ATR: {str(e)}")
        return pd.Series(0, index=high.index)


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price action features to DataFrame
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with added price action features
    """
    try:
        # Create copy to avoid modifying the original
        result = df.copy()
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in result.columns:
                logger.error(f"Required column '{col}' not found in DataFrame")
                return df
        
        # Returns and volatility
        result['returns'] = result['close'].pct_change()
        result['volatility'] = result['returns'].rolling(window=20).std()
        
        # Price patterns
        result['price_range'] = result['high'] - result['low']
        result['body_size'] = (result['close'] - result['open']).abs()
        result['upper_shadow'] = result['high'] - result[['open', 'close']].max(axis=1)
        result['lower_shadow'] = result[['open', 'close']].min(axis=1) - result['low']
        
        # Momentum
        result['momentum'] = result['close'] - result['close'].shift(10)
        
        return result
        
    except Exception as e:
        logger.error(f"Error adding price action features: {str(e)}")
        return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based features to DataFrame
    
    Args:
        df: DataFrame with price and volume data
        
    Returns:
        DataFrame with added volume features
    """
    try:
        # Create copy to avoid modifying the original
        result = df.copy()
        
        # Ensure required columns exist
        if 'tick_volume' not in result.columns:
            logger.warning("Column 'tick_volume' not found in DataFrame")
            return df
        
        # Volume features
        result['volume_ma'] = result['tick_volume'].rolling(window=20).mean()
        result['volume_std'] = result['tick_volume'].rolling(window=20).std()
        
        # Prevent division by zero
        result['volume_ma'] = result['volume_ma'].replace(0, 1e-10)
        
        # Relative volume
        result['relative_volume'] = result['tick_volume'] / result['volume_ma']
        
        return result
        
    except Exception as e:
        logger.error(f"Error adding volume features: {str(e)}")
        return df


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators and features
    
    Args:
        df: DataFrame with price and volume data
        
    Returns:
        DataFrame with all calculated indicators
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided to calculate_all_indicators")
        return pd.DataFrame()
        
    try:
        # Add price action features
        df = add_price_action_features(df)
        
        # Add volume features
        df = add_volume_features(df)
        
        # Add momentum indicators
        df['rsi'] = calculate_rsi(df['close'])
        
        # Add trend indicators
        macd, signal, hist = calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # Add volatility indicators
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
        
        # Fill NaN values with forward fill, then backward fill, then zeros
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        df = df.fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return df 