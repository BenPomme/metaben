import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Add necessary paths
sys.path.append('./python_scripts')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_enhanced_connector")

try:
    from mt5_connector_enhanced import MT5ConnectorEnhanced
except ImportError as e:
    logger.error(f"Failed to import MT5ConnectorEnhanced: {e}")
    logger.error("Make sure you have created the enhanced connector file")
    sys.exit(1)

def test_enhanced_connector():
    """Test the enhanced MT5 connector with data preprocessing"""
    
    # Create output directory for plots
    os.makedirs('test_results', exist_ok=True)
    
    # Create connector instances - one with preprocessing and one without
    connector_with_preprocess = MT5ConnectorEnhanced(preprocess_data=True)
    connector_no_preprocess = MT5ConnectorEnhanced(preprocess_data=False)
    
    # Connect to MT5
    logger.info("Connecting to MT5...")
    if not connector_with_preprocess.connect():
        logger.error("Failed to connect to MT5")
        return
    
    try:
        # Symbol and timeframes to test
        symbol = "EURUSD"
        timeframes = ["H1", "H4", "D1"]
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 90 days of data
        
        logger.info(f"Testing data retrieval for {symbol} from {start_date} to {end_date}")
        
        # Get data without preprocessing
        logger.info("Getting data WITHOUT preprocessing...")
        raw_data = connector_no_preprocess.get_multi_timeframe_data(
            symbol=symbol,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            preprocess=False
        )
        
        # Get data with preprocessing
        logger.info("Getting data WITH preprocessing...")
        processed_data = connector_with_preprocess.get_multi_timeframe_data(
            symbol=symbol,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date
        )
        
        if raw_data is None or processed_data is None:
            logger.error("Failed to retrieve data")
            return
        
        # Compare the data
        for tf in timeframes:
            raw_df = raw_data[tf]
            processed_df = processed_data[tf]
            
            logger.info(f"\n{tf} data comparison:")
            logger.info(f"  Raw data: {len(raw_df)} rows")
            logger.info(f"  Processed data: {len(processed_df)} rows")
            
            # Calculate time differences to identify gaps
            raw_time_diff = raw_df.index.to_series().diff()
            processed_time_diff = processed_df.index.to_series().diff()
            
            # Get expected time difference for this timeframe
            if tf == 'H1':
                expected_diff = pd.Timedelta(hours=1)
            elif tf == 'H4':
                expected_diff = pd.Timedelta(hours=4)
            elif tf == 'D1':
                expected_diff = pd.Timedelta(days=1)
            else:
                expected_diff = pd.Timedelta(hours=1)
            
            # Find gaps
            raw_gaps = raw_time_diff[raw_time_diff > expected_diff * 1.5]
            processed_gaps = processed_time_diff[processed_time_diff > expected_diff * 1.5]
            
            logger.info(f"  Raw data gaps: {len(raw_gaps)}")
            logger.info(f"  Processed data gaps: {len(processed_gaps)}")
            
            # Plot the data for comparison
            plt.figure(figsize=(15, 10))
            
            # Price data
            plt.subplot(2, 1, 1)
            plt.plot(raw_df.index, raw_df['close'], label='Raw Data', color='blue', alpha=0.5)
            plt.plot(processed_df.index, processed_df['close'], label='Processed Data', color='red', alpha=0.5)
            plt.title(f'{symbol} {tf} Close Price Comparison')
            plt.ylabel('Price')
            plt.grid(True)
            plt.legend()
            
            # Gap visualization
            plt.subplot(2, 1, 2)
            # Convert time differences to a more readable format (hours or days)
            if tf == 'D1':
                raw_hours = raw_time_diff.dt.total_seconds() / (24 * 3600)
                processed_hours = processed_time_diff.dt.total_seconds() / (24 * 3600)
                unit = 'days'
            else:
                raw_hours = raw_time_diff.dt.total_seconds() / 3600
                processed_hours = processed_time_diff.dt.total_seconds() / 3600
                unit = 'hours'
            
            plt.plot(raw_df.index[1:], raw_hours[1:], label=f'Raw Data Time Diff ({unit})', 
                    color='blue', alpha=0.5, linestyle=':')
            plt.plot(processed_df.index[1:], processed_hours[1:], label=f'Processed Data Time Diff ({unit})', 
                    color='red', alpha=0.5)
            
            # Add a horizontal line at the expected time difference
            expected_hours = expected_diff.total_seconds() / 3600
            if tf == 'D1':
                expected_hours = expected_hours / 24
            plt.axhline(y=expected_hours, color='green', linestyle='--', 
                        label=f'Expected Time Diff ({expected_hours} {unit})')
            
            plt.title(f'Time Difference Between Consecutive Candles')
            plt.ylabel(f'Time Difference ({unit})')
            plt.ylim(0, min(10 * expected_hours, plt.ylim()[1]))  # Limit y-axis for better visibility
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'test_results/{symbol}_{tf}_comparison.png')
            plt.close()
            
            logger.info(f"  Saved comparison plot to test_results/{symbol}_{tf}_comparison.png")
            
            # Calculate and save technical indicators on both datasets
            for df_type, df in [('raw', raw_df), ('processed', processed_df)]:
                # Add some basic indicators
                df = add_indicators(df)
                
                # Save to CSV for inspection
                df.to_csv(f'test_results/{symbol}_{tf}_{df_type}_with_indicators.csv')
                
                # Plot with indicators
                plt.figure(figsize=(15, 10))
                
                # Price and indicators
                plt.subplot(2, 1, 1)
                plt.plot(df.index, df['close'], label='Close', color='black')
                plt.plot(df.index, df['sma_20'], label='SMA 20', color='blue')
                plt.plot(df.index, df['sma_50'], label='SMA 50', color='green')
                plt.plot(df.index, df['ema_14'], label='EMA 14', color='red')
                plt.title(f'{symbol} {tf} {df_type.capitalize()} Data with Indicators')
                plt.ylabel('Price')
                plt.grid(True)
                plt.legend()
                
                # RSI
                plt.subplot(2, 1, 2)
                plt.plot(df.index, df['rsi_14'], label='RSI 14', color='purple')
                plt.axhline(y=70, color='red', linestyle='--')
                plt.axhline(y=30, color='green', linestyle='--')
                plt.title('RSI Indicator')
                plt.ylabel('RSI')
                plt.grid(True)
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(f'test_results/{symbol}_{tf}_{df_type}_indicators.png')
                plt.close()
                
                logger.info(f"  Saved {df_type} indicators plot to test_results/{symbol}_{tf}_{df_type}_indicators.png")
        
        logger.info("\nTest completed successfully!")
        
    except Exception as e:
        logger.exception(f"Error during testing: {e}")
        
    finally:
        # Disconnect from MT5
        connector_with_preprocess.disconnect()
        logger.info("Disconnected from MT5")

def add_indicators(df):
    """Add technical indicators to the dataframe"""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # Exponential Moving Average
    df['ema_14'] = df['close'].ewm(span=14, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    
    return df

if __name__ == "__main__":
    test_enhanced_connector() 