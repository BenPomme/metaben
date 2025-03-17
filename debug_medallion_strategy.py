import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Add necessary paths
sys.path.append('./python_scripts')
sys.path.append('./backtests')

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum info
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_medallion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("debug_medallion")

try:
    from mt5_connector import MT5Connector
    from backtests.backtest_medallion_strategy import MedallionStrategyBacktester
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    logger.error("Make sure all required modules are in the correct paths")
    sys.exit(1)

def debug_medallion_strategy():
    """Debug the Medallion strategy on a small data range"""
    symbol = "EURUSD"
    timeframe = "H1"
    secondary_timeframes = ["H4", "D1"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Just last 30 days for faster debugging
    balance = 10000
    
    logger.info(f"Debugging Medallion Strategy for {symbol} from {start_date} to {end_date}")
    
    try:
        # Create backtester
        backtester = MedallionStrategyBacktester(
            symbol=symbol,
            primary_timeframe=timeframe,
            secondary_timeframes=secondary_timeframes,
            initial_balance=balance,
            data_start=start_date,
            data_end=end_date
        )
        
        # Download data with force download to get fresh data
        logger.info("Downloading data...")
        data = backtester.download_data(force_download=True)
        
        if data is None:
            logger.error("Failed to download data")
            return
        
        # Log data information
        for tf in [timeframe] + secondary_timeframes:
            if tf in data and not data[tf].empty:
                logger.info(f"{tf} data: {len(data[tf])} candles from {data[tf].index[0]} to {data[tf].index[-1]}")
                # Check for gaps
                time_diff = data[tf].index.to_series().diff()
                expected_diff = pd.Timedelta(hours=1) if tf == "H1" else \
                               (pd.Timedelta(hours=4) if tf == "H4" else pd.Timedelta(days=1))
                gaps = time_diff[time_diff > expected_diff]
                if not gaps.empty:
                    logger.warning(f"Found {len(gaps)} gaps in {tf} data")
                    for time, diff in gaps.items():
                        logger.warning(f"Gap at {time}: {diff}")
            else:
                logger.error(f"No data for {tf} timeframe or data is empty")
                return
        
        # Prepare strategy
        logger.info("Preparing strategy...")
        strategy = backtester.prepare_strategy(data)
        
        if strategy is None:
            logger.error("Failed to prepare strategy")
            return
        
        # Run backtest with more debug information
        logger.info("Running backtest...")
        results = backtester.run_backtest()
        
        if results:
            logger.info("Backtest results:")
            for metric, value in results.items():
                if metric != 'equity_curve':  # Skip printing the whole equity curve
                    logger.info(f"{metric}: {value}")
            
            # Save the results to file
            results_dir = 'debug_results'
            os.makedirs(results_dir, exist_ok=True)
            
            # Plot and save equity curve
            if 'equity_curve' in results:
                plt.figure(figsize=(10, 6))
                plt.plot(results['equity_curve'])
                plt.title('Medallion Strategy Equity Curve')
                plt.xlabel('Date')
                plt.ylabel('Equity')
                plt.grid(True)
                plt.savefig(f'{results_dir}/medallion_equity_curve.png')
                plt.close()
                
                # Save equity curve to csv
                results['equity_curve'].to_csv(f'{results_dir}/medallion_equity_curve.csv')
                
            # Save trades to csv if available
            if 'trades' in results:
                trades_df = pd.DataFrame(results['trades'])
                trades_df.to_csv(f'{results_dir}/medallion_trades.csv', index=False)
                
        else:
            logger.error("No results returned from backtest")
            
    except Exception as e:
        logger.exception(f"Exception occurred during backtesting: {e}")

if __name__ == "__main__":
    debug_medallion_strategy() 