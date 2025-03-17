import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Add necessary paths
sys.path.append('./python_scripts')
sys.path.append('./backtests')

try:
    from mt5_connector import MT5Connector
    from backtests.backtest_medallion_strategy import MedallionStrategyBacktester
    from backtests.backtest_ml_strategy import MLStrategyBacktester
except ImportError as e:
    print(f"Failed to import modules: {e}")
    print("Make sure all required modules are in the correct paths")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_backtests")

def run_medallion_backtest(symbol="EURUSD", 
                          timeframe="H1", 
                          secondary_timeframes=None,
                          start_date=None,
                          end_date=None,
                          balance=10000):
    """Run the Medallion strategy backtest"""
    logger.info("Starting Medallion Strategy backtest...")
    
    # Create backtester
    backtester = MedallionStrategyBacktester(
        symbol=symbol,
        primary_timeframe=timeframe,
        secondary_timeframes=secondary_timeframes,
        initial_balance=balance,
        data_start=start_date,
        data_end=end_date
    )
    
    # Download data
    data = backtester.download_data(force_download=True)
    if data is None:
        logger.error("Failed to download data for Medallion strategy")
        return None
    
    # Prepare strategy
    strategy = backtester.prepare_strategy(data)
    if strategy is None:
        logger.error("Failed to prepare Medallion strategy")
        return None
    
    # Run backtest
    results = backtester.run_backtest()
    logger.info("Medallion Strategy backtest completed")
    
    return results

def run_ml_backtest(symbol="EURUSD", 
                   timeframe="H1", 
                   secondary_timeframes=None,
                   start_date=None,
                   end_date=None,
                   balance=10000):
    """Run the ML strategy backtest"""
    logger.info("Starting ML Strategy backtest...")
    
    # Create backtester
    backtester = MLStrategyBacktester(
        symbol=symbol,
        primary_timeframe=timeframe,
        secondary_timeframes=secondary_timeframes,
        initial_balance=balance,
        data_start=start_date,
        data_end=end_date
    )
    
    # Download data
    data = backtester.download_data(force_download=True)
    if data is None:
        logger.error("Failed to download data for ML strategy")
        return None
    
    # Run backtest
    results = backtester.run_backtest()
    logger.info("ML Strategy backtest completed")
    
    return results

def main():
    """Main function"""
    symbol = "EURUSD"
    timeframe = "H1"
    secondary_timeframes = ["H4", "D1"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # Last 6 months
    balance = 10000
    
    logger.info(f"Running backtests for {symbol} from {start_date} to {end_date}")
    
    # Run Medallion strategy backtest
    medallion_results = run_medallion_backtest(
        symbol=symbol,
        timeframe=timeframe,
        secondary_timeframes=secondary_timeframes,
        start_date=start_date,
        end_date=end_date,
        balance=balance
    )
    
    if medallion_results:
        logger.info("Medallion Strategy Results:")
        for metric, value in medallion_results.items():
            logger.info(f"{metric}: {value}")
    
    # Run ML strategy backtest
    ml_results = run_ml_backtest(
        symbol=symbol,
        timeframe=timeframe,
        secondary_timeframes=secondary_timeframes,
        start_date=start_date,
        end_date=end_date,
        balance=balance
    )
    
    if ml_results:
        logger.info("ML Strategy Results:")
        for metric, value in ml_results.items():
            logger.info(f"{metric}: {value}")
    
    # Compare results if both are available
    if medallion_results and ml_results:
        logger.info("Comparing strategy performance:")
        comparison = pd.DataFrame({
            'Medallion': medallion_results,
            'ML': ml_results
        })
        print(comparison)
        
        # Plot equity curves if available
        if 'equity_curve' in medallion_results and 'equity_curve' in ml_results:
            plt.figure(figsize=(12, 6))
            plt.plot(medallion_results['equity_curve'], label='Medallion Strategy')
            plt.plot(ml_results['equity_curve'], label='ML Strategy')
            plt.title('Equity Curve Comparison')
            plt.xlabel('Date')
            plt.ylabel('Equity')
            plt.legend()
            plt.grid(True)
            plt.savefig('results/strategy_comparison.png')
            plt.show()

if __name__ == "__main__":
    main() 