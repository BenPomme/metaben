"""
Test script for the Simplified ML Strategy
This version avoids complex configurations and Pydantic dependencies
"""
import asyncio
import json
from datetime import datetime, timedelta
import pandas as pd
import traceback
import os
import logging

from mt5_connector import MT5Connector
from simple_ml_strategy import SimpleMlStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_simple_ml_strategy")

async def test_strategy():
    """Test the simplified ML strategy"""
    logger.info("Starting test of simplified ML strategy")
    
    # Initialize MT5 connector
    connector = MT5Connector()
    connected = connector.connect()
    if not connected:
        logger.error("Failed to connect to MT5")
        return
    
    logger.info("Connected to MT5 successfully")
    
    try:
        # Initialize strategy
        strategy = SimpleMlStrategy(
            symbol="EURUSD",
            primary_timeframe="H1",
            secondary_timeframes=["H4", "D1"],
            mt5_connector=connector
        )
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 30 days of data
        
        logger.info(f"Loading historical data from {start_date} to {end_date}")
        data = {}
        for timeframe in ["H1", "H4", "D1"]:
            df = connector.get_data(
                symbol="EURUSD",
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            if df is not None and not df.empty:
                logger.info(f"Loaded {len(df)} candles for {timeframe}")
                data[timeframe] = df
            else:
                logger.error(f"Failed to load data for {timeframe}")
                return
        
        # Set data in the strategy
        strategy.data = data
        
        # Train the ML model (simplified)
        success = strategy.train_ml_model(data["H1"])
        if not success:
            logger.error("Failed to train ML model")
            return
        
        logger.info("ML model training completed")
        
        # Calculate ML signal
        ml_signal, ml_strength = strategy.calculate_ml_signal()
        logger.info(f"ML signal: {ml_signal} (strength: {ml_strength:.2f})")
        
        # Calculate combined signal
        final_signal, final_strength = await strategy.calculate_multi_timeframe_signal()
        logger.info(f"Final signal: {final_signal} (strength: {final_strength:.2f})")
        
        # Generate trade parameters
        trade_params = strategy.generate_trade_parameters()
        if trade_params:
            logger.info("Trade parameters:")
            for key, value in trade_params.items():
                logger.info(f"- {key}: {value}")
        else:
            logger.info("No trade parameters generated")
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Disconnect from MT5
        connector.disconnect()
        logger.info("Disconnected from MT5")

def main():
    """Main entry point"""
    try:
        # Run the async test
        asyncio.run(test_strategy())
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 