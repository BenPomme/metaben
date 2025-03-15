"""
Simplified test script for the ML-enhanced strategy
This version doesn't rely on Pydantic for configuration validation
"""
import asyncio
import json
from datetime import datetime, timedelta
import pandas as pd
import traceback
import os
import logging

from mt5_connector import MT5Connector
from ml_enhanced_strategy import MLEnhancedStrategy

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_ml_strategy_simple")

async def test_strategy():
    """Test the ML-enhanced strategy with a simplified approach"""
    logger.info("Initializing strategy test...")
    
    # Initialize MT5 connector
    connector = MT5Connector()
    connected = connector.connect()
    if not connected:
        logger.error("Failed to connect to MT5")
        return
    
    logger.info("Connected to MT5 successfully")
    
    try:
        # Initialize strategy with direct parameters
        logger.info("Initializing ML-enhanced strategy...")
        strategy = MLEnhancedStrategy(
            symbol="EURUSD",
            primary_timeframe="H1",
            secondary_timeframes=["H4", "D1"],
            mt5_connector=connector
        )
        
        # Override configuration parameters directly
        strategy.feature_window = 20
        strategy.prediction_threshold = 0.65
        
        # Get historical data for training
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # 60 days for faster testing
        
        logger.info(f"Loading historical data from {start_date} to {end_date}...")
        data = {}
        for timeframe in ["H1", "H4", "D1"]:
            logger.info(f"Fetching {timeframe} data...")
            df = connector.get_data(
                symbol="EURUSD",
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            if df is not None and not df.empty:
                logger.info(f"{timeframe} data loaded: {len(df)} candles")
                logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
                data[timeframe] = df
            else:
                logger.error(f"Failed to load {timeframe} data")
                return
        
        # Set the data in the strategy
        logger.info("Processing data...")
        strategy.data = data
        
        # Train ML model
        logger.info("Training ML model...")
        success = strategy.train_ml_model(data["H1"])
        
        if not success:
            logger.error("Failed to train ML model")
            return
        
        # Test ML signal generation
        logger.info("Testing ML signal generation...")
        try:
            ml_signal, ml_strength = strategy.calculate_ml_signal()
            logger.info(f"ML Signal: {ml_signal} (Strength: {ml_strength:.2%})")
        except Exception as e:
            logger.error(f"Error during ML signal generation: {str(e)}")
            logger.debug(traceback.format_exc())
            return
        
        # Test combined signal generation
        logger.info("Testing combined signal generation...")
        try:
            final_signal, final_strength = await asyncio.wait_for(
                strategy.calculate_multi_timeframe_signal(),
                timeout=30.0  # 30 second timeout
            )
            logger.info(f"Final Signal: {final_signal} (Strength: {final_strength:.2%})")
        except asyncio.TimeoutError:
            logger.error("Signal generation timed out")
            return
        except Exception as e:
            logger.error(f"Error during signal generation: {str(e)}")
            logger.debug(traceback.format_exc())
            return
        
        # Test trade parameter generation
        logger.info("Testing trade parameter generation...")
        try:
            trade_params = strategy.generate_trade_parameters()
            if trade_params:
                logger.info("Trade Parameters:")
                for key, value in trade_params.items():
                    logger.info(f"- {key}: {value}")
            else:
                logger.info("No trade parameters generated")
        except Exception as e:
            logger.error(f"Error generating trade parameters: {str(e)}")
            logger.debug(traceback.format_exc())
        
        # Test completed successfully
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        logger.debug(traceback.format_exc())
    finally:
        # Disconnect from MT5
        connector.disconnect()
        logger.info("Disconnected from MT5")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_strategy()) 