import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import pandas as pd
import traceback

from mt5_connector import MT5Connector
from ml_enhanced_strategy import MLEnhancedStrategy
from utils.logging_config import setup_logging
from utils.config_manager import load_config, save_config, MLStrategyConfig

# Set up logger
logger = setup_logging(__name__)

async def test_strategy(
    symbol: str = "EURUSD",
    primary_timeframe: str = "H1",
    secondary_timeframes: list = ["H4", "D1"],
    days: int = 90,
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Test the ML-enhanced strategy
    
    Args:
        symbol: Trading symbol
        primary_timeframe: Primary timeframe for analysis
        secondary_timeframes: List of secondary timeframes
        days: Number of days of historical data to use
        config_path: Path to configuration file (optional)
        
    Returns:
        Dictionary with test results
    """
    logger.info("Initializing strategy test...")
    
    # Track results
    results = {
        "success": False,
        "error": None,
        "signal": None,
        "strength": None,
        "trade_params": None,
        "market_analysis": None
    }
    
    # Initialize MT5 connector
    connector = MT5Connector()
    connected = connector.connect()
    if not connected:
        error_msg = "Failed to connect to MT5"
        logger.error(error_msg)
        results["error"] = error_msg
        return results
    
    logger.info("Connected to MT5 successfully")
    
    try:
        # Initialize strategy
        logger.info(f"Initializing ML-enhanced strategy for {symbol}...")
        strategy = MLEnhancedStrategy(
            symbol=symbol,
            primary_timeframe=primary_timeframe,
            secondary_timeframes=secondary_timeframes,
            mt5_connector=connector,
            config_path=config_path
        )
        
        # Get historical data for training
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Loading historical data from {start_date} to {end_date}...")
        data = {}
        for timeframe in [primary_timeframe] + secondary_timeframes:
            logger.info(f"Fetching {timeframe} data...")
            df = connector.get_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            if df is not None and not df.empty:
                logger.info(f"{timeframe} data loaded: {len(df)} candles")
                logger.debug(f"Date range: {df.index[0]} to {df.index[-1]}")
                data[timeframe] = df
            else:
                error_msg = f"Failed to load {timeframe} data"
                logger.error(error_msg)
                results["error"] = error_msg
                return results
        
        # Set the data in the strategy
        logger.info("Processing data...")
        strategy.data = data
        
        # Train ML model
        logger.info("Training ML model...")
        success = strategy.train_ml_model(data[primary_timeframe])
        
        if not success:
            error_msg = "Failed to train ML model"
            logger.error(error_msg)
            results["error"] = error_msg
            return results
        
        # Test ML signal generation
        logger.info("Testing ML signal generation...")
        try:
            ml_signal, ml_strength = strategy.calculate_ml_signal()
            logger.info(f"ML Signal: {ml_signal} (Strength: {ml_strength:.2%})")
        except Exception as e:
            error_msg = f"Error during ML signal generation: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            results["error"] = error_msg
            return results
        
        # Test OpenAI market analysis
        logger.info("Testing OpenAI market analysis...")
        try:
            market_analysis = await asyncio.wait_for(
                strategy.get_market_analysis(),
                timeout=30.0  # 30 second timeout
            )
            if market_analysis:
                logger.info(f"Market Analysis: {json.dumps(market_analysis, indent=2)}")
                results["market_analysis"] = market_analysis
            else:
                logger.warning("No market analysis available")
        except asyncio.TimeoutError:
            logger.warning("Market analysis request timed out")
        except Exception as e:
            logger.error(f"Error during market analysis: {str(e)}")
            logger.debug(traceback.format_exc())
        
        # Test combined signal generation
        logger.info("Testing combined signal generation...")
        try:
            final_signal, final_strength = await asyncio.wait_for(
                strategy.calculate_multi_timeframe_signal(),
                timeout=30.0  # 30 second timeout
            )
            logger.info(f"Final Signal: {final_signal} (Strength: {final_strength:.2%})")
            
            # Store signal results
            results["signal"] = final_signal
            results["strength"] = final_strength
            
        except asyncio.TimeoutError:
            error_msg = "Signal generation timed out"
            logger.error(error_msg)
            results["error"] = error_msg
            return results
        except Exception as e:
            error_msg = f"Error during signal generation: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            results["error"] = error_msg
            return results
        
        # Test trade parameter generation
        logger.info("Testing trade parameter generation...")
        try:
            trade_params = strategy.generate_trade_parameters()
            if trade_params:
                logger.info("Trade Parameters:")
                for key, value in trade_params.items():
                    logger.info(f"- {key}: {value}")
                results["trade_params"] = trade_params
            else:
                logger.info("No trade parameters generated")
        except Exception as e:
            logger.error(f"Error generating trade parameters: {str(e)}")
            logger.debug(traceback.format_exc())
        
        # Test completed successfully
        results["success"] = True
        logger.info("Test completed successfully")
        
    except Exception as e:
        error_msg = f"Error during testing: {str(e)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        results["error"] = error_msg
    finally:
        # Disconnect from MT5
        connector.disconnect()
        logger.info("Disconnected from MT5")
        
    return results

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test ML-enhanced trading strategy")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Trading symbol")
    parser.add_argument("--primary", type=str, default="H1", help="Primary timeframe")
    parser.add_argument("--secondary", type=str, nargs="+", default=["H4", "D1"], help="Secondary timeframes")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()
    
    # Run the test
    asyncio.run(test_strategy(
        symbol=args.symbol,
        primary_timeframe=args.primary,
        secondary_timeframes=args.secondary,
        days=args.days,
        config_path=args.config
    )) 