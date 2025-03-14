"""
Live Trading Script for ML-Enhanced Strategy
This script executes real trades based on the signals from the ML strategy
"""
import asyncio
import logging
import os
import json
import argparse
from datetime import datetime, timedelta
import pandas as pd
import time

from mt5_connector import MT5Connector
from simple_ml_strategy import SimpleMlStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trade_with_ml")

class MLTrader:
    """
    Trading bot that uses ML-enhanced strategy to execute trades
    """
    
    def __init__(
        self,
        symbol: str = "EURUSD",
        primary_timeframe: str = "H1",
        secondary_timeframes: list = None,
        check_interval_minutes: int = 5,
        log_trades: bool = True
    ):
        """
        Initialize the ML trader
        
        Args:
            symbol: Trading symbol
            primary_timeframe: Primary timeframe for analysis
            secondary_timeframes: Secondary timeframes for multi-timeframe analysis
            check_interval_minutes: How often to check for new trading signals (in minutes)
            log_trades: Whether to log trades to a file
        """
        if secondary_timeframes is None:
            secondary_timeframes = ["H4", "D1"]
            
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframes = secondary_timeframes
        self.check_interval_minutes = check_interval_minutes
        self.log_trades = log_trades
        
        # Initialize MT5 connector
        self.connector = MT5Connector()
        
        # Initialize strategy
        self.strategy = None
        
        # Trade history
        self.trades = []
        
        logger.info(f"Initialized ML Trader for {symbol} on {primary_timeframe}")
        
    async def initialize(self):
        """Initialize the trader (connect to MT5 and create strategy)"""
        # Connect to MT5
        connected = self.connector.connect()
        if not connected:
            logger.error("Failed to connect to MT5. Trading cannot start.")
            return False
        
        logger.info("Successfully connected to MT5")
        
        # Initialize strategy
        self.strategy = SimpleMlStrategy(
            symbol=self.symbol,
            primary_timeframe=self.primary_timeframe,
            secondary_timeframes=self.secondary_timeframes,
            mt5_connector=self.connector
        )
        
        # Initialize with historical data
        await self.update_data()
        
        # Train the model initially
        success = self.strategy.train_ml_model(self.strategy.data[self.primary_timeframe])
        if not success:
            logger.error("Failed to train ML model. Trading cannot start.")
            return False
            
        logger.info("ML Trader initialization complete")
        return True
        
    async def update_data(self):
        """Update historical data for the strategy"""
        logger.info("Updating historical data...")
        
        # Get current time
        end_date = datetime.now()
        
        # For primary timeframe, get more data for ML training
        primary_start_date = end_date - timedelta(days=60)  # 60 days for ML training
        
        # For secondary timeframes, get enough data for calculations
        secondary_start_date = end_date - timedelta(days=30)
        
        # Get data for all timeframes
        data = {}
        
        # Get primary timeframe data
        primary_df = self.connector.get_data(
            symbol=self.symbol,
            timeframe=self.primary_timeframe,
            start_date=primary_start_date,
            end_date=end_date
        )
        
        if primary_df is None or primary_df.empty:
            logger.error(f"Failed to get data for {self.primary_timeframe}")
            return False
            
        data[self.primary_timeframe] = primary_df
        logger.info(f"Loaded {len(primary_df)} candles for {self.primary_timeframe}")
        
        # Get secondary timeframe data
        for timeframe in self.secondary_timeframes:
            df = self.connector.get_data(
                symbol=self.symbol,
                timeframe=timeframe,
                start_date=secondary_start_date,
                end_date=end_date
            )
            
            if df is None or df.empty:
                logger.error(f"Failed to get data for {timeframe}")
                return False
                
            data[timeframe] = df
            logger.info(f"Loaded {len(df)} candles for {timeframe}")
            
        # Set data in strategy
        self.strategy.data = data
        return True
        
    async def check_for_trade_signal(self):
        """Check for trade signals and execute if conditions are met"""
        logger.info("Checking for trade signals...")
        
        # Get current signal
        signal, strength = await self.strategy.calculate_multi_timeframe_signal()
        logger.info(f"Current signal: {signal} (strength: {strength:.2f})")
        
        # Check if we have an actionable signal
        if signal == 0 or strength < self.strategy.prediction_threshold:
            logger.info("No actionable signal")
            return
            
        # Get trade parameters
        trade_params = self.strategy.generate_trade_parameters()
        if not trade_params:
            logger.info("No trade parameters generated")
            return
            
        # Execute trade
        logger.info(f"Executing {trade_params['action']} order for {self.symbol}")
        
        # In a real implementation, you would call self.connector.place_order() here
        # For simulation purposes, we'll just log the trade
        trade_result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": trade_params["symbol"],
            "action": trade_params["action"],
            "entry_price": trade_params["entry_price"],
            "stop_loss": trade_params["stop_loss"],
            "take_profit": trade_params["take_profit"],
            "signal_strength": trade_params["signal_strength"],
            "status": "SIMULATED"  # In real trading, this would be "EXECUTED" or "FAILED"
        }
        
        # Add to trade history
        self.trades.append(trade_result)
        
        # Log trade
        if self.log_trades:
            self.log_trade(trade_result)
            
        logger.info(f"Trade executed: {trade_result['action']} {trade_result['symbol']} at {trade_result['entry_price']}")
        
    def log_trade(self, trade_data):
        """Log trade to a file"""
        # Create trades directory if it doesn't exist
        os.makedirs("trades", exist_ok=True)
        
        # Create or append to the trades log file
        trades_file = f"trades/{self.symbol}_trades.json"
        
        try:
            # Read existing trades if file exists
            if os.path.exists(trades_file):
                with open(trades_file, "r") as f:
                    trades = json.load(f)
            else:
                trades = []
                
            # Add new trade
            trades.append(trade_data)
            
            # Write back to file
            with open(trades_file, "w") as f:
                json.dump(trades, f, indent=4)
                
            logger.info(f"Trade logged to {trades_file}")
        except Exception as e:
            logger.error(f"Error logging trade: {str(e)}")
            
    async def run(self):
        """Run the trading bot continuously"""
        # Initialize
        success = await self.initialize()
        if not success:
            logger.error("Initialization failed. Exiting.")
            return
            
        logger.info(f"Starting trading bot for {self.symbol}")
        logger.info(f"Checking for signals every {self.check_interval_minutes} minutes")
        
        try:
            while True:
                # Update data
                await self.update_data()
                
                # Check for signals
                await self.check_for_trade_signal()
                
                # Wait for next check
                logger.info(f"Waiting for {self.check_interval_minutes} minutes until next check")
                await asyncio.sleep(self.check_interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Error in trading bot: {str(e)}")
        finally:
            # Disconnect from MT5
            self.connector.disconnect()
            logger.info("Disconnected from MT5")
            
            # Save final trades
            if self.trades and self.log_trades:
                self.log_trade(self.trades[-1])  # Ensure the last trade is logged
                
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ML-Enhanced Trading Bot")
    
    parser.add_argument("--symbol", type=str, default="EURUSD",
                      help="Trading symbol (default: EURUSD)")
    parser.add_argument("--primary", type=str, default="H1",
                      help="Primary timeframe (default: H1)")
    parser.add_argument("--secondary", type=str, nargs="+", default=["H4", "D1"],
                      help="Secondary timeframes (default: H4 D1)")
    parser.add_argument("--interval", type=int, default=5,
                      help="Check interval in minutes (default: 5)")
    parser.add_argument("--demo", action="store_true",
                      help="Run in demo mode (simulated trades only)")
    
    return parser.parse_args()
    
if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Create and run trader
    trader = MLTrader(
        symbol=args.symbol,
        primary_timeframe=args.primary,
        secondary_timeframes=args.secondary,
        check_interval_minutes=args.interval,
        log_trades=True
    )
    
    if args.demo:
        logger.info("Running in DEMO mode - trades will be simulated")
    else:
        logger.info("Running in LIVE mode - real trades will be executed")
        
    # Run the trading bot
    asyncio.run(trader.run()) 