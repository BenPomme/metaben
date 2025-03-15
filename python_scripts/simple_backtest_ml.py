"""
Simplified Backtest for ML Strategy
Avoids complex dependencies and configurations, focusing on core functionality
"""
import asyncio
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import os
import logging
from pathlib import Path

from mt5_connector import MT5Connector
from simple_ml_strategy import SimpleMlStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple_backtest_ml")

class SimpleMLBacktest:
    """A streamlined backtester for ML strategy that minimizes dependencies"""
    
    def __init__(self, 
                 symbol="EURUSD",
                 primary_timeframe="H1",
                 secondary_timeframes=None,
                 initial_balance=10000,
                 start_date=None,
                 end_date=None):
        """
        Initialize the backtester
        
        Args:
            symbol: Trading symbol
            primary_timeframe: Primary timeframe
            secondary_timeframes: List of secondary timeframes
            initial_balance: Initial account balance
            start_date: Start date for data (default: 1 year ago)
            end_date: End date for data (default: today)
        """
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframes = secondary_timeframes or ["H4", "D1"]
        self.initial_balance = initial_balance
        
        # Set default dates if not provided
        if end_date is None:
            self.end_date = datetime.now()
        else:
            self.end_date = end_date
            
        if start_date is None:
            self.start_date = self.end_date - timedelta(days=365)  # 1 year
        else:
            self.start_date = start_date
            
        # Data cache folder
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize connector and data
        self.connector = MT5Connector()
        self.data = {}
        self.strategy = None
        
        # Backtesting results
        self.trades = []
        self.equity_curve = None
    
    def download_data(self):
        """Download or load historical data for all timeframes"""
        logger.info(f"Getting data for {self.symbol} from {self.start_date} to {self.end_date}")
        
        # Connect to MT5
        connected = self.connector.connect()
        if not connected:
            logger.error("Failed to connect to MT5")
            return False
        
        # Download data for all timeframes
        for tf in [self.primary_timeframe] + self.secondary_timeframes:
            logger.info(f"Getting data for {tf} timeframe")
            self.data[tf] = self.connector.get_data(
                symbol=self.symbol,
                timeframe=tf,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if self.data[tf] is None or len(self.data[tf]) == 0:
                logger.error(f"Failed to get data for {tf} timeframe")
                return False
                
            logger.info(f"Got {len(self.data[tf])} candles for {tf}")
        
        return True
    
    def initialize_strategy(self):
        """Initialize the ML strategy with downloaded data"""
        logger.info("Initializing ML strategy")
        
        # Create strategy instance
        self.strategy = SimpleMlStrategy(
            symbol=self.symbol,
            primary_timeframe=self.primary_timeframe,
            secondary_timeframes=self.secondary_timeframes,
            mt5_connector=self.connector
        )
        
        # Load data into strategy
        for tf, df in self.data.items():
            self.strategy.data[tf] = df
        
        # Train the ML model
        self.strategy.train_ml_model(self.data[self.primary_timeframe])
        
        return True
    
    def run_backtest(self):
        """Run the backtest on the downloaded data"""
        if not self.data or self.primary_timeframe not in self.data:
            logger.error("No data available for backtesting")
            return False
            
        if self.strategy is None:
            logger.error("Strategy not initialized")
            return False
            
        primary_data = self.data[self.primary_timeframe]
        start_idx = 100  # Skip initial data for indicators calculation
        
        # Initialize backtest tracking
        account_balance = self.initial_balance
        equity = [account_balance]
        dates = [primary_data.index[start_idx]]
        open_position = None
        
        # Iterate through data
        for i in range(start_idx + 1, len(primary_data)):
            current_date = primary_data.index[i]
            
            # Update strategy with current data
            for tf, df in self.data.items():
                # Find data up to current date
                tf_data = df[df.index <= current_date]
                self.strategy.data[tf] = tf_data
            
            # Generate trade signal
            trade_params = self.strategy.generate_trade_parameters()
            
            # Handle open position
            if open_position:
                current_price = primary_data.iloc[i]['close']
                entry_price = open_position['entry_price']
                
                # Check if stop loss or take profit hit
                if open_position['direction'] == 'BUY':
                    profit_pips = (current_price - entry_price) / 0.0001  # For EURUSD
                    stop_hit = current_price <= open_position['stop_loss']
                    take_profit_hit = current_price >= open_position['take_profit']
                else:
                    profit_pips = (entry_price - current_price) / 0.0001  # For EURUSD
                    stop_hit = current_price >= open_position['stop_loss']
                    take_profit_hit = current_price <= open_position['take_profit']
                
                if stop_hit or take_profit_hit:
                    # Close position
                    pip_value = open_position['lot_size'] * 10  # $10 per pip for 1.0 lot
                    profit_amount = profit_pips * pip_value
                    account_balance += profit_amount
                    
                    # Record trade
                    open_position['exit_price'] = current_price
                    open_position['exit_date'] = current_date
                    open_position['profit_pips'] = profit_pips
                    open_position['profit_amount'] = profit_amount
                    
                    self.trades.append(open_position)
                    open_position = None
            
            # Open new position if no open position and we have a signal
            if not open_position and trade_params:
                action = trade_params['action']
                entry_price = trade_params['entry_price']
                stop_loss = trade_params['stop_loss']
                take_profit = trade_params['take_profit']
                
                # Calculate position size
                if action == 'BUY':
                    stop_distance_pips = (entry_price - stop_loss) / 0.0001
                else:
                    stop_distance_pips = (stop_loss - entry_price) / 0.0001
                
                risk_amount = account_balance * (self.strategy.risk_percent / 100)
                lot_size = risk_amount / (stop_distance_pips * 10)  # $10 per pip per 1.0 lot
                lot_size = max(0.01, min(10.0, lot_size))  # Limit position size
                
                # Open position
                open_position = {
                    'symbol': self.symbol,
                    'direction': action,
                    'entry_date': current_date,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'lot_size': lot_size,
                }
            
            # Track equity
            equity.append(account_balance)
            dates.append(current_date)
        
        # Create equity curve
        self.equity_curve = pd.DataFrame({
            'balance': equity
        }, index=dates)
        
        logger.info(f"Backtest complete. Final balance: ${account_balance:.2f}")
        
        # Calculate and display metrics
        self.calculate_metrics()
        
        return True
    
    def calculate_metrics(self):
        """Calculate and display trading performance metrics"""
        if self.equity_curve is None or len(self.trades) == 0:
            logger.warning("No trades or equity data to calculate metrics")
            return
        
        # Basic metrics
        initial_balance = self.initial_balance
        final_balance = self.equity_curve['balance'].iloc[-1]
        profit = final_balance - initial_balance
        return_pct = (profit / initial_balance) * 100
        
        # Trade statistics
        num_trades = len(self.trades)
        profitable_trades = sum(1 for trade in self.trades if trade['profit_amount'] > 0)
        win_rate = (profitable_trades / num_trades) * 100 if num_trades > 0 else 0
        
        # Calculate average win/loss
        if profitable_trades > 0:
            avg_win = sum(trade['profit_amount'] for trade in self.trades if trade['profit_amount'] > 0) / profitable_trades
        else:
            avg_win = 0
            
        losing_trades = num_trades - profitable_trades
        if losing_trades > 0:
            avg_loss = sum(trade['profit_amount'] for trade in self.trades if trade['profit_amount'] <= 0) / losing_trades
        else:
            avg_loss = 0
        
        # Calculate maximum drawdown
        peak = self.equity_curve['balance'].expanding().max()
        drawdown = ((self.equity_curve['balance'] - peak) / peak) * 100
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (assuming annual)
        daily_returns = self.equity_curve['balance'].pct_change().dropna()
        if len(daily_returns) > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Print results
        logger.info("=== BACKTEST RESULTS ===")
        logger.info(f"Initial Balance: ${initial_balance:.2f}")
        logger.info(f"Final Balance: ${final_balance:.2f}")
        logger.info(f"Net Profit: ${profit:.2f} ({return_pct:.2f}%)")
        logger.info(f"Number of Trades: {num_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Average Win: ${avg_win:.2f}")
        logger.info(f"Average Loss: ${avg_loss:.2f}")
        logger.info(f"Maximum Drawdown: {abs(max_drawdown):.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        
        return {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'profit': profit,
            'return_pct': return_pct,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe
        }
    
    def plot_results(self, show_fig=True, save_path=None):
        """Plot equity curve and trade results"""
        if self.equity_curve is None:
            logger.warning("No equity data to plot")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(self.equity_curve.index, self.equity_curve['balance'])
        plt.title('Equity Curve')
        plt.ylabel('Account Balance ($)')
        plt.grid(True)
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        peak = self.equity_curve['balance'].expanding().max()
        drawdown = ((self.equity_curve['balance'] - peak) / peak) * 100
        plt.fill_between(self.equity_curve.index, drawdown, 0, color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.ylabel('Drawdown (%)')
        plt.xlabel('Date')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
            
        if show_fig:
            plt.show()


def run_backtest(args):
    """Run backtest with command line arguments"""
    # Parse dates
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    else:
        start_date = None
        
    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    else:
        end_date = None
    
    # Create secondary timeframes list
    secondary_timeframes = args.secondary.split() if args.secondary else None
    
    # Initialize and run backtest
    backtester = SimpleMLBacktest(
        symbol=args.symbol,
        primary_timeframe=args.timeframe,
        secondary_timeframes=secondary_timeframes,
        initial_balance=args.balance,
        start_date=start_date,
        end_date=end_date
    )
    
    # Download data
    if not backtester.download_data():
        logger.error("Failed to download data. Exiting.")
        return
    
    # Initialize strategy
    if not backtester.initialize_strategy():
        logger.error("Failed to initialize strategy. Exiting.")
        return
    
    # Run backtest
    if not backtester.run_backtest():
        logger.error("Failed to run backtest. Exiting.")
        return
    
    # Plot results
    backtester.plot_results(save_path=f"results/{args.symbol}_{args.timeframe}_backtest.png")
    
    logger.info("Backtest complete!")


if __name__ == "__main__":
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run ML strategy backtest")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default="H1", help="Primary timeframe")
    parser.add_argument("--secondary", type=str, help="Secondary timeframes (space separated)")
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    run_backtest(args) 