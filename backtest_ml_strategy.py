"""
ML Strategy Backtester
Downloads data once and uses it for both backtesting and optimization
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import pickle
import logging
from pathlib import Path

from mt5_connector import MT5Connector
from simple_ml_strategy import SimpleMLStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backtest_ml_strategy")

class MLStrategyBacktester:
    """Backtester for the ML enhanced trading strategy"""
    
    def __init__(self, 
                 symbol="EURUSD",
                 primary_timeframe="H1",
                 secondary_timeframes=None,
                 initial_balance=10000,
                 data_start=None,
                 data_end=None):
        """
        Initialize the backtester
        
        Args:
            symbol: Trading symbol
            primary_timeframe: Primary timeframe
            secondary_timeframes: List of secondary timeframes
            initial_balance: Initial account balance
            data_start: Start date for data (default: 1 year ago)
            data_end: End date for data (default: today)
        """
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframes = secondary_timeframes or ["H4", "D1"]
        self.initial_balance = initial_balance
        
        # Set date range
        self.end_date = data_end or datetime.now()
        self.start_date = data_start or (self.end_date - timedelta(days=365))
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Initialize MT5 connector
        self.connector = MT5Connector()
        
        # Initialize strategy
        self.strategy = None
        
        # Backtest results
        self.trades = []
        self.equity_curve = None
        self.metrics = {}
        
        logger.info(f"Initialized ML Strategy Backtester for {symbol} on {primary_timeframe}")
        logger.info(f"Date range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
    
    def download_data(self, force_download=False):
        """
        Download or load historical data
        
        Args:
            force_download: Whether to force download even if cached data exists
            
        Returns:
            Dict of DataFrames for each timeframe
        """
        data = {}
        data_file = f'data/{self.symbol}_data.pkl'
        
        # Check if data file exists
        if os.path.exists(data_file) and not force_download:
            logger.info(f"Loading data from cache: {data_file}")
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                
            # Verify we have all required timeframes
            required_timeframes = [self.primary_timeframe] + self.secondary_timeframes
            if all(tf in data for tf in required_timeframes):
                logger.info(f"Loaded cached data with {len(data[self.primary_timeframe])} candles for {self.primary_timeframe}")
                return data
            else:
                logger.info("Cached data doesn't have all required timeframes, downloading...")
        
        # Connect to MT5
        connected = self.connector.connect()
        if not connected:
            logger.error("Failed to connect to MT5")
            return None
        
        try:
            # Download data for each timeframe
            for timeframe in [self.primary_timeframe] + self.secondary_timeframes:
                logger.info(f"Downloading {timeframe} data for {self.symbol}...")
                df = self.connector.get_data(
                    symbol=self.symbol,
                    timeframe=timeframe,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                
                if df is None or df.empty:
                    logger.error(f"Failed to get {timeframe} data")
                    return None
                
                data[timeframe] = df
                logger.info(f"Downloaded {len(df)} candles for {timeframe}")
            
            # Save data to file
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
                
            logger.info(f"Saved data to {data_file}")
            
        finally:
            # Disconnect from MT5
            self.connector.disconnect()
        
        return data
    
    def prepare_strategy(self, data, params=None):
        """
        Prepare the ML strategy with data and optional parameters
        
        Args:
            data: Dict of DataFrames for each timeframe
            params: Optional parameters to override default settings
            
        Returns:
            Initialized strategy instance
        """
        # Initialize strategy
        self.strategy = SimpleMLStrategy(
            symbol=self.symbol,
            primary_timeframe=self.primary_timeframe,
            secondary_timeframes=self.secondary_timeframes,
            mt5_connector=None  # No need for connector in backtest
        )
        
        # Set data
        self.strategy.data = data
        
        # Apply custom parameters if provided
        if params:
            for key, value in params.items():
                if hasattr(self.strategy, key):
                    setattr(self.strategy, key, value)
                    logger.info(f"Set strategy parameter {key} = {value}")
        
        # Train ML model
        success = self.strategy.train_ml_model(data[self.primary_timeframe])
        if not success:
            logger.error("Failed to train ML model")
            return None
            
        logger.info("Strategy prepared successfully")
        return self.strategy
    
    def run_backtest(self, start_idx=None, end_idx=None):
        """
        Run backtest on the strategy
        
        Args:
            start_idx: Starting index for backtest (default: 100 to allow for indicators)
            end_idx: Ending index for backtest (default: end of data)
            
        Returns:
            Dict of backtest results
        """
        if not self.strategy or not self.strategy.data:
            logger.error("Strategy not initialized or no data available")
            return None
        
        # Get primary data
        data = self.strategy.data[self.primary_timeframe]
        
        # Set backtest range
        start_idx = start_idx or 100  # Allow room for indicators
        end_idx = end_idx or len(data) - 1
        
        logger.info(f"Running backtest from index {start_idx} to {end_idx} ({end_idx - start_idx + 1} candles)")
        
        # Initialize backtest variables
        balance = self.initial_balance
        equity = balance
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        
        # Reset trades
        self.trades = []
        
        # Create equity curve DataFrame
        dates = data.index[start_idx:end_idx+1]
        equity_data = {
            'balance': [balance] * len(dates),
            'equity': [equity] * len(dates),
            'position': [0] * len(dates),
            'drawdown': [0.0] * len(dates)
        }
        self.equity_curve = pd.DataFrame(equity_data, index=dates)
        
        # Track max equity for drawdown calculation
        max_equity = equity
        
        # Loop through historical data
        for i in range(start_idx, end_idx + 1):
            current_time = data.index[i]
            
            # Update equity curve
            self.equity_curve.loc[current_time, 'balance'] = balance
            self.equity_curve.loc[current_time, 'equity'] = equity
            self.equity_curve.loc[current_time, 'position'] = position
            
            # Calculate drawdown
            max_equity = max(max_equity, equity)
            drawdown_pct = (max_equity - equity) / max_equity * 100 if max_equity > 0 else 0
            self.equity_curve.loc[current_time, 'drawdown'] = drawdown_pct
            
            # If we have an open position, check for exit
            if position != 0:
                # Get current price
                current_price = data.iloc[i]['close']
                
                # Check for stop loss or take profit
                if position > 0:  # Long position
                    if current_price <= stop_loss:  # Stop loss hit
                        # Calculate profit/loss
                        profit_loss = (current_price - entry_price) * position
                        
                        # Update balance
                        balance += profit_loss
                        equity = balance
                        
                        # Record trade exit
                        self.trades[-1].update({
                            'exit_date': current_time,
                            'exit_price': current_price,
                            'exit_reason': 'stop_loss',
                            'profit_loss': profit_loss,
                            'profit_loss_pct': profit_loss / (entry_price * position) * 100
                        })
                        
                        logger.info(f"Stop Loss: Closed LONG position at {current_price:.5f}, P/L: {profit_loss:.2f} ({self.trades[-1]['profit_loss_pct']:.2f}%)")
                        
                        # Reset position
                        position = 0
                        
                    elif current_price >= take_profit:  # Take profit hit
                        # Calculate profit/loss
                        profit_loss = (current_price - entry_price) * position
                        
                        # Update balance
                        balance += profit_loss
                        equity = balance
                        
                        # Record trade exit
                        self.trades[-1].update({
                            'exit_date': current_time,
                            'exit_price': current_price,
                            'exit_reason': 'take_profit',
                            'profit_loss': profit_loss,
                            'profit_loss_pct': profit_loss / (entry_price * position) * 100
                        })
                        
                        logger.info(f"Take Profit: Closed LONG position at {current_price:.5f}, P/L: {profit_loss:.2f} ({self.trades[-1]['profit_loss_pct']:.2f}%)")
                        
                        # Reset position
                        position = 0
                        
                elif position < 0:  # Short position
                    if current_price >= stop_loss:  # Stop loss hit
                        # Calculate profit/loss
                        profit_loss = (entry_price - current_price) * abs(position)
                        
                        # Update balance
                        balance += profit_loss
                        equity = balance
                        
                        # Record trade exit
                        self.trades[-1].update({
                            'exit_date': current_time,
                            'exit_price': current_price,
                            'exit_reason': 'stop_loss',
                            'profit_loss': profit_loss,
                            'profit_loss_pct': profit_loss / (entry_price * abs(position)) * 100
                        })
                        
                        logger.info(f"Stop Loss: Closed SHORT position at {current_price:.5f}, P/L: {profit_loss:.2f} ({self.trades[-1]['profit_loss_pct']:.2f}%)")
                        
                        # Reset position
                        position = 0
                        
                    elif current_price <= take_profit:  # Take profit hit
                        # Calculate profit/loss
                        profit_loss = (entry_price - current_price) * abs(position)
                        
                        # Update balance
                        balance += profit_loss
                        equity = balance
                        
                        # Record trade exit
                        self.trades[-1].update({
                            'exit_date': current_time,
                            'exit_price': current_price,
                            'exit_reason': 'take_profit',
                            'profit_loss': profit_loss,
                            'profit_loss_pct': profit_loss / (entry_price * abs(position)) * 100
                        })
                        
                        logger.info(f"Take Profit: Closed SHORT position at {current_price:.5f}, P/L: {profit_loss:.2f} ({self.trades[-1]['profit_loss_pct']:.2f}%)")
                        
                        # Reset position
                        position = 0
            
            # Only check for new signals if we don't have an open position
            if position == 0:
                # Create a copy of the strategy data up to the current index
                temp_data = {}
                for tf, df in self.strategy.data.items():
                    # Filter data up to current time
                    if tf == self.primary_timeframe:
                        temp_data[tf] = df.iloc[:i+1].copy()
                    else:
                        # For secondary timeframes, filter data up to the current time
                        temp_data[tf] = df[df.index <= current_time].copy()
                
                # Update strategy with the data up to current point
                self.strategy.data = temp_data
                
                # Generate trade parameters
                trade_params = self.strategy.generate_trade_parameters()
                
                # Check if we have a valid trade signal
                if trade_params:
                    # Get trade details
                    signal = 1 if trade_params['action'] == 'BUY' else -1
                    current_price = trade_params['entry_price']
                    
                    # Calculate position size based on risk
                    risk_amount = balance * self.strategy.risk_percent / 100
                    
                    if signal > 0:  # Long position
                        stop_loss = trade_params['stop_loss']
                        risk_per_unit = current_price - stop_loss
                    else:  # Short position
                        stop_loss = trade_params['stop_loss'] 
                        risk_per_unit = stop_loss - current_price
                    
                    # Ensure risk_per_unit is positive
                    risk_per_unit = abs(risk_per_unit)
                    
                    # Calculate position size
                    if risk_per_unit > 0:
                        position_size = risk_amount / risk_per_unit
                    else:
                        position_size = 0
                        
                    # Limit position size to a percentage of balance
                    max_position_size = balance * 0.1 / current_price  # Max 10% of balance
                    position_size = min(position_size, max_position_size)
                    
                    if position_size > 0:
                        # Open position
                        position = position_size * signal
                        entry_price = current_price
                        stop_loss = trade_params['stop_loss']
                        take_profit = trade_params['take_profit']
                        
                        # Record trade entry
                        trade = {
                            'entry_date': current_time,
                            'entry_price': entry_price,
                            'position': position,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'signal_strength': trade_params['signal_strength'],
                            'direction': trade_params['action']
                        }
                        self.trades.append(trade)
                        
                        logger.info(f"Opened {trade['direction']} position at {entry_price:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}, Size: {abs(position):.2f}")
        
        # Close any open position at the end of the backtest
        if position != 0:
            # Get final price
            final_price = data.iloc[end_idx]['close']
            
            # Calculate profit/loss
            if position > 0:  # Long position
                profit_loss = (final_price - entry_price) * position
            else:  # Short position
                profit_loss = (entry_price - final_price) * abs(position)
            
            # Update balance
            balance += profit_loss
            equity = balance
            
            # Record trade exit
            self.trades[-1].update({
                'exit_date': data.index[end_idx],
                'exit_price': final_price,
                'exit_reason': 'end_of_data',
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss / (entry_price * abs(position)) * 100
            })
            
            logger.info(f"End of Data: Closed {self.trades[-1]['direction']} position at {final_price:.5f}, P/L: {profit_loss:.2f} ({self.trades[-1]['profit_loss_pct']:.2f}%)")
            
            # Reset position
            position = 0
        
        # Calculate performance metrics
        metrics = self.calculate_metrics(self.initial_balance, balance, self.trades, self.equity_curve)
        self.metrics = metrics
        
        logger.info(f"Backtest completed with {len(self.trades)} trades")
        logger.info(f"Final balance: ${balance:.2f} (Return: {metrics['return_pct']:.2f}%)")
        logger.info(f"Win rate: {metrics['win_rate']:.2f}%, Profit factor: {metrics['profit_factor']:.2f}")
        
        return metrics
    
    def calculate_metrics(self, initial_balance, final_balance, trades, equity_curve):
        """
        Calculate performance metrics
        
        Args:
            initial_balance: Initial account balance
            final_balance: Final account balance
            trades: List of trades
            equity_curve: DataFrame with equity curve
            
        Returns:
            Dict of performance metrics
        """
        metrics = {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'net_profit': final_balance - initial_balance,
            'return_pct': (final_balance / initial_balance - 1) * 100,
            'max_drawdown': equity_curve['drawdown'].max(),
            'trade_count': len(trades)
        }
        
        # Calculate win rate and profit factor
        if trades:
            winning_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
            losing_trades = [t for t in trades if t.get('profit_loss', 0) < 0]
            
            metrics['win_count'] = len(winning_trades)
            metrics['loss_count'] = len(losing_trades)
            metrics['win_rate'] = len(winning_trades) / len(trades) * 100 if trades else 0
            
            gross_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
            gross_loss = sum(abs(t.get('profit_loss', 0)) for t in losing_trades)
            
            metrics['gross_profit'] = gross_profit
            metrics['gross_loss'] = gross_loss
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss else float('inf')
            
            # Calculate average profit/loss
            metrics['avg_profit'] = gross_profit / len(winning_trades) if winning_trades else 0
            metrics['avg_loss'] = gross_loss / len(losing_trades) if losing_trades else 0
            metrics['avg_trade'] = metrics['net_profit'] / len(trades) if trades else 0
            
            # Calculate max consecutive winners/losers
            results = [1 if t.get('profit_loss', 0) > 0 else -1 for t in trades]
            
            max_wins = 0
            current_wins = 0
            max_losses = 0
            current_losses = 0
            
            for r in results:
                if r > 0:
                    current_wins += 1
                    current_losses = 0
                    max_wins = max(max_wins, current_wins)
                else:
                    current_losses += 1
                    current_wins = 0
                    max_losses = max(max_losses, current_losses)
            
            metrics['max_consecutive_wins'] = max_wins
            metrics['max_consecutive_losses'] = max_losses
            
        # Calculate Sharpe Ratio (if we have enough data)
        if len(equity_curve) > 1:
            daily_returns = equity_curve['equity'].pct_change().dropna()
            if len(daily_returns) > 0:
                annualized_sharpe = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() > 0 else 0
                metrics['sharpe_ratio'] = annualized_sharpe
        
        return metrics
    
    def plot_results(self, show_trades=True, save_path=None):
        """
        Plot backtest results
        
        Args:
            show_trades: Whether to show individual trades on the plot
            save_path: Path to save the plot image (or None to display)
        """
        if self.equity_curve is None or self.equity_curve.empty:
            logger.error("No equity curve data to plot")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        self.equity_curve['equity'].plot(ax=ax1, color='blue', label='Equity')
        ax1.set_title(f'ML Strategy Backtest Results - {self.symbol} {self.primary_timeframe}')
        ax1.set_ylabel('Equity')
        ax1.grid(True)
        
        # Mark trades on the equity curve
        if show_trades and self.trades:
            for trade in self.trades:
                if 'entry_date' in trade and 'exit_date' in trade and 'profit_loss' in trade:
                    # Determine color based on profit/loss
                    color = 'green' if trade['profit_loss'] > 0 else 'red'
                    
                    # Plot entry point
                    ax1.scatter(trade['entry_date'], self.equity_curve.loc[trade['entry_date'], 'equity'], 
                               color=color, marker='^' if trade['direction'] == 'BUY' else 'v', s=50)
                    
                    # Plot exit point
                    ax1.scatter(trade['exit_date'], self.equity_curve.loc[trade['exit_date'], 'equity'], 
                               color=color, marker='o', s=50)
                    
                    # Connect entry and exit with a line
                    ax1.plot([trade['entry_date'], trade['exit_date']], 
                             [self.equity_curve.loc[trade['entry_date'], 'equity'], 
                              self.equity_curve.loc[trade['exit_date'], 'equity']], 
                             color=color, linestyle='--', alpha=0.3)
        
        # Plot drawdown
        self.equity_curve['drawdown'].plot(ax=ax2, color='red', alpha=0.5, label='Drawdown %')
        ax2.set_ylabel('Drawdown %')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        # Add metrics text
        if self.metrics:
            metrics_text = (
                f"Initial Balance: ${self.metrics['initial_balance']:.2f}\n"
                f"Final Balance: ${self.metrics['final_balance']:.2f}\n"
                f"Net Profit: ${self.metrics['net_profit']:.2f} ({self.metrics['return_pct']:.2f}%)\n"
                f"Max Drawdown: {self.metrics['max_drawdown']:.2f}%\n"
                f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}\n"
                f"Win Rate: {self.metrics['win_rate']:.2f}%\n"
                f"Profit Factor: {self.metrics['profit_factor']:.2f}\n"
                f"Total Trades: {self.metrics['trade_count']}"
            )
            
            # Add text box to the plot
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    def save_results(self, filename_prefix='ml_strategy'):
        """
        Save backtest results to files
        
        Args:
            filename_prefix: Prefix for saved files
            
        Returns:
            Dict with saved file paths
        """
        saved_files = {}
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Save trades to CSV
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = f'results/{filename_prefix}_trades.csv'
            trades_df.to_csv(trades_file)
            saved_files['trades'] = trades_file
            logger.info(f"Saved trades to {trades_file}")
        
        # Save equity curve to CSV
        if self.equity_curve is not None and not self.equity_curve.empty:
            equity_file = f'results/{filename_prefix}_equity.csv'
            self.equity_curve.to_csv(equity_file)
            saved_files['equity'] = equity_file
            logger.info(f"Saved equity curve to {equity_file}")
        
        # Save metrics to JSON
        if self.metrics:
            metrics_file = f'results/{filename_prefix}_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            saved_files['metrics'] = metrics_file
            logger.info(f"Saved metrics to {metrics_file}")
        
        # Save plot
        plot_file = f'results/{filename_prefix}_plot.png'
        self.plot_results(save_path=plot_file)
        saved_files['plot'] = plot_file
        
        return saved_files

def run_backtest(args):
    """Run backtest with command line arguments"""
    # Convert date strings to datetime objects
    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None
    
    # Create backtester
    backtester = MLStrategyBacktester(
        symbol=args.symbol,
        primary_timeframe=args.timeframe,
        secondary_timeframes=args.secondary,
        initial_balance=args.balance,
        data_start=start_date,
        data_end=end_date
    )
    
    # Download or load data
    data = backtester.download_data(force_download=args.force_download)
    if data is None:
        logger.error("Failed to get data")
        return
    
    # Load parameters if specified
    params = None
    if args.params:
        with open(args.params, 'r') as f:
            params = json.load(f)
    
    # Prepare strategy
    strategy = backtester.prepare_strategy(data, params)
    if strategy is None:
        logger.error("Failed to prepare strategy")
        return
    
    # Run backtest
    metrics = backtester.run_backtest()
    if metrics is None:
        logger.error("Backtest failed")
        return
    
    # Save results
    saved_files = backtester.save_results(args.output_prefix)
    
    # Print summary
    print("\n" + "="*50)
    print(f"Backtest Results - {args.symbol} {args.timeframe}")
    print("="*50)
    print(f"Initial Balance: ${metrics['initial_balance']:.2f}")
    print(f"Final Balance: ${metrics['final_balance']:.2f}")
    print(f"Net Profit: ${metrics['net_profit']:.2f} ({metrics['return_pct']:.2f}%)")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Total Trades: {metrics['trade_count']}")
    print("="*50)
    print("\nResults saved to:")
    for k, v in saved_files.items():
        print(f"- {v}")
    print("\n")
    
    if metrics['return_pct'] > 0 and metrics['win_rate'] > 50 and metrics['profit_factor'] > 1.5:
        print("✅ This strategy is profitable!")
    else:
        print("❌ This strategy needs optimization.")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ML Strategy Backtester")
    
    parser.add_argument("--symbol", type=str, default="EURUSD",
                        help="Trading symbol (default: EURUSD)")
    parser.add_argument("--timeframe", type=str, default="H1",
                        help="Primary timeframe (default: H1)")
    parser.add_argument("--secondary", type=str, nargs="+", default=["H4", "D1"],
                        help="Secondary timeframes (default: H4 D1)")
    parser.add_argument("--balance", type=float, default=10000,
                        help="Initial balance (default: 10000)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD) (default: 1 year ago)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYY-MM-DD) (default: today)")
    parser.add_argument("--params", type=str, default=None,
                        help="JSON file with strategy parameters")
    parser.add_argument("--force-download", action="store_true",
                        help="Force download data even if cached data exists")
    parser.add_argument("--output-prefix", type=str, default="ml_strategy",
                        help="Prefix for output files (default: ml_strategy)")
    
    args = parser.parse_args()
    run_backtest(args) 