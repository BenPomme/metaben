"""
Enhanced Backtester Base Class
Uses the enhanced MT5 connector with data preprocessing
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path
from abc import ABC, abstractmethod

# Add necessary paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_scripts.mt5_connector_enhanced import MT5ConnectorEnhanced

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enhanced_backtester")

class EnhancedBacktester(ABC):
    """
    Abstract base class for enhanced backtesting with data preprocessing
    """
    
    def __init__(self, 
                symbol="EURUSD",
                primary_timeframe="H1",
                secondary_timeframes=None,
                initial_balance=10000,
                data_start=None,
                data_end=None,
                preprocess_data=True,
                enable_plots=True,
                output_dir="backtest_results"):
        """
        Initialize the enhanced backtester
        
        Args:
            symbol: Trading symbol
            primary_timeframe: Primary timeframe
            secondary_timeframes: List of secondary timeframes
            initial_balance: Initial account balance
            data_start: Start date for data (default: 1 year ago)
            data_end: End date for data (default: today)
            preprocess_data: Whether to preprocess data
            enable_plots: Whether to generate plots
            output_dir: Directory for output files
        """
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframes = secondary_timeframes or ["H4", "D1"]
        self.initial_balance = initial_balance
        
        # Set date range
        self.end_date = data_end or datetime.now()
        self.start_date = data_start or (self.end_date - timedelta(days=365))
        
        # Data preprocessing flag
        self.preprocess_data = preprocess_data
        
        # Plotting
        self.enable_plots = enable_plots
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize MT5 connector
        self.connector = MT5ConnectorEnhanced(preprocess_data=self.preprocess_data)
        
        # Backtest results
        self.trades = []
        self.equity_curve = None
        self.metrics = {}
        
        logger.info(f"Initialized Enhanced Backtester for {symbol} on {primary_timeframe}")
        logger.info(f"Date range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Data preprocessing: {'Enabled' if self.preprocess_data else 'Disabled'}")
    
    def download_data(self, force_download=False):
        """
        Download or load historical data
        
        Args:
            force_download: Whether to force download even if cached data exists
            
        Returns:
            Dict of DataFrames for each timeframe
        """
        cache_file = self.output_dir / f"{self.symbol}_data_{'processed' if self.preprocess_data else 'raw'}.pkl"
        
        # Check if cached data exists
        if os.path.exists(cache_file) and not force_download:
            logger.info(f"Loading data from cache: {cache_file}")
            try:
                data = pd.read_pickle(cache_file)
                
                # Verify we have all required timeframes
                required_timeframes = [self.primary_timeframe] + self.secondary_timeframes
                if all(tf in data for tf in required_timeframes):
                    logger.info(f"Loaded cached data with {len(data[self.primary_timeframe])} candles for {self.primary_timeframe}")
                    return data
                else:
                    logger.info("Cached data doesn't have all required timeframes, downloading...")
            except Exception as e:
                logger.error(f"Error loading cached data: {e}")
                logger.info("Will download fresh data")
        
        # Connect to MT5
        connected = self.connector.connect()
        if not connected:
            logger.error("Failed to connect to MT5")
            return None
        
        try:
            # Get data for all timeframes at once
            timeframes = [self.primary_timeframe] + self.secondary_timeframes
            
            logger.info(f"Downloading data for {self.symbol} on timeframes: {', '.join(timeframes)}")
            data = self.connector.get_multi_timeframe_data(
                symbol=self.symbol,
                timeframes=timeframes,
                start_date=self.start_date,
                end_date=self.end_date,
                preprocess=self.preprocess_data
            )
            
            if data is None:
                logger.error("Failed to get data")
                return None
            
            # Save data to cache
            logger.info(f"Saving data to cache: {cache_file}")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            pd.to_pickle(data, cache_file)
            
            for tf in timeframes:
                if tf in data:
                    logger.info(f"Downloaded {len(data[tf])} candles for {tf}")
                    
                    # Save to CSV for inspection
                    csv_file = self.output_dir / f"{self.symbol}_{tf}_data.csv"
                    data[tf].to_csv(csv_file)
                    logger.info(f"Saved {tf} data to {csv_file}")
            
        finally:
            # Disconnect from MT5
            self.connector.disconnect()
        
        return data
    
    @abstractmethod
    def prepare_strategy(self, data):
        """
        Prepare the strategy for backtesting
        
        Args:
            data: Dict of DataFrames for each timeframe
            
        Returns:
            Strategy object
        """
        pass
    
    @abstractmethod
    def run_backtest(self):
        """
        Run the backtest
        
        Returns:
            Dict of backtest results
        """
        pass
    
    def calculate_metrics(self, equity_curve=None, trades=None):
        """
        Calculate performance metrics
        
        Args:
            equity_curve: Equity curve DataFrame (None to use self.equity_curve)
            trades: List of trades (None to use self.trades)
            
        Returns:
            Dict of metrics
        """
        equity_curve = equity_curve or self.equity_curve
        trades = trades or self.trades
        
        if equity_curve is None or equity_curve.empty:
            logger.error("No equity curve data for metrics calculation")
            return {}
            
        if not trades:
            logger.warning("No trades found for metrics calculation")
        
        metrics = {}
        
        # Initial and final balance
        initial_balance = equity_curve['balance'].iloc[0]
        final_balance = equity_curve['balance'].iloc[-1]
        
        # Overall return
        total_return = final_balance - initial_balance
        percent_return = (total_return / initial_balance) * 100
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days > 0:
            annual_return = ((1 + percent_return/100) ** (365/days) - 1) * 100
        else:
            annual_return = 0
        
        # Maximum drawdown
        equity_curve['peak'] = equity_curve['balance'].cummax()
        equity_curve['drawdown'] = (equity_curve['balance'] - equity_curve['peak']) / equity_curve['peak'] * 100
        max_drawdown = abs(equity_curve['drawdown'].min())
        
        # Sharpe ratio (assuming risk-free rate of 0%)
        if len(equity_curve) > 1:
            daily_returns = equity_curve['balance'].pct_change().dropna()
            if len(daily_returns) > 0 and daily_returns.std() > 0:
                sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Win rate
        if trades:
            winning_trades = [t for t in trades if t.get('profit', 0) > 0]
            win_rate = len(winning_trades) / len(trades) * 100
            
            # Average profit/loss
            profits = [t.get('profit', 0) for t in trades]
            avg_profit = sum(profits) / len(profits) if profits else 0
            
            # Profit factor
            gross_profit = sum([p for p in profits if p > 0])
            gross_loss = abs(sum([p for p in profits if p < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss else float('inf')
            
            metrics.update({
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'profit_factor': profit_factor
            })
        
        # Compile metrics
        metrics.update({
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_return': total_return,
            'percent_return': percent_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'equity_curve': equity_curve,
        })
        
        return metrics
    
    def plot_results(self, metrics=None, save_only=False):
        """
        Plot backtest results
        
        Args:
            metrics: Dict of metrics (None to use self.metrics)
            save_only: Whether to only save the plots without displaying
            
        Returns:
            None
        """
        if not self.enable_plots:
            logger.info("Plotting is disabled")
            return
            
        metrics = metrics or self.metrics
        
        if not metrics or 'equity_curve' not in metrics:
            logger.error("No metrics to plot")
            return
            
        equity_curve = metrics['equity_curve']
        
        # Create output directory
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Equity curve plot
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve.index, equity_curve['balance'], label='Equity')
        
        # Add drawdown overlay
        if 'drawdown' in equity_curve.columns:
            twin_axis = plt.twinx()
            twin_axis.fill_between(
                equity_curve.index, 
                0, 
                equity_curve['drawdown'], 
                alpha=0.3, 
                color='red', 
                label='Drawdown'
            )
            twin_axis.set_ylabel('Drawdown (%)')
            twin_axis.invert_yaxis()  # Invert to show drawdowns going down
        
        plt.title(f'{self.symbol} {self.primary_timeframe} Backtest Results')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        equity_plot_file = plots_dir / f"{self.symbol}_{self.primary_timeframe}_equity_curve.png"
        plt.savefig(equity_plot_file)
        logger.info(f"Saved equity curve plot to {equity_plot_file}")
        
        if not save_only:
            plt.show()
        else:
            plt.close()
        
        # Trades analysis if trades exist
        if self.trades and len(self.trades) > 0:
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(self.trades)
            
            # Plot trade profits
            plt.figure(figsize=(12, 6))
            plt.bar(
                range(len(trades_df)), 
                trades_df['profit'],
                color=[('green' if p > 0 else 'red') for p in trades_df['profit']]
            )
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title(f'Trade Profits ({len(trades_df)} trades)')
            plt.xlabel('Trade #')
            plt.ylabel('Profit')
            plt.grid(True)
            
            # Save the plot
            trades_plot_file = plots_dir / f"{self.symbol}_{self.primary_timeframe}_trades.png"
            plt.savefig(trades_plot_file)
            logger.info(f"Saved trades plot to {trades_plot_file}")
            
            if not save_only:
                plt.show()
            else:
                plt.close()
    
    def save_results(self, metrics=None):
        """
        Save backtest results to files
        
        Args:
            metrics: Dict of metrics (None to use self.metrics)
            
        Returns:
            None
        """
        metrics = metrics or self.metrics
        
        if not metrics:
            logger.error("No metrics to save")
            return
        
        # Create a copy to avoid modifying the original
        results = metrics.copy()
        
        # Remove equity curve as it's a DataFrame
        if 'equity_curve' in results:
            equity_curve = results.pop('equity_curve')
            
            # Save equity curve to CSV
            equity_file = self.output_dir / f"{self.symbol}_{self.primary_timeframe}_equity.csv"
            equity_curve.to_csv(equity_file)
            logger.info(f"Saved equity curve to {equity_file}")
        
        # Save trades to CSV if available
        if self.trades:
            trades_file = self.output_dir / f"{self.symbol}_{self.primary_timeframe}_trades.csv"
            pd.DataFrame(self.trades).to_csv(trades_file, index=False)
            logger.info(f"Saved trades to {trades_file}")
        
        # Save metrics to JSON
        metrics_file = self.output_dir / f"{self.symbol}_{self.primary_timeframe}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({k: v for k, v in results.items() if isinstance(v, (int, float, str, list, dict, bool))}, 
                      f, indent=4)
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Generate a summary report
        report_file = self.output_dir / f"{self.symbol}_{self.primary_timeframe}_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"Backtest Report for {self.symbol} {self.primary_timeframe}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Date Range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n")
            f.write(f"Initial Balance: ${results.get('initial_balance', 0):.2f}\n")
            f.write(f"Final Balance: ${results.get('final_balance', 0):.2f}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"{'-'*20}\n")
            f.write(f"Total Return: ${results.get('total_return', 0):.2f} ({results.get('percent_return', 0):.2f}%)\n")
            f.write(f"Annual Return: {results.get('annual_return', 0):.2f}%\n")
            f.write(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%\n")
            f.write(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}\n\n")
            
            if 'total_trades' in results:
                f.write("Trade Statistics:\n")
                f.write(f"{'-'*20}\n")
                f.write(f"Total Trades: {results.get('total_trades', 0)}\n")
                f.write(f"Win Rate: {results.get('win_rate', 0):.2f}%\n")
                f.write(f"Average Profit: ${results.get('avg_profit', 0):.2f}\n")
                f.write(f"Profit Factor: {results.get('profit_factor', 0):.2f}\n")
                
        logger.info(f"Saved summary report to {report_file}")
        
        return metrics_file 