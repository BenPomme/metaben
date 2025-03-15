import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class BacktestVisualizer:
    def __init__(self):
        plt.style.use('default')
        
    def plot_trades(self, data, trades, title="Trading Performance"):
        """Plot price action with entry/exit points"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price
        ax1.plot(data.index, data['close'], label='Price', color='blue', alpha=0.6)
        
        # Plot trades
        for trade in trades:
            if trade['type'] == 'buy':
                ax1.scatter(trade['entry_time'], trade['entry_price'], marker='^', color='green', s=100)
                ax1.scatter(trade['exit_time'], trade['exit_price'], marker='v', color='red', s=100)
            else:
                ax1.scatter(trade['entry_time'], trade['entry_price'], marker='v', color='red', s=100)
                ax1.scatter(trade['exit_time'], trade['exit_price'], marker='^', color='green', s=100)
                
        # Plot equity curve
        equity_curve = self._calculate_equity_curve(trades)
        ax2.plot(equity_curve.index, equity_curve['equity'], label='Equity', color='green')
        
        # Formatting
        ax1.set_title(title)
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Equity')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
        
    def plot_performance_metrics(self, trades, initial_balance=10000):
        """Plot various performance metrics"""
        metrics = self._calculate_metrics(trades, initial_balance)
        
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplots
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        # Daily returns distribution
        daily_returns = self._calculate_daily_returns(trades)
        sns.histplot(daily_returns, kde=True, ax=ax1)
        ax1.set_title('Daily Returns Distribution')
        ax1.set_xlabel('Return %')
        ax1.set_ylabel('Frequency')
        
        # Drawdown chart
        equity_curve = self._calculate_equity_curve(trades)
        drawdown = self._calculate_drawdown(equity_curve)
        ax2.fill_between(drawdown.index, 0, -drawdown['drawdown'], color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown %')
        
        # Monthly returns heatmap
        monthly_returns = self._calculate_monthly_returns(trades)
        sns.heatmap(monthly_returns, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax3)
        ax3.set_title('Monthly Returns %')
        
        # Add metrics text
        metrics_text = (
            f"Total Return: {metrics['total_return']:.2f}%\n"
            f"Avg Daily Return: {metrics['avg_daily_return']:.2f}%\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2f}%\n"
            f"Win Rate: {metrics['win_rate']:.2f}%"
        )
        fig.text(0.02, 0.02, metrics_text, fontsize=10, family='monospace')
        
        plt.tight_layout()
        return fig
        
    def _calculate_equity_curve(self, trades):
        """Calculate equity curve from trades"""
        if not trades:
            return pd.DataFrame({'equity': []})
            
        equity_points = []
        current_equity = 10000  # Starting equity
        
        for trade in trades:
            profit = trade['profit']
            equity_points.append({
                'date': trade['exit_time'],
                'equity': current_equity + profit
            })
            current_equity += profit
            
        equity_df = pd.DataFrame(equity_points)
        if not equity_df.empty:
            equity_df.set_index('date', inplace=True)
            
        return equity_df
        
    def _calculate_daily_returns(self, trades):
        """Calculate daily returns from trades"""
        if not trades:
            return pd.Series()
            
        daily_pnl = {}
        
        for trade in trades:
            date = trade['exit_time'].date()
            if date not in daily_pnl:
                daily_pnl[date] = 0
            daily_pnl[date] += trade['profit']
            
        daily_returns = pd.Series(daily_pnl)
        return daily_returns.pct_change()
        
    def _calculate_drawdown(self, equity_curve):
        """Calculate drawdown series"""
        if equity_curve.empty:
            return pd.DataFrame({'drawdown': []})
            
        rolling_max = equity_curve['equity'].cummax()
        drawdown = (equity_curve['equity'] - rolling_max) / rolling_max * 100
        return pd.DataFrame({'drawdown': drawdown})
        
    def _calculate_monthly_returns(self, trades):
        """Calculate monthly returns heatmap data"""
        if not trades:
            return pd.DataFrame()
            
        daily_returns = self._calculate_daily_returns(trades)
        monthly_returns = daily_returns.groupby([daily_returns.index.year, daily_returns.index.month]).sum()
        
        # Convert to matrix format
        matrix = monthly_returns.unstack()
        matrix.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        return matrix
        
    def _calculate_metrics(self, trades, initial_balance):
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_return': 0,
                'avg_daily_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0
            }
            
        daily_returns = self._calculate_daily_returns(trades)
        equity_curve = self._calculate_equity_curve(trades)
        drawdown = self._calculate_drawdown(equity_curve)
        
        total_return = ((equity_curve['equity'].iloc[-1] - initial_balance) / initial_balance) * 100
        avg_daily_return = daily_returns.mean() * 100
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
        max_drawdown = abs(drawdown['drawdown'].min())
        win_rate = (sum(t['profit'] > 0 for t in trades) / len(trades)) * 100
        
        return {
            'total_return': total_return,
            'avg_daily_return': avg_daily_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        } 