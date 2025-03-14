#!/usr/bin/env python
"""
Competition Dashboard

This module provides a real-time dashboard for the trading strategy competition,
visualizing the performance and evolution of both strategies.
"""
import os
import json
import time
import random
import threading
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

class CompetitionDashboard:
    """Competition Dashboard class for visualizing strategy competition"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.competition_dir = Path('competition')
        self.results_dir = self.competition_dir / 'results'
        self.checkpoints_dir = self.competition_dir / 'checkpoints'
        self.leaderboard_file = self.competition_dir / 'leaderboard.json'
        
        # Create necessary directories
        for directory in [self.competition_dir, self.results_dir]:
            directory.mkdir(exist_ok=True, parents=True)
        
        # Initialize data structures
        self.ml_metrics_history = []
        self.medallion_metrics_history = []
        self.ml_score_history = []
        self.medallion_score_history = []
        self.rounds = []
        
        # Plotting attributes
        self.fig = None
        self.axes = None
        self.animation = None
        self.updating = False
        
        # Thread for automated updates
        self.update_thread = None
    
    def initialize(self):
        """Initialize the dashboard"""
        print("Initializing competition dashboard...")
        
        # Set up the plot
        # Use a safer style that works across matplotlib versions
        plt.style.use('default')
        
        self.fig, self.axes = plt.subplots(3, 2, figsize=(15, 10))
        self.fig.suptitle('Trading Strategy Competition Dashboard', fontsize=16)
        
        # Save empty plot as initial dashboard
        self._save_dashboard()
        
        # Start update thread
        self.updating = True
        self.update_thread = threading.Thread(target=self._auto_update)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        print("Dashboard initialized. Real-time visualization available.")
    
    def update(self, round_num, ml_metrics, medallion_metrics, ml_score, medallion_score, ml_wins, medallion_wins):
        """
        Update the dashboard with new data
        
        Args:
            round_num: Current tournament round
            ml_metrics: ML strategy metrics
            medallion_metrics: Medallion strategy metrics
            ml_score: ML strategy score
            medallion_score: Medallion strategy score
            ml_wins: Total ML strategy wins
            medallion_wins: Total Medallion strategy wins
        """
        # Append data to history
        self.rounds.append(round_num)
        self.ml_metrics_history.append(ml_metrics)
        self.medallion_metrics_history.append(medallion_metrics)
        self.ml_score_history.append(ml_score)
        self.medallion_score_history.append(medallion_score)
        
        # Update plots
        self._update_plots(ml_wins, medallion_wins)
        
        # Save dashboard
        self._save_dashboard()
    
    def _update_plots(self, ml_wins, medallion_wins):
        """Update all dashboard plots"""
        # Clear all axes
        for ax_row in self.axes:
            for ax in ax_row:
                ax.clear()
        
        # If no data yet, return
        if not self.rounds:
            return
        
        # Convert data to pandas DataFrames for easier plotting
        ml_df = pd.DataFrame(self.ml_metrics_history)
        medallion_df = pd.DataFrame(self.medallion_metrics_history)
        
        # Plot 1: Strategy Performance Scores
        ax = self.axes[0, 0]
        ax.plot(self.rounds, self.ml_score_history, label='ML Strategy', color='blue', marker='o')
        ax.plot(self.rounds, self.medallion_score_history, label='Medallion Strategy', color='red', marker='s')
        ax.set_title('Strategy Performance Scores')
        ax.set_xlabel('Tournament Round')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True)
        
        # Plot 2: Win Count Pie Chart
        ax = self.axes[0, 1]
        if ml_wins + medallion_wins > 0:
            ax.pie([ml_wins, medallion_wins], 
                   labels=['ML Strategy', 'Medallion Strategy'],
                   autopct='%1.1f%%',
                   colors=['blue', 'red'],
                   startangle=90)
            ax.set_title(f'Win Distribution (Total Rounds: {ml_wins + medallion_wins})')
        else:
            ax.text(0.5, 0.5, 'No data yet', ha='center', va='center', fontsize=12)
            ax.set_title('Win Distribution')
        
        # Plot 3: Annual Return Comparison
        ax = self.axes[1, 0]
        ax.plot(self.rounds, ml_df['annual_return'], label='ML Strategy', color='blue', marker='o')
        ax.plot(self.rounds, medallion_df['annual_return'], label='Medallion Strategy', color='red', marker='s')
        ax.set_title('Annual Return (%)')
        ax.set_xlabel('Tournament Round')
        ax.set_ylabel('Annual Return')
        ax.legend()
        ax.grid(True)
        
        # Plot 4: Sharpe Ratio Comparison
        ax = self.axes[1, 1]
        ax.plot(self.rounds, ml_df['sharpe_ratio'], label='ML Strategy', color='blue', marker='o')
        ax.plot(self.rounds, medallion_df['sharpe_ratio'], label='Medallion Strategy', color='red', marker='s')
        ax.set_title('Sharpe Ratio')
        ax.set_xlabel('Tournament Round')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend()
        ax.grid(True)
        
        # Plot 5: Win Rate Comparison
        ax = self.axes[2, 0]
        ax.plot(self.rounds, ml_df['win_rate'], label='ML Strategy', color='blue', marker='o')
        ax.plot(self.rounds, medallion_df['win_rate'], label='Medallion Strategy', color='red', marker='s')
        ax.set_title('Win Rate (%)')
        ax.set_xlabel('Tournament Round')
        ax.set_ylabel('Win Rate')
        ax.legend()
        ax.grid(True)
        
        # Plot 6: Max Drawdown Comparison
        ax = self.axes[2, 1]
        ax.plot(self.rounds, ml_df['max_drawdown'], label='ML Strategy', color='blue', marker='o')
        ax.plot(self.rounds, medallion_df['max_drawdown'], label='Medallion Strategy', color='red', marker='s')
        ax.set_title('Max Drawdown (%)')
        ax.set_xlabel('Tournament Round')
        ax.set_ylabel('Max Drawdown')
        ax.legend()
        ax.grid(True)
        
        # Add timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.fig.text(0.5, 0.01, f'Last Updated: {timestamp}', ha='center')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    def _save_dashboard(self):
        """Save the current dashboard as an image"""
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(exist_ok=True)
        
        # Save figure
        dashboard_file = self.results_dir / 'competition_dashboard.png'
        plt.savefig(dashboard_file)
        
        # Print update message
        if self.rounds:
            print(f"Dashboard updated (Round {self.rounds[-1]})")
    
    def _auto_update(self):
        """Auto-update the dashboard by reading leaderboard file"""
        while self.updating:
            try:
                if self.leaderboard_file.exists():
                    with open(self.leaderboard_file, 'r') as f:
                        leaderboard = json.load(f)
                    
                    # Update dashboard based on leaderboard data
                    if leaderboard['rounds']:
                        # Get the last round data
                        latest_round = leaderboard['rounds'][-1]
                        round_num = latest_round['round']
                        
                        # Only update if this is a new round
                        if not self.rounds or round_num > self.rounds[-1]:
                            ml_metrics = latest_round['ml_metrics']
                            medallion_metrics = latest_round['medallion_metrics']
                            
                            # Calculate scores
                            ml_score = (
                                0.35 * ml_metrics['annual_return'] +
                                0.25 * ml_metrics['sharpe_ratio'] * 10 -
                                0.20 * ml_metrics['max_drawdown'] +
                                0.15 * ml_metrics['win_rate'] +
                                0.05 * ml_metrics['profit_factor'] * 10
                            )
                            
                            medallion_score = (
                                0.35 * medallion_metrics['annual_return'] +
                                0.25 * medallion_metrics['sharpe_ratio'] * 10 -
                                0.20 * medallion_metrics['max_drawdown'] +
                                0.15 * medallion_metrics['win_rate'] +
                                0.05 * medallion_metrics['profit_factor'] * 10
                            )
                            
                            # Update dashboard
                            self.update(
                                round_num,
                                ml_metrics,
                                medallion_metrics,
                                ml_score,
                                medallion_score,
                                leaderboard['ml_wins'],
                                leaderboard['medallion_wins']
                            )
            except Exception as e:
                print(f"Error updating dashboard: {e}")
            
            # Sleep for a while
            time.sleep(5)
    
    def stop(self):
        """Stop the dashboard updates"""
        self.updating = False
        if self.update_thread:
            self.update_thread.join(timeout=1)
        print("Dashboard stopped")

# Test the dashboard directly if run as script
if __name__ == '__main__':
    dashboard = CompetitionDashboard()
    dashboard.initialize()
    
    # Simulate some updates for testing
    for i in range(10):
        ml_metrics = {
            'win_rate': 50 + random.uniform(-10, 10),
            'annual_return': 15 + random.uniform(-5, 15),
            'max_drawdown': 15 + random.uniform(-5, 10),
            'total_return': 10 + random.uniform(-3, 10),
            'profit_factor': 1.5 + random.uniform(-0.3, 1.0),
            'sharpe_ratio': 1.0 + random.uniform(-0.3, 1.0),
            'total_trades': 50 + random.randint(-10, 20)
        }
        
        medallion_metrics = {
            'win_rate': 55 + random.uniform(-10, 15),
            'annual_return': 18 + random.uniform(-5, 10),
            'max_drawdown': 12 + random.uniform(-4, 8),
            'total_return': 12 + random.uniform(-5, 8),
            'profit_factor': 1.8 + random.uniform(-0.4, 0.8),
            'sharpe_ratio': 1.2 + random.uniform(-0.4, 0.8),
            'total_trades': 60 + random.randint(-15, 15)
        }
        
        ml_score = (
            0.35 * ml_metrics['annual_return'] +
            0.25 * ml_metrics['sharpe_ratio'] * 10 -
            0.20 * ml_metrics['max_drawdown'] +
            0.15 * ml_metrics['win_rate'] +
            0.05 * ml_metrics['profit_factor'] * 10
        )
        
        medallion_score = (
            0.35 * medallion_metrics['annual_return'] +
            0.25 * medallion_metrics['sharpe_ratio'] * 10 -
            0.20 * medallion_metrics['max_drawdown'] +
            0.15 * medallion_metrics['win_rate'] +
            0.05 * medallion_metrics['profit_factor'] * 10
        )
        
        # Determine winner
        ml_wins = sum(1 for j in range(i) if random.random() < 0.5)
        medallion_wins = i - ml_wins
        
        dashboard.update(i, ml_metrics, medallion_metrics, ml_score, medallion_score, ml_wins, medallion_wins)
        time.sleep(1)
    
    print("Demo complete. Dashboard will continue updating from leaderboard.json")
    
    # Keep running for a while to demonstrate auto-updates
    try:
        time.sleep(300)  # Run for 5 minutes
    except KeyboardInterrupt:
        pass
    
    dashboard.stop() 