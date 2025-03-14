#!/usr/bin/env python
"""
Strategy Competition

This script orchestrates a competition between the ML and Medallion trading strategies,
continuously optimizing and testing them against each other to determine the best performer.
The optimization runs until interrupted, with results updated in real-time.
"""
import os
import time
import json
import random
import argparse
import threading
import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Set up directory structure
BASE_DIR = Path('.')
COMPETITION_DIR = BASE_DIR / 'competition'
RESULTS_DIR = COMPETITION_DIR / 'results'
CHECKPOINTS_DIR = COMPETITION_DIR / 'checkpoints'
LEADERBOARD_FILE = COMPETITION_DIR / 'leaderboard.json'

# Create necessary directories
for directory in [COMPETITION_DIR, RESULTS_DIR, CHECKPOINTS_DIR]:
    directory.mkdir(exist_ok=True)

# Initialize global state
running = True
tournament_round = 0
ml_wins = 0
medallion_wins = 0
current_ml_best = None
current_medallion_best = None

def load_ml_optimizer():
    """Load the ML strategy optimizer module"""
    from ml_strategy_optimizer import MLStrategyOptimizer
    return MLStrategyOptimizer()

def load_medallion_optimizer():
    """Load the Medallion strategy optimizer module"""
    from medallion_strategy_optimizer import MedallionStrategyOptimizer
    return MedallionStrategyOptimizer()

def load_dashboard():
    """Load the competition dashboard module"""
    from competition_dashboard import CompetitionDashboard
    return CompetitionDashboard()

def save_checkpoint(strategy_type, params, metrics, round_num):
    """Save optimization checkpoint"""
    checkpoint = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'round': round_num,
        'parameters': params,
        'metrics': metrics
    }
    
    # Save to checkpoint file
    checkpoint_file = CHECKPOINTS_DIR / f"{strategy_type}_round_{round_num}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    # Log checkpoint
    print(f"Checkpoint saved for {strategy_type} (Round {round_num})")
    print(f"Parameters: {json.dumps(params, indent=2)}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    return checkpoint

def update_leaderboard(ml_checkpoint, medallion_checkpoint, winner, round_num):
    """Update the competition leaderboard"""
    if not LEADERBOARD_FILE.exists():
        leaderboard = {
            'last_updated': None,
            'current_round': 0,
            'ml_wins': 0,
            'medallion_wins': 0,
            'rounds': []
        }
    else:
        with open(LEADERBOARD_FILE, 'r') as f:
            leaderboard = json.load(f)
    
    # Update leaderboard
    leaderboard['last_updated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    leaderboard['current_round'] = round_num
    
    if winner == 'ml':
        leaderboard['ml_wins'] += 1
    else:
        leaderboard['medallion_wins'] += 1
    
    # Add round details
    round_data = {
        'round': round_num,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'winner': winner,
        'ml_metrics': ml_checkpoint['metrics'],
        'medallion_metrics': medallion_checkpoint['metrics'],
        'ml_parameters': ml_checkpoint['parameters'],
        'medallion_parameters': medallion_checkpoint['parameters']
    }
    
    leaderboard['rounds'].append(round_data)
    
    # Save updated leaderboard
    with open(LEADERBOARD_FILE, 'w') as f:
        json.dump(leaderboard, f, indent=2)
    
    print(f"Leaderboard updated - Round {round_num}: {winner.upper()} wins")
    print(f"Current score - ML: {leaderboard['ml_wins']}, Medallion: {leaderboard['medallion_wins']}")

def compare_strategies(ml_metrics, medallion_metrics):
    """Compare strategy performance and determine the winner"""
    # Calculate weighted score based on multiple metrics
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
    
    print(f"Strategy Comparison - Round {tournament_round}")
    print(f"ML Score: {ml_score:.2f}, Medallion Score: {medallion_score:.2f}")
    
    if ml_score > medallion_score:
        return 'ml', ml_score, medallion_score
    else:
        return 'medallion', ml_score, medallion_score

def run_tournament_round():
    """Run a single round of the tournament"""
    global tournament_round, ml_wins, medallion_wins
    global current_ml_best, current_medallion_best
    
    print(f"\n{'='*50}")
    print(f"TOURNAMENT ROUND {tournament_round}")
    print(f"{'='*50}")
    
    # Load optimizers
    ml_optimizer = load_ml_optimizer()
    medallion_optimizer = load_medallion_optimizer()
    
    # Run optimization round
    print("Optimizing ML Strategy...")
    ml_params, ml_metrics = ml_optimizer.optimize(current_ml_best, tournament_round)
    
    print("Optimizing Medallion Strategy...")
    medallion_params, medallion_metrics = medallion_optimizer.optimize(current_medallion_best, tournament_round)
    
    # Save checkpoints
    ml_checkpoint = save_checkpoint('ml', ml_params, ml_metrics, tournament_round)
    medallion_checkpoint = save_checkpoint('medallion', medallion_params, medallion_metrics, tournament_round)
    
    # Compare and update winners
    winner, ml_score, medallion_score = compare_strategies(ml_metrics, medallion_metrics)
    
    if winner == 'ml':
        ml_wins += 1
        current_ml_best = ml_params
    else:
        medallion_wins += 1
        current_medallion_best = medallion_params
    
    # Update leaderboard
    update_leaderboard(ml_checkpoint, medallion_checkpoint, winner, tournament_round)
    
    # Update dashboard
    try:
        dashboard = load_dashboard()
        dashboard.update(tournament_round, ml_metrics, medallion_metrics, 
                        ml_score, medallion_score, ml_wins, medallion_wins)
    except Exception as e:
        print(f"Error updating dashboard: {e}")
    
    # Increment round counter
    tournament_round += 1
    
    return winner

def monitor_keyboard_interrupt():
    """Monitor for keyboard interrupt to stop the competition"""
    global running
    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Stopping competition after current round...")
        running = False

def main():
    """Main entry point"""
    global tournament_round, running
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Trading Strategy Competition')
    parser.add_argument('--symbols', type=str, default='EURUSD,GBPUSD',
                      help='Comma-separated list of symbols to test on')
    parser.add_argument('--timeframes', type=str, default='H1,H4',
                      help='Comma-separated list of timeframes to test on')
    parser.add_argument('--start_date', type=str, default='2024-01-01',
                      help='Start date for testing data')
    parser.add_argument('--end_date', type=str, default='2025-03-06',
                      help='End date for testing data')
    parser.add_argument('--rounds', type=int, default=1000000,
                      help='Maximum number of tournament rounds')
    parser.add_argument('--run_until', type=str, default=None,
                      help='Run until specified time (format: YYYY-MM-DD HH:MM)')
    
    args = parser.parse_args()
    
    print("Starting Trading Strategy Competition")
    print(f"Symbols: {args.symbols}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Testing period: {args.start_date} to {args.end_date}")
    
    if args.run_until:
        target_time = datetime.datetime.strptime(args.run_until, '%Y-%m-%d %H:%M')
        print(f"Competition will run until {args.run_until}")
    else:
        target_time = None
        print(f"Competition will run for up to {args.rounds} rounds or until interrupted")
    
    # Start keyboard interrupt monitor in a separate thread
    interrupt_thread = threading.Thread(target=monitor_keyboard_interrupt)
    interrupt_thread.daemon = True
    interrupt_thread.start()
    
    # Initialize dashboard
    try:
        dashboard = load_dashboard()
        dashboard.initialize()
    except Exception as e:
        print(f"Error initializing dashboard: {e}")
    
    # Run tournament until stopped or max rounds reached
    while running and tournament_round < args.rounds:
        # Check if we've reached target time
        if target_time and datetime.datetime.now() >= target_time:
            print(f"Target time reached ({args.run_until}). Stopping competition.")
            break
        
        # Run a tournament round
        winner = run_tournament_round()
        
        # Artificial delay between rounds
        time.sleep(2)
    
    print("\nCompetition complete!")
    print(f"Final Score: ML: {ml_wins}, Medallion: {medallion_wins}")
    
    if ml_wins > medallion_wins:
        print("ML STRATEGY is the champion!")
    elif medallion_wins > ml_wins:
        print("MEDALLION STRATEGY is the champion!")
    else:
        print("The competition ended in a TIE!")

if __name__ == '__main__':
    main() 