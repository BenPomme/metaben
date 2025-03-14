import os
import sys
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def load_latest_results():
    """Load the most recent iterative backtest results"""
    result_files = glob.glob('results/iterative_backtest_*.json')
    if not result_files:
        print("No backtest results found!")
        return None
        
    # Sort by creation time (newest first)
    latest_file = max(result_files, key=os.path.getctime)
    print(f"Loading latest results: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
        
    return results, latest_file

def display_performance_summary(results):
    """Display a performance summary table"""
    if not results:
        return
        
    # Extract baseline performance
    baseline = next((iter for iter in results if iter["name"] == "Baseline"), results[0])
    baseline_return = baseline["results"]["total_return_pct"]
    
    # Print header
    print("\n" + "="*80)
    print(f"{'Iteration':<20} {'Return %':<15} {'Win Rate':<12} {'Profit Factor':<15} {'Drawdown':<12} {'Trades':<10}")
    print("-"*80)
    
    # Print rows
    for iteration in results:
        name = iteration["name"]
        return_pct = iteration["results"]["total_return_pct"]
        win_rate = iteration["results"]["win_rate"] * 100
        profit_factor = min(iteration["results"]["profit_factor"], 999.99)  # Cap for display
        drawdown = abs(iteration["results"]["max_drawdown_pct"])
        trades = iteration["results"]["trade_count"]
        
        # Calculate improvement from baseline
        if name != "Baseline":
            improvement = return_pct - baseline_return
            improvement_str = f" ({improvement:+.2f})"
        else:
            improvement_str = ""
            
        print(f"{name:<20} {return_pct:.2f}%{improvement_str:<7} {win_rate:.2f}%{'':<5} {profit_factor:.2f}{'':<8} {drawdown:.2f}%{'':<5} {trades}")
    
    print("="*80)

def display_config_changes(results):
    """Display configuration changes between iterations"""
    if not results or len(results) < 2:
        return
        
    print("\n" + "="*80)
    print("CONFIGURATION CHANGES ACROSS ITERATIONS")
    print("-"*80)
    
    # Start with baseline config
    prev_config = results[0]["config"]
    
    for iteration in results[1:]:
        current_config = iteration["config"]
        
        # Find differences
        print(f"\n{results[0]['name']} → {iteration['name']}:")
        changes = False
        
        for key, value in current_config.items():
            if key in prev_config and prev_config[key] != value:
                print(f"  {key}: {prev_config[key]} → {value}")
                changes = True
                
        if not changes:
            print("  No changes")
            
        # Update prev_config for next iteration
        prev_config = current_config
        
    print("="*80)

def display_trade_statistics(results):
    """Display detailed trade statistics"""
    if not results:
        return
        
    print("\n" + "="*80)
    print("DETAILED TRADE STATISTICS")
    print("-"*80)
    
    for iteration in results:
        name = iteration["name"]
        trades = iteration["results"]["trades"]
        
        if not trades:
            print(f"\n{name}: No trades executed")
            continue
            
        # Calculate trade stats
        wins = sum(1 for t in trades if t["trade_result"] > 0)
        losses = sum(1 for t in trades if t["trade_result"] < 0)
        win_rate = (wins / len(trades)) * 100 if trades else 0
        
        # Average profit/loss
        avg_win = sum(t["trade_result"] for t in trades if t["trade_result"] > 0) / wins if wins else 0
        avg_loss = sum(t["trade_result"] for t in trades if t["trade_result"] < 0) / losses if losses else 0
        
        # Average holding periods
        avg_hold_win = sum(t["trade_bars"] for t in trades if t["trade_result"] > 0) / wins if wins else 0
        avg_hold_loss = sum(t["trade_bars"] for t in trades if t["trade_result"] < 0) / losses if losses else 0
        
        # Buy/Sell distribution
        buys = sum(1 for t in trades if t["action"] == "BUY")
        sells = sum(1 for t in trades if t["action"] == "SELL")
        
        print(f"\n{name}:")
        print(f"  Total Trades: {len(trades)}")
        print(f"  Wins/Losses: {wins}/{losses} ({win_rate:.2f}% win rate)")
        print(f"  Avg Win: ${avg_win:.2f}, Avg Loss: ${avg_loss:.2f}")
        print(f"  Avg Hold Period (Win): {avg_hold_win:.1f} bars, (Loss): {avg_hold_loss:.1f} bars")
        print(f"  Buy/Sell: {buys}/{sells} ({(buys/len(trades)*100):.1f}% buys)")
        
    print("="*80)

def display_equity_curve_comparison(results, results_file):
    """Display equity curve comparison"""
    if not results:
        return
    
    # Extract base filename without extension
    base_filename = os.path.splitext(os.path.basename(results_file))[0]
    
    # Check if the equity curve image exists
    equity_image = f"results/equity_curves_{base_filename.split('_', 1)[1]}.png"
    print(f"\nEquity curve visualization saved to: {equity_image}")
    
    # Check if the performance metrics image exists
    metrics_image = f"results/performance_metrics_{base_filename.split('_', 1)[1]}.png"
    print(f"Performance metrics visualization saved to: {metrics_image}")

def display_optimization_recommendations(results):
    """Display recommendations for further optimization"""
    if not results or len(results) < 2:
        return
        
    # Find the best performing iteration
    best_iteration = max(results, key=lambda x: x["results"]["total_return_pct"])
    best_config = best_iteration["config"]
    
    print("\n" + "="*80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("-"*80)
    
    print(f"\nBest performing configuration: {best_iteration['name']}")
    print(f"Return: {best_iteration['results']['total_return_pct']:.2f}%")
    
    # Trend analysis in parameters
    param_trends = {}
    
    # Only analyze if we have at least 3 iterations
    if len(results) >= 3:
        # Check which parameters changed over iterations
        baseline_config = results[0]["config"]
        for i in range(1, len(results)):
            current_config = results[i]["config"]
            current_return = results[i]["results"]["total_return_pct"]
            prev_return = results[i-1]["results"]["total_return_pct"]
            improvement = current_return > prev_return
            
            for key, value in current_config.items():
                if key in baseline_config and baseline_config[key] != value:
                    if key not in param_trends:
                        param_trends[key] = []
                    param_trends[key].append({
                        "old_value": results[i-1]["config"].get(key),
                        "new_value": value,
                        "improved": improvement
                    })
    
    # Generate recommendations
    print("\nRecommendations for further optimization:")
    
    # Analyze parameter trends
    for param, changes in param_trends.items():
        if len(changes) >= 2:
            # Check if consistent direction improved performance
            increases = [c for c in changes if c["new_value"] > c["old_value"] and c["improved"]]
            decreases = [c for c in changes if c["new_value"] < c["old_value"] and c["improved"]]
            
            if len(increases) > len(decreases) and increases:
                direction = "increase"
                current_value = best_config.get(param)
                # Suggest 10% increase
                new_value = current_value * 1.1
                print(f"  Try increasing {param} from {current_value} to {new_value:.2f}")
            elif len(decreases) > len(increases) and decreases:
                direction = "decrease"
                current_value = best_config.get(param)
                # Suggest 10% decrease
                new_value = current_value * 0.9
                print(f"  Try decreasing {param} from {current_value} to {new_value:.2f}")
    
    # General recommendations
    print("\nGeneral optimization suggestions:")
    print("  1. Test different risk:reward ratios (currently using 1:1.5)")
    print("  2. Explore adaptive position sizing based on volatility")
    print("  3. Add additional technical indicators as filters")
    print("  4. Fine-tune machine learning parameters")
    print("  5. Test longer backtest periods to ensure robustness")
    
    print("="*80)

def main():
    """Main function"""
    print("\nML Strategy Iterative Backtest Results Viewer")
    print("="*50)
    
    # Load the latest results
    results, results_file = load_latest_results()
    if not results:
        return
        
    # Display performance summary
    display_performance_summary(results)
    
    # Display configuration changes
    display_config_changes(results)
    
    # Display trade statistics
    display_trade_statistics(results)
    
    # Display equity curve comparison
    display_equity_curve_comparison(results, results_file)
    
    # Display optimization recommendations
    display_optimization_recommendations(results)
    
if __name__ == "__main__":
    main() 