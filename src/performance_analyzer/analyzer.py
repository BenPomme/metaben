"""
Performance Analyzer Module

Analyzes the performance of backtested trading strategies and calculates key metrics.
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from config.settings import PERFORMANCE_WEIGHTS


class PerformanceAnalyzer:
    """Analyzes trading strategy backtest results."""
    
    def __init__(self):
        """Initialize the Performance Analyzer."""
        self.performance_weights = PERFORMANCE_WEIGHTS
    
    def load_results(self, results_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Load backtest results from a file.
        
        Args:
            results_file: Path to the results file
            
        Returns:
            Dictionary containing the results
        """
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            logger.info(f"Loaded results from {results_file}")
            return results
        except Exception as e:
            logger.error(f"Error loading results from {results_file}: {e}")
            return {}
    
    def calculate_additional_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate additional performance metrics not provided by TradingView.
        
        Args:
            results: Dictionary containing backtest results
            
        Returns:
            Dictionary with added metrics
        """
        metrics = results.get("metrics", {})
        trades = results.get("trades", [])
        
        # Create a copy to avoid modifying the original
        enhanced_results = {**results}
        enhanced_metrics = {**metrics}
        
        # Check if we have enough data
        if not metrics or not trades:
            logger.warning("Insufficient data to calculate additional metrics")
            return enhanced_results
        
        try:
            # Convert trades to DataFrame if available
            if trades:
                trades_df = pd.DataFrame(trades)
                
                # Convert profit to numeric if it's not already
                if 'profit' in trades_df.columns:
                    trades_df['profit'] = trades_df['profit'].apply(
                        lambda x: float(str(x).replace('$', '').replace('%', '').replace(',', ''))
                        if isinstance(x, str) else float(x)
                    )
                
                # Get profit as a numpy array
                if 'profit' in trades_df.columns:
                    profits = trades_df['profit'].values
                    
                    # Calculate additional metrics
                    if len(profits) > 1:
                        # Profit consistency (standard deviation of profits)
                        enhanced_metrics['profit_std_dev'] = float(np.std(profits))
                        
                        # Profit factor (if not already provided)
                        if 'profit_factor' not in enhanced_metrics:
                            positive_sum = np.sum(profits[profits > 0])
                            negative_sum = abs(np.sum(profits[profits < 0]))
                            enhanced_metrics['profit_factor'] = float(positive_sum / negative_sum) if negative_sum > 0 else float('inf')
                        
                        # Average win and average loss
                        win_profits = profits[profits > 0]
                        loss_profits = profits[profits < 0]
                        
                        if len(win_profits) > 0:
                            enhanced_metrics['avg_win'] = float(np.mean(win_profits))
                        if len(loss_profits) > 0:
                            enhanced_metrics['avg_loss'] = float(np.mean(loss_profits))
                        
                        # Win rate (if not already provided)
                        if 'win_rate' not in enhanced_metrics:
                            win_count = len(win_profits)
                            total_trades = len(profits)
                            enhanced_metrics['win_rate'] = float(win_count / total_trades) if total_trades > 0 else 0
                        
                        # Risk-adjusted return (Sharpe-like ratio)
                        if 'sharpe_ratio' not in enhanced_metrics and enhanced_metrics.get('profit_std_dev', 0) > 0:
                            avg_profit = np.mean(profits)
                            enhanced_metrics['sharpe_ratio'] = float(avg_profit / enhanced_metrics['profit_std_dev'])
            
            # Update the results
            enhanced_results["metrics"] = enhanced_metrics
            
            logger.info(f"Calculated additional metrics: {set(enhanced_metrics.keys()) - set(metrics.keys())}")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error calculating additional metrics: {e}")
            return enhanced_results
    
    def calculate_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate an overall performance score for the strategy.
        
        Args:
            results: Dictionary containing backtest results
            
        Returns:
            Performance score (higher is better)
        """
        metrics = results.get("metrics", {})
        
        if not metrics:
            logger.warning("No metrics available to calculate score")
            return 0.0
        
        try:
            score = 0.0
            total_weight = 0.0
            
            # Calculate weighted score based on configured weights
            for metric_name, weight in self.performance_weights.items():
                if metric_name in metrics:
                    metric_value = metrics[metric_name]
                    
                    # Handle special metrics like max_drawdown (lower is better)
                    if metric_name == 'max_drawdown':
                        # Convert to positive for scoring (smaller drawdown is better)
                        if isinstance(metric_value, str) and '%' in metric_value:
                            metric_value = float(metric_value.replace('%', '').replace('-', '')) / 100
                        score += weight * (1 - abs(float(metric_value)))
                    else:
                        # For most metrics, higher is better
                        if isinstance(metric_value, str):
                            # Convert string percentage or dollar values
                            if '%' in metric_value:
                                metric_value = float(metric_value.replace('%', '')) / 100
                            elif '$' in metric_value:
                                metric_value = float(metric_value.replace('$', '').replace(',', ''))
                        
                        score += weight * float(metric_value)
                    
                    total_weight += abs(weight)
            
            # Normalize score
            normalized_score = score / total_weight if total_weight > 0 else 0
            
            logger.info(f"Calculated performance score: {normalized_score:.4f}")
            return normalized_score
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0
    
    def compare_strategies(self, results_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compare multiple strategies and rank them based on performance.
        
        Args:
            results_list: List of strategy result dictionaries
            
        Returns:
            List of strategies with scores, sorted by score (best first)
        """
        if not results_list:
            logger.warning("No strategies to compare")
            return []
        
        try:
            # Calculate score for each strategy and add it to the results
            scored_strategies = []
            for i, result in enumerate(results_list):
                score = self.calculate_score(result)
                strategy_info = {
                    "strategy_name": result.get("strategy_name", f"Strategy_{i}"),
                    "symbol": result.get("symbol", "Unknown"),
                    "score": score,
                    "net_profit": result.get("metrics", {}).get("net_profit_percent", 0),
                    "win_rate": result.get("metrics", {}).get("win_rate", 0),
                    "max_drawdown": result.get("metrics", {}).get("max_drawdown", 0),
                    "result_details": result
                }
                scored_strategies.append(strategy_info)
            
            # Sort strategies by score (descending)
            sorted_strategies = sorted(scored_strategies, key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Compared {len(sorted_strategies)} strategies")
            return sorted_strategies
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return []
    
    def identify_improvement_areas(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify areas where the strategy could be improved.
        
        Args:
            results: Dictionary containing backtest results
            
        Returns:
            Dictionary with improvement suggestions
        """
        metrics = results.get("metrics", {})
        trades = results.get("trades", [])
        
        improvement_areas = {
            "suggestions": [],
            "strengths": [],
            "weaknesses": []
        }
        
        if not metrics:
            logger.warning("No metrics available to identify improvement areas")
            return improvement_areas
        
        try:
            # Check win rate
            win_rate = metrics.get("win_rate", 0)
            if isinstance(win_rate, str) and '%' in win_rate:
                win_rate = float(win_rate.replace('%', '')) / 100
            
            if float(win_rate) < 0.4:
                improvement_areas["weaknesses"].append("Low win rate")
                improvement_areas["suggestions"].append("Improve entry conditions to increase win rate")
            elif float(win_rate) > 0.6:
                improvement_areas["strengths"].append("High win rate")
            
            # Check profit factor
            profit_factor = metrics.get("profit_factor", 0)
            if float(profit_factor) < 1.5:
                improvement_areas["weaknesses"].append("Low profit factor")
                improvement_areas["suggestions"].append("Increase profit targets or improve exit conditions")
            elif float(profit_factor) > 2.0:
                improvement_areas["strengths"].append("High profit factor")
            
            # Check drawdown
            max_drawdown = metrics.get("max_drawdown", 0)
            if isinstance(max_drawdown, str) and '%' in max_drawdown:
                max_drawdown = float(max_drawdown.replace('%', '').replace('-', '')) / 100
            
            if float(max_drawdown) > 0.2:  # 20% drawdown
                improvement_areas["weaknesses"].append("High maximum drawdown")
                improvement_areas["suggestions"].append("Implement better risk management or stop-loss mechanisms")
            elif float(max_drawdown) < 0.1:  # 10% drawdown
                improvement_areas["strengths"].append("Low maximum drawdown")
            
            # Check number of trades
            total_trades = metrics.get("total_trades", 0)
            if int(total_trades) < 20:
                improvement_areas["weaknesses"].append("Low number of trades")
                improvement_areas["suggestions"].append("Consider more sensitive entry conditions to increase trade frequency")
            
            # Check average trade duration
            avg_trade_duration = metrics.get("avg_trade_bars", 0)
            if int(avg_trade_duration) < 3:
                improvement_areas["weaknesses"].append("Very short average trade duration")
                improvement_areas["suggestions"].append("Consider implementing trade duration minimum to avoid noise")
            elif int(avg_trade_duration) > 50:
                improvement_areas["weaknesses"].append("Very long average trade duration")
                improvement_areas["suggestions"].append("Consider adding earlier exit conditions")
            
            # Check trade consistency
            if "profit_std_dev" in metrics and metrics["profit_std_dev"] > 0:
                avg_profit = metrics.get("avg_trade", 0)
                if isinstance(avg_profit, str) and '$' in avg_profit:
                    avg_profit = float(avg_profit.replace('$', '').replace(',', ''))
                
                variation_coefficient = float(metrics["profit_std_dev"]) / abs(float(avg_profit)) if float(avg_profit) != 0 else float('inf')
                
                if variation_coefficient > 2.0:
                    improvement_areas["weaknesses"].append("High profit variability between trades")
                    improvement_areas["suggestions"].append("Implement more consistent position sizing or risk per trade")
                elif variation_coefficient < 1.0:
                    improvement_areas["strengths"].append("Consistent trade profits")
            
            logger.info(f"Identified {len(improvement_areas['suggestions'])} improvement suggestions")
            return improvement_areas
            
        except Exception as e:
            logger.error(f"Error identifying improvement areas: {e}")
            return improvement_areas
    
    def analyze_trades(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze individual trades to find patterns.
        
        Args:
            results: Dictionary containing backtest results
            
        Returns:
            Dictionary with trade analysis
        """
        trades = results.get("trades", [])
        
        trade_analysis = {
            "patterns": [],
            "trade_stats": {}
        }
        
        if not trades:
            logger.warning("No trades available to analyze")
            return trade_analysis
        
        try:
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(trades)
            
            # Convert profit to numeric if needed
            if 'profit' in trades_df.columns:
                trades_df['profit'] = trades_df['profit'].apply(
                    lambda x: float(str(x).replace('$', '').replace('%', '').replace(',', ''))
                    if isinstance(x, str) else float(x)
                )
            
            # Trade statistics
            if 'position' in trades_df.columns:
                position_counts = trades_df['position'].value_counts()
                trade_analysis["trade_stats"]["position_distribution"] = position_counts.to_dict()
            
            if 'profit' in trades_df.columns:
                # Calculate consecutive wins/losses
                trades_df['is_win'] = trades_df['profit'] > 0
                
                # Find longest winning streak
                win_streak = 0
                max_win_streak = 0
                for is_win in trades_df['is_win']:
                    if is_win:
                        win_streak += 1
                        max_win_streak = max(max_win_streak, win_streak)
                    else:
                        win_streak = 0
                
                # Find longest losing streak
                loss_streak = 0
                max_loss_streak = 0
                for is_win in trades_df['is_win']:
                    if not is_win:
                        loss_streak += 1
                        max_loss_streak = max(max_loss_streak, loss_streak)
                    else:
                        loss_streak = 0
                
                trade_analysis["trade_stats"]["max_win_streak"] = max_win_streak
                trade_analysis["trade_stats"]["max_loss_streak"] = max_loss_streak
                
                # Win/loss by position type
                if 'position' in trades_df.columns:
                    win_by_position = trades_df[trades_df['profit'] > 0]['position'].value_counts()
                    loss_by_position = trades_df[trades_df['profit'] <= 0]['position'].value_counts()
                    
                    trade_analysis["trade_stats"]["win_by_position"] = win_by_position.to_dict()
                    trade_analysis["trade_stats"]["loss_by_position"] = loss_by_position.to_dict()
            
            # Identify patterns
            if len(trades_df) > 5:
                # Look for time-based patterns
                if 'entry_time' in trades_df.columns:
                    # Try to extract hour information if available
                    try:
                        trades_df['entry_hour'] = trades_df['entry_time'].apply(
                            lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour 
                            if ':' in x else None
                        )
                        hour_performance = trades_df.groupby('entry_hour')['profit'].mean().to_dict()
                        best_hour = max(hour_performance.items(), key=lambda x: x[1])
                        worst_hour = min(hour_performance.items(), key=lambda x: x[1])
                        
                        if best_hour[1] > 0:
                            trade_analysis["patterns"].append(f"Best performance at hour {best_hour[0]}")
                        if worst_hour[1] < 0:
                            trade_analysis["patterns"].append(f"Worst performance at hour {worst_hour[0]}")
                    except Exception as e:
                        logger.debug(f"Could not analyze hourly patterns: {e}")
            
            logger.info(f"Completed trade analysis with {len(trade_analysis['patterns'])} identified patterns")
            return trade_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trades: {e}")
            return trade_analysis
    
    def generate_performance_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report for a strategy.
        
        Args:
            results: Dictionary containing backtest results
            
        Returns:
            Dictionary with complete performance analysis
        """
        if not results:
            logger.warning("No results to generate performance report")
            return {}
        
        try:
            # Calculate additional metrics
            enhanced_results = self.calculate_additional_metrics(results)
            
            # Calculate performance score
            score = self.calculate_score(enhanced_results)
            
            # Identify improvement areas
            improvements = self.identify_improvement_areas(enhanced_results)
            
            # Analyze trades
            trade_analysis = self.analyze_trades(enhanced_results)
            
            # Combine all analyses into a comprehensive report
            report = {
                "strategy_name": enhanced_results.get("strategy_name", "Unknown"),
                "symbol": enhanced_results.get("symbol", "Unknown"),
                "timestamp": enhanced_results.get("timestamp", datetime.now().isoformat()),
                "performance_score": score,
                "metrics": enhanced_results.get("metrics", {}),
                "improvement_areas": improvements,
                "trade_analysis": trade_analysis,
                "raw_results": enhanced_results
            }
            
            logger.info(f"Generated performance report for {report['strategy_name']}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {} 