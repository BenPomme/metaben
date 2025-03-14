# MetaBen Trading Strategy Optimization System

## Overview
This project implements a trading strategy optimization system that allows for parameter optimization of both machine learning-based and traditional technical indicator-based trading strategies. The system includes backtesting capabilities, optimization algorithms, and visualization tools to help traders find the most profitable strategy parameters.

## Features
- **Dual Strategy Support**: Optimize both ML-based and traditional (Medallion) trading strategies
- **Parameter Optimization**: Find optimal parameters using various algorithms (random search, Bayesian optimization)
- **Continuous Optimization**: Run optimization continuously with the ability to stop manually
- **Interactive Dashboard**: Visualize optimization progress and results in real-time
- **Backtesting Framework**: Test strategies with historical data and evaluate performance
- **MetaTrader 5 Integration**: Export optimized strategies as MT5 Expert Advisors

## Project Structure
```
metaben/
├── config/                  # Configuration files
│   └── best_parameters.json # Best parameters found during optimization
├── data/                    # Historical price data
├── logs/                    # Log files
├── optimization/            # Optimization modules
│   ├── optimization_engine.py
│   ├── strategy_optimizer_controller.py
│   └── optimization_dashboard.py
├── optimization_checkpoints/ # Checkpoint files from optimization runs
│   ├── ml/
│   └── medallion/
├── results/                 # Test results and visualizations
├── run_continuous_optimization.py # Main script for continuous optimization
├── test_optimized_strategy.py     # Script for testing optimized strategies
├── ml_strategy_backtester_extension.py # ML strategy backtester
└── Medallion_Strategy_EA.mq5      # MetaTrader 5 Expert Advisor
```

## Strategy Performance Summary (as of March 2025)

### Strategy Comparison

| Metric | ML Strategy | Medallion Strategy |
|--------|------------|-------------------|
| Win Rate | 44.59% | 62.16% |
| Annual Return | 3.81% | 17.16% |
| Max Drawdown | 16.64% | 21.15% |
| Total Return | 3.86% | 17.40% |
| Profit Factor | 1.33 | 2.40 |
| Sharpe Ratio | 0.31 | 1.28 |

The Medallion strategy has demonstrated superior performance and robustness in out-of-sample testing, making it the recommended choice for live trading.

## Getting Started

### Prerequisites
- Python 3.8+
- MetaTrader 5 (for live trading)
- Required Python packages (see requirements.txt)

### Installation
1. Clone the repository:
```
git clone https://github.com/yourusername/metaben.git
cd metaben
```

2. Install dependencies:
```
pip install -r requirements.txt
```

### Running Optimization
To run continuous optimization with dashboard:
```
python run_continuous_optimization.py --strategies ml,medallion --symbol EURUSD --primary_timeframe H1
```

### Testing Optimized Strategies
To test an optimized strategy with recent data:
```
python test_optimized_strategy.py --strategy medallion_strategy --symbol EURUSD --timeframe H1 --start_date 2024-03-01 --end_date 2025-03-06
```

### Using the MetaTrader 5 Expert Advisor
1. Copy the `Medallion_Strategy_EA.mq5` file to your MetaTrader 5 Experts folder
2. Compile the EA in MetaTrader 5
3. Attach the EA to a chart with the optimized parameters

## Optimization Parameters

### ML Strategy Parameters
```json
{
  "lookback_periods": 44,
  "prediction_horizon": 5,
  "model_type": "xgboost",
  "feature_selection": "pca",
  "stop_loss_pct": 2.95,
  "take_profit_pct": 1.46,
  "risk_per_trade_pct": 1.13,
  "confidence_threshold": 0.57
}
```

### Medallion Strategy Parameters
```json
{
  "fast_ma_periods": 26,
  "slow_ma_periods": 142,
  "rsi_periods": 16,
  "rsi_overbought": 79,
  "rsi_oversold": 31,
  "volatility_factor": 1.7099,
  "stop_loss_pct": 0.7367,
  "take_profit_pct": 2.325,
  "risk_per_trade_pct": 0.9979
}
```

## Recommendations for Live Trading
1. Start with the Medallion strategy, which has shown more robust performance
2. Use proper risk management (the optimized risk per trade is approximately 1%)
3. Monitor performance regularly and compare with backtest results
4. Re-optimize parameters periodically (every 3-6 months) to adapt to changing market conditions

## Future Improvements
- Implement walk-forward optimization for more robust parameter selection
- Add more advanced ML models and feature engineering techniques
- Develop a portfolio optimization system to trade multiple currency pairs
- Create a web-based interface for remote monitoring and control

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Thanks to all contributors who have helped develop and test this system
- Special thanks to the open-source community for providing valuable tools and libraries 