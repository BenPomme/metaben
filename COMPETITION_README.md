# Trading Strategy Competition Framework

## Overview
This framework pits two trading strategies against each other in a continuous optimization tournament to find the most profitable parameters. The ML and Medallion strategies compete in a round-robin tournament, with each round involving optimization and testing against market data.

## Features
- **Real-time Competition**: Strategies are continuously optimized and compared in real-time
- **Multiple Optimization Techniques**: Random search, evolutionary algorithms, golden ratio optimization, and more
- **Adaptive Parameter Tuning**: Strategies adapt to changing market conditions
- **Real-time Dashboard**: Visualize performance metrics and competition results as they happen
- **Robust Backtesting**: Test strategies on historical data with realistic market conditions
- **Extensible Framework**: Easy to add new strategies or optimization techniques

## Components
- `strategy_competition.py`: Main competition controller
- `ml_strategy_optimizer.py`: ML strategy optimizer
- `medallion_strategy_optimizer.py`: Medallion strategy optimizer
- `strategy_tester.py`: Backtester for both strategies
- `competition_dashboard.py`: Real-time visualization tools
- `run_competition.py`: Command-line interface to run the competition

## How It Works
1. The competition runs multiple rounds of optimization for both strategies
2. In each round, different optimization techniques are applied based on the round number
3. Strategies are evaluated on historical data with realistic market conditions
4. Performance is scored based on a weighted combination of metrics (returns, drawdown, Sharpe, etc.)
5. The strategy with the higher score wins the round
6. Each win is recorded and checkpoints are saved
7. The competition continues until it reaches the maximum rounds or until a specified time

## Getting Started

### Requirements
- Python 3.8 or higher
- Libraries: numpy, pandas, matplotlib, scikit-learn, tqdm

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/trading-competition.git
cd trading-competition

# Install dependencies
pip install -r requirements.txt
```

### Running the Competition
```bash
# Run with default settings (until tomorrow)
python run_competition.py

# Run with specific settings
python run_competition.py --symbols EURUSD,USDJPY --timeframes H1,D1 --run_until "2025-03-10 12:00"

# Run for a specific number of rounds
python run_competition.py --rounds 100
```

## ML Strategy
The ML strategy uses machine learning to predict market movements. It has the following parameters:
- `lookback_periods`: Number of periods to look back for feature generation
- `prediction_horizon`: Number of periods ahead to predict
- `model_type`: Type of ML model (xgboost, randomforest, linear)
- `feature_selection`: Method for feature selection (pca, recursive, mutual_info, none)
- `stop_loss_pct`: Stop loss as a percentage of entry price
- `take_profit_pct`: Take profit as a percentage of entry price
- `risk_per_trade_pct`: Risk per trade as a percentage of account
- `confidence_threshold`: Threshold for confidence in signals

## Medallion Strategy
The Medallion strategy is based on technical indicators. It has the following parameters:
- `fast_ma_periods`: Fast moving average periods
- `slow_ma_periods`: Slow moving average periods
- `rsi_periods`: RSI periods
- `rsi_overbought`: RSI overbought level
- `rsi_oversold`: RSI oversold level
- `volatility_factor`: Factor for volatility filter
- `stop_loss_pct`: Stop loss as a percentage of entry price
- `take_profit_pct`: Take profit as a percentage of entry price
- `risk_per_trade_pct`: Risk per trade as a percentage of account

## Competition Results
The competition results are stored in the following locations:
- Leaderboard: `competition/leaderboard.json`
- Checkpoints: `competition/checkpoints/`
- Dashboard images: `competition/results/`

## Tips for Better Performance
1. **Balance Exploration and Exploitation**: Try different optimization techniques
2. **Adapt to Market Conditions**: Consider using different parameters for different market regimes
3. **Focus on Risk-Adjusted Returns**: Optimize for Sharpe ratio and drawdown, not just returns
4. **Golden Ratio**: The ratio of 1.618 often works well for take profit to stop loss ratios
5. **Parameter Constraints**: Ensure parameter combinations make logical sense (e.g., fast MA < slow MA)

## Contributing
Feel free to fork and contribute to this project. Some ideas for improvements:
- Add more strategies
- Implement more optimization algorithms
- Improve the dashboard
- Add real market data integration

## License
This project is licensed under the MIT License - see the LICENSE file for details. 