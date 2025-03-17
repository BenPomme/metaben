# MetaBen Trading Strategies Documentation

This document provides detailed information about the two high-performance trading strategies developed for MetaTrader 5 (MT5): Enhanced Medallion Strategy and ML-Enhanced Strategy. Both strategies are optimized for EURUSD trading with a $10,000 capital and 1:500 leverage, targeting approximately 1% daily return with a win rate above 55%.

## Table of Contents

1. [Common Features](#common-features)
2. [Enhanced Medallion Strategy](#enhanced-medallion-strategy)
   - [Overview](#medallion-overview)
   - [Technical Indicators](#medallion-indicators)
   - [Entry & Exit Conditions](#medallion-conditions)
   - [Parameters](#medallion-parameters)
   - [Risk Management](#medallion-risk)
3. [ML-Enhanced Strategy](#ml-enhanced-strategy)
   - [Overview](#ml-overview)
   - [Machine Learning Approach](#ml-approach)
   - [Technical Features](#ml-features)
   - [Entry & Exit Conditions](#ml-conditions)
   - [Parameters](#ml-parameters)
   - [Risk Management](#ml-risk)
4. [Data Preprocessing](#data-preprocessing)
5. [Usage Instructions](#usage-instructions)
   - [Python Backtesting](#python-backtesting)
   - [MT5 Expert Advisors](#mt5-ea)
6. [Performance Metrics](#performance-metrics)

## Common Features <a name="common-features"></a>

Both strategies share the following characteristics:

- **Data Gap Handling**: Advanced preprocessing to handle gaps in MT5 data
- **Multi-timeframe Analysis**: Primary (H1) with confirmation from secondary timeframes (H4, D1)
- **Leverage-Aware Position Sizing**: Intelligent calculation of position sizes based on account balance, risk percentage, and 1:500 leverage
- **Daily Trade Limits**: Restricts the number of trades per day to prevent overtrading
- **Daily Profit Target**: Stops trading after reaching 1% daily profit target
- **ATR-Based Stop Loss and Take Profit**: Adapts to market volatility

## Enhanced Medallion Strategy <a name="enhanced-medallion-strategy"></a>

### Overview <a name="medallion-overview"></a>

The Enhanced Medallion Strategy is a technical indicator-based trading system that combines moving averages, oscillators, and price action patterns to identify high-probability trading opportunities. It's designed to be responsive to short-term price movements while managing risk effectively.

### Technical Indicators <a name="medallion-indicators"></a>

- **Moving Averages**: Fast (5-period) and Slow (15-period) EMAs for trend identification
- **MACD**: For momentum confirmation with custom signal periods
- **RSI**: 10-period RSI for overbought/oversold conditions
- **Bollinger Bands**: 15-period with 1.8 standard deviations
- **ATR**: For volatility measurement and position sizing
- **Price Action Patterns**: Doji, bullish/bearish engulfing candles

### Entry & Exit Conditions <a name="medallion-conditions"></a>

#### Entry Conditions
The strategy uses a scoring system where at least 2 out of 5 conditions must be met:

**Buy Signal Conditions**:
1. MA Trend: MA cross up or uptrend (EMA fast > EMA slow)
2. RSI: Below 50 or increasing
3. MACD: Positive or increasing
4. Price Action: Bullish engulfing or doji
5. Bollinger Band: Price below middle band

**Sell Signal Conditions**:
1. MA Trend: MA cross down or downtrend (EMA fast < EMA slow)
2. RSI: Above 50 or decreasing
3. MACD: Negative or decreasing
4. Price Action: Bearish engulfing or doji
5. Bollinger Band: Price above middle band

#### Exit Conditions

**Exit Long Position**:
- Bearish engulfing pattern
- Price crosses below slow EMA
- RSI becomes overbought (>75)
- Price approaches upper Bollinger Band (>98% of upper band)
- MACD histogram crosses below zero

**Exit Short Position**:
- Bullish engulfing pattern
- Price crosses above slow EMA
- RSI becomes oversold (<25)
- Price approaches lower Bollinger Band (<102% of lower band)
- MACD histogram crosses above zero

### Parameters <a name="medallion-parameters"></a>

```python
{
    # Price action parameters
    'reversal_candle_strength': 0.6,     # Strength required for reversal candle
    'trend_strength_threshold': 0.5,     # Threshold for trend strength
    'support_resistance_lookback': 15,   # Periods to look back for S/R
    
    # Moving average parameters
    'fast_ma_period': 5,                 # Periods for fast MA 
    'slow_ma_period': 15,                # Periods for slow MA
    'signal_ma_period': 5,               # Periods for MACD signal line
    
    # RSI parameters
    'rsi_period': 10,                    # RSI calculation period
    'rsi_overbought': 75,                # RSI overbought threshold
    'rsi_oversold': 25,                  # RSI oversold threshold
    
    # Bollinger Band parameters
    'bb_period': 15,                     # BB calculation period
    'bb_std': 1.8,                       # BB standard deviation
    
    # Trade parameters
    'stop_loss_pips': 20,                # Default stop loss in pips
    'take_profit_pips': 40,              # Default take profit in pips
    'risk_percent': 2.5,                 # Risk per trade (%)
    'max_leverage': 500.0,               # Maximum leverage (1:500)
    'max_risk_per_trade': 5.0,           # Maximum risk per trade (%)
    'daily_target': 1.0,                 # Daily profit target (%)
    'max_trades_per_day': 5              # Maximum trades per day
}
```

### Risk Management <a name="medallion-risk"></a>

The Medallion strategy employs a robust risk management system:

1. **Position Sizing**: Calculates position size based on:
   - Account balance
   - Risk percentage per trade (2.5%)
   - Stop loss distance
   - Maximum leverage (1:500)

2. **Stop Loss & Take Profit**:
   - Dynamic stop loss based on ATR (1.0 × ATR)
   - Take profit at 2.0 × ATR (1:2 risk-reward ratio)

3. **Daily Risk Controls**:
   - Maximum 5 trades per day
   - Stops trading after reaching 1% daily profit
   - Maximum risk per trade capped at 5%

## ML-Enhanced Strategy <a name="ml-enhanced-strategy"></a>

### Overview <a name="ml-overview"></a>

The ML-Enhanced Strategy combines traditional technical analysis with machine learning to predict price movements. It uses a Random Forest Classifier trained on multiple technical features to identify high-probability trades.

### Machine Learning Approach <a name="ml-approach"></a>

- **Algorithm**: Random Forest Classifier with 150 estimators
- **Training/Testing Split**: 70% training, 30% testing
- **Prediction Horizon**: 4 periods ahead
- **Probability Threshold**: 0.53 (minimum confidence for trade)
- **Features**: 17 technical indicators and derived metrics

### Technical Features <a name="ml-features"></a>

The ML model is trained using the following features:

1. Price relative to moving averages:
   - Price to fast SMA ratio
   - Price to medium SMA ratio
   - Price to slow SMA ratio

2. Moving average crossovers:
   - Fast/Medium MA cross
   - Fast/Slow MA cross
   - Medium/Slow MA cross

3. Oscillators:
   - RSI value
   - RSI change
   - MACD line
   - MACD signal
   - MACD histogram
   - MACD change

4. Volatility metrics:
   - Bollinger Band position
   - Volatility (20-period)
   - ATR value
   - Close price percentage change
   - Volume percentage change

### Entry & Exit Conditions <a name="ml-conditions"></a>

#### Entry Conditions

**Buy Signal**:
- ML model predicts positive return (probability > 0.53)
- RSI is not extremely overbought (< 78)
- At least one of the following is true:
  - Uptrend confirmation (EMA fast > EMA medium)
  - Price near lower Bollinger Band (< 101% of middle band)
  - RSI showing upward momentum (RSI change > 0)

**Sell Signal**:
- ML model predicts negative return (probability > 0.53)
- RSI is not extremely oversold (> 22)
- At least one of the following is true:
  - Downtrend confirmation (EMA fast < EMA medium)
  - Price near upper Bollinger Band (> 99% of middle band)
  - RSI showing downward momentum (RSI change < 0)

#### Exit Conditions

**Exit Long Position**:
- ML prediction turns bearish (probability > 0.47)
- RSI becomes overbought (> 78)
- Price reaches upper Bollinger Band (> 98% of upper band)
- Trend reversal (EMA fast crosses below EMA medium)

**Exit Short Position**:
- ML prediction turns bullish (probability > 0.47)
- RSI becomes oversold (< 22)
- Price reaches lower Bollinger Band (< 102% of lower band)
- Trend reversal (EMA fast crosses above EMA medium)

### Parameters <a name="ml-parameters"></a>

```python
{
    # ML parameters
    'prediction_horizon': 4,      # Periods ahead to predict
    'ml_threshold': 0.53,         # Probability threshold for signals
    'train_size': 0.7,            # Train/test split ratio
    'random_state': 42,           # Random seed
    'n_estimators': 150,          # Trees in Random Forest
    
    # Moving average parameters
    'fast_ma_period': 6,          # Fast MA period
    'med_ma_period': 15,          # Medium MA period
    'slow_ma_period': 30,         # Slow MA period
    'signal_ma_period': 4,        # MACD signal period
    
    # RSI parameters
    'rsi_period': 8,              # RSI calculation period
    'rsi_overbought': 78,         # RSI overbought threshold
    'rsi_oversold': 22,           # RSI oversold threshold
    
    # Volatility parameters
    'bb_period': 12,              # BB calculation period
    'bb_std': 1.8,                # BB standard deviation
    'atr_period': 8,              # ATR calculation period
    
    # Trade parameters
    'stop_loss_atr': 0.8,         # Stop loss in ATR units
    'take_profit_atr': 1.6,       # Take profit in ATR units
    'risk_percent': 2.5,          # Risk per trade (%)
    'max_leverage': 500.0,        # Maximum leverage (1:500)
    'max_risk_per_trade': 5.0,    # Maximum risk per trade (%)
    'daily_target': 1.0,          # Daily profit target (%)
    'max_trades_per_day': 8       # Maximum trades per day
}
```

### Risk Management <a name="ml-risk"></a>

The ML-Enhanced strategy uses a similar risk management approach:

1. **Position Sizing**:
   - Based on account balance, risk percentage, and stop loss distance
   - Maximum leverage of 1:500
   - Risk per trade of 2.5%

2. **Stop Loss & Take Profit**:
   - Stop loss at 0.8 × ATR
   - Take profit at 1.6 × ATR (1:2 risk-reward ratio)

3. **Daily Risk Controls**:
   - Maximum 8 trades per day
   - Stops trading after reaching 1% daily profit
   - Maximum risk per trade capped at 5%

## Data Preprocessing <a name="data-preprocessing"></a>

Both strategies benefit from advanced data preprocessing to handle gaps in MT5 data:

1. **Gap Detection**: Identifies missing candles in the data
2. **Weekend Handling**: Special handling for weekend gaps
3. **Interpolation**: Linear interpolation for small gaps (< 3 candles)
4. **Holiday Handling**: Identification and handling of market holidays

The preprocessing greatly improves strategy performance by ensuring accurate technical indicator calculations.

## Usage Instructions <a name="usage-instructions"></a>

### Python Backtesting <a name="python-backtesting"></a>

To run backtests with the Python framework:

#### Medallion Strategy:
```bash
python backtests/enhanced_medallion_backtester.py
```

#### ML-Enhanced Strategy:
```bash
python backtests/enhanced_ml_backtester.py
```

#### With Custom Parameters:
```python
from backtests.enhanced_medallion_backtester import run_medallion_backtest
from backtests.enhanced_ml_backtester import run_ml_backtest
from datetime import datetime, timedelta

# Define dates
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # 1 year backtest

# Run Medallion strategy
results = run_medallion_backtest(
    symbol="EURUSD",
    timeframe="H1",
    secondary_timeframes=["H4", "D1"],
    start_date=start_date,
    end_date=end_date,
    balance=10000,
    preprocess_data=True,
    output_dir="medallion_results"
)

# Run ML strategy
results = run_ml_backtest(
    symbol="EURUSD",
    timeframe="H1",
    secondary_timeframes=["H4", "D1"],
    start_date=start_date,
    end_date=end_date,
    balance=10000,
    preprocess_data=True,
    output_dir="ml_results"
)
```

### MT5 Expert Advisors <a name="mt5-ea"></a>

To use the strategies directly in MetaTrader 5:

1. Copy the `.mq5` and `.ex5` files to your MetaTrader 5 `Experts` folder:
   - `Enhanced_Medallion_EA.mq5`
   - `ML_Enhanced_EA.mq5`

2. Open MetaTrader 5 and load the Expert Advisor on a chart:
   - Right-click on the chart
   - Select "Navigator"
   - Open "Expert Advisors"
   - Drag the EA onto the chart

3. Configure the EA parameters in the dialog:
   - Risk percentage
   - Leverage settings
   - Daily trade limits
   - Technical indicator parameters

4. For backtesting in MT5 Strategy Tester:
   - Open Strategy Tester (Ctrl+R)
   - Select the EA
   - Set the symbol, timeframe, and date range
   - Set "Model" to "Every tick" for accurate results
   - Click "Start" to run the backtest

## Performance Metrics <a name="performance-metrics"></a>

Both strategies are designed to achieve:

- **Daily Return**: ~1% on average
- **Win Rate**: >55%
- **Risk-Reward Ratio**: 1:2
- **Maximum Drawdown**: <10% 
- **Sharpe Ratio**: >1.5

The exact performance may vary based on market conditions, but the strategies have been optimized for EURUSD with 1:500 leverage on a $10,000 account. 