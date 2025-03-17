# Zorbite Trading Agent for MetaTrader 5

Zorbite is an advanced algorithmic trading agent designed specifically for XAUUSD (Gold) trading in MetaTrader 5. It combines traditional technical analysis with machine learning techniques to generate high-probability trading signals.

## Key Features

- **Intelligent Market Analysis**: Automatically detects market regimes and adjusts trading strategies accordingly.
- **Machine Learning Integration**: Uses XGBoost model trained on historical data to predict price movements.
- **Adaptive Risk Management**: Dynamically adjusts position sizes based on market volatility and prediction confidence.
- **Gold-Specific Strategies**: Optimized specifically for XAUUSD characteristics and behavior.
- **Multiple Timeframe Analysis**: Considers data from multiple timeframes for more robust trading decisions.
- **Mean Reversion and Trend Following**: Combines both strategies depending on market conditions.
- **Automated Pattern Detection**: Identifies key technical patterns and price action setups.

## System Requirements

- MetaTrader 5 (latest version)
- Windows 10 or later (for ML component)
- Python 3.8 or later (for ML component)
- Python packages: pandas, numpy, scikit-learn, xgboost, etc. (see requirements.txt)

## Installation

### Setting up the Expert Advisor in MT5

1. Copy `Zorbite_Strategy_EA.mq5` to your MetaTrader 5 `Experts` directory
   (typically `C:\Users\<YourUsername>\AppData\Roaming\MetaQuotes\Terminal\<YourTerminalID>\MQL5\Experts\`)
   
2. Open MetaTrader 5 and compile the EA by double-clicking on the file in the Navigator panel

### Setting up the ML Component

1. Install Python requirements:
   ```
   pip install -r requirements.txt
   ```

2. Configure your MT5 connection in `.env` file:
   ```
   MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
   MT5_LOGIN=your_login
   MT5_PASSWORD=your_password
   MT5_SERVER=your_server
   ```

3. Run the ML server:
   ```
   python python_scripts/zorbite_ml.py
   ```
   
   Or simply use the provided batch file:
   ```
   run_zorbite_ml.bat
   ```

## Usage

### Running the EA

1. Drag and drop the Zorbite EA onto a XAUUSD chart
2. Configure the input parameters as needed
3. Ensure that the ML server is running if you want to use ML-powered features
4. Click "OK" to start the EA

### Parameter Configuration

The EA provides many parameters that can be adjusted for different trading styles and market conditions. Key parameters include:

- **Risk_Percent**: Percentage of account balance to risk per trade (default: 1.0%)
- **StopLoss_Pct**: Stop loss as a percentage of price (default: 1.5%)
- **TakeProfit_Pct**: Take profit as a percentage of price (default: 2.5%)
- **Use_ML_Prediction**: Whether to use machine learning predictions (default: true)
- **ML_Confidence_Threshold**: Minimum confidence level required for ML signals (default: 0.65)
- **Use_Market_Regime_Detection**: Whether to detect market regimes and adapt strategies (default: true)

See the EA's input settings for a complete list of parameters.

## Trading Strategy

Zorbite combines several trading strategies based on the current market regime:

1. **Trending Market**:
   - Employs trend-following strategies with moving average crossovers
   - Uses longer-term momentum indicators for confirmation
   - Sets wider stop losses to accommodate volatility

2. **Ranging Market**:
   - Uses mean-reversion techniques with Bollinger Bands and RSI
   - Identifies support and resistance levels
   - Sets tighter stop losses and take profits

3. **Volatile/Uncertain Market**:
   - More conservative approach with higher confidence thresholds
   - Waits for ML confirmation with high confidence
   - Reduces position sizes

4. **Default Market**:
   - Balanced approach combining both trend and mean-reversion signals
   - Standard risk parameters
   - Requires confluence of multiple indicators

## Backtest Results

The EA has been extensively tested on historical XAUUSD data. Detailed backtest results can be found in the `/backtests` directory.

## Troubleshooting

**ML Server Connection Issues**:
- Ensure the ML server is running
- Check that port 9876 is not blocked by firewall
- Verify that the client can connect to localhost

**MT5 Operation Issues**:
- Check that AutoTrading is enabled in MT5
- Ensure you have proper permissions for your account
- Verify that the symbol is available for trading

## Disclaimer

Trading involves risk. Past performance is not indicative of future results. This Expert Advisor is provided for educational and informational purposes only. Always test thoroughly on a demo account before using on a live account.

## License

Copyright © 2024 Zorbite AI Technologies

This software is provided "as is" without warranty of any kind.

## Contact

For support, questions, or feature requests, please contact support@zorbite.ai or open an issue on GitHub. 