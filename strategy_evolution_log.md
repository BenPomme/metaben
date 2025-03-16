# Strategy Evolution Log

## Version 1.0 - Base Strategies (March 2024)

### ML_Strategy_EA_v1
- **Type**: Machine Learning Based Strategy
- **Performance**: Poor
  - Win Rate: 44.59%
  - Annual Return: 3.81%
  - Max Drawdown: 16.64%
  - Sharpe Ratio: 0.31
- **Key Components**:
  - XGBoost model
  - PCA feature selection
  - Simple MA crossover confirmation
  - RSI filter
- **Technical Issues Fixed**:
  - Fixed unsupported filling mode error by implementing proper filling mode detection
  - Added proper error handling for order execution
- **Learnings**:
  1. Model showed signs of overfitting (large gap between backtest and live performance)
  2. Simple MA/RSI confirmation may be too basic for ML strategy
  3. Fixed trade execution issues but performance remained poor
  4. Need better feature engineering and model validation

### Medallion_Strategy_EA_v1
- **Type**: Traditional Technical Strategy
- **Performance**: Moderate
  - Win Rate: 62.16%
  - Annual Return: 17.16%
  - Max Drawdown: 21.15%
  - Sharpe Ratio: 1.28
- **Key Components**:
  - EMA crossover (24/52)
  - RSI (20) with dynamic levels
  - Volatility filter using ATR
- **Technical Issues Fixed**:
  - Resolved filling mode errors using symbol-specific filling mode detection
  - Improved order execution reliability
- **Learnings**:
  1. More robust than ML strategy in live trading
  2. Trade execution improved after fixing filling mode issues
  3. Risk management parameters need optimization
  4. ATR-based volatility filter shows promise

## Naming Convention for Future Versions

### Format: `{StrategyType}_{Version}_{Variant}`

Examples:
- `ML_Strategy_EA_v2_Enhanced` - Second version with enhanced features
- `ML_Strategy_EA_v2_Ensemble` - Second version using ensemble approach
- `Medallion_Strategy_EA_v2_Adaptive` - Second version with adaptive parameters
- `Medallion_Strategy_EA_v2_MultiTF` - Second version with multiple timeframe analysis

### Version Numbering:
- Major versions (v1, v2, v3): Significant strategy changes
- Variants (Enhanced, Adaptive, etc.): Specific improvements within major versions

## Recommendations for Next Iterations

### ML Strategy v2:
1. Implement proper walk-forward optimization
2. Add more sophisticated feature engineering
3. Consider ensemble approach with multiple models
4. Add adaptive parameter adjustment

### Medallion Strategy v2:
1. Implement adaptive parameter optimization
2. Add multiple timeframe analysis
3. Enhance volatility filtering mechanism
4. Improve exit strategy logic

## Technical Implementation Notes

### Critical Components to Maintain:
```mql5
// Proper filling mode handling
int filling_mode = (int)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
if((filling_mode & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
    request.type_filling = ORDER_FILLING_IOC;
else if((filling_mode & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
    request.type_filling = ORDER_FILLING_FOK;
else
    request.type_filling = ORDER_FILLING_RETURN;
```

### Risk Management Template:
```mql5
double CalculateLotSize(double entry_price, double sl_price)
{
    double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double risk_amount = account_balance * RiskPerTrade_Pct / 100;
    // ... lot size calculation logic ...
}
``` 