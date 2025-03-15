from mt5_connector import MT5Connector
from adaptive_ma_strategy import AdaptiveMAStrategy
import json
from datetime import datetime, timedelta

def main():
    # Load strategy config
    with open('config/strategy_config.json', 'r') as f:
        config = json.load(f)
    
    # Create MT5 connector instance
    mt5 = MT5Connector()
    
    # Connect to MT5
    print("Connecting to MetaTrader 5...")
    if mt5.connect():
        print("Connected successfully!")
        
        # Create strategy instance
        strategy = AdaptiveMAStrategy(
            symbol="EURUSD",
            primary_timeframe="H1",
            secondary_timeframes=["H4", "D1"],
            mt5_connector=mt5
        )
        
        # Set strategy parameters
        for key, value in config['parameters'].items():
            setattr(strategy, key, value)
        
        # Load recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days
        
        print("\nLoading historical data...")
        data = strategy.load_data(start_date=start_date, end_date=end_date)
        
        # Verify data
        if data:
            print("\nData Summary:")
            for tf, df in data.items():
                if df is not None:
                    print(f"\n{tf} Timeframe:")
                    print(f"Start: {df.index[0]}")
                    print(f"End: {df.index[-1]}")
                    print(f"Rows: {len(df)}")
                    print("\nLast 5 rows:")
                    print(df.tail())
        
        # Get current signal
        print("\nCalculating trading signals...")
        signal, strength = strategy.calculate_multi_timeframe_signal()
        
        print("\nTrading Signal:")
        print(f"Direction: {'Buy' if signal == 1 else 'Sell' if signal == -1 else 'Neutral'}")
        print(f"Strength: {strength:.2%}")
        
        # Get trade parameters if there's a signal
        if signal != 0:
            trade_params = strategy.generate_trade_parameters()
            if trade_params:
                print("\nTrade Parameters:")
                print(f"Entry Price: {trade_params['entry_price']:.5f}")
                print(f"Stop Loss: {trade_params['stop_loss']:.5f}")
                print(f"Take Profit: {trade_params['take_profit']:.5f}")
                print(f"Position Size: {trade_params['position_size']:.2f} lots")
                print(f"Risk Amount: ${trade_params['risk_amount']:.2f}")
                print(f"Potential Profit: ${trade_params['potential_profit']:.2f}")
        
        # Disconnect
        mt5.disconnect()
        print("\nDisconnected from MetaTrader 5")
    else:
        print("Failed to connect!")

if __name__ == "__main__":
    main() 