from mt5_connector import MT5Connector
import pandas as pd
from datetime import datetime, timedelta

def main():
    # Create MT5 connector instance
    mt5 = MT5Connector()
    
    # Try to connect
    print("Connecting to MetaTrader 5...")
    if mt5.connect():
        print("Connected successfully!")
        print("\nAccount Info:")
        print(f"Balance: ${mt5.account_info['balance']:.2f}")
        print(f"Equity: ${mt5.account_info['equity']:.2f}")
        
        # Try to get some recent data
        print("\nFetching recent EURUSD data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        data = mt5.get_data("EURUSD", "H1", start_date, end_date)
        if data is not None:
            print("\nRecent EURUSD H1 data:")
            print(data.tail())
        
        # Disconnect
        mt5.disconnect()
        print("\nDisconnected from MetaTrader 5")
    else:
        print("Failed to connect!")

if __name__ == "__main__":
    main() 