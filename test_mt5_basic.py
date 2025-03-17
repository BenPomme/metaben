try:
    import MetaTrader5 as mt5
    print("Successfully imported MetaTrader5!")
    
    # Try to initialize
    print("Initializing MT5...")
    initialized = mt5.initialize()
    print(f"MT5 initialization: {'Successful' if initialized else 'Failed'}")
    if not initialized:
        print(f"Last error: {mt5.last_error()}")
    else:
        # Get terminal info
        terminal_info = mt5.terminal_info()._asdict()
        print(f"Terminal path: {terminal_info['path']}")
        print(f"Connected: {terminal_info['connected']}")
        print(f"Community account: {terminal_info['community_account']}")
        
        # Get account info if connected
        if terminal_info['connected']:
            account_info = mt5.account_info()
            if account_info is not None:
                account_dict = account_info._asdict()
                print(f"Account: {account_dict.get('login')}")
                print(f"Server: {account_dict.get('server')}")
                print(f"Balance: {account_dict.get('balance')}")
            else:
                print("Failed to get account info")
        
        # Shutdown
        mt5.shutdown()
        print("MT5 shutdown complete")
        
except ImportError as e:
    print(f"Failed to import MetaTrader5: {e}")
    print("Make sure the package is installed with: pip install MetaTrader5") 