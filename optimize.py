#!/usr/bin/env python
"""
Wrapper script to run the trading strategy optimization system.
This script simply imports and runs the main optimization script.
"""

import os
import sys
from pathlib import Path

# Add the optimization directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization'))

# Import and run the main optimization script
try:
    from optimization.run_strategy_optimization import main
    
    if __name__ == "__main__":
        sys.exit(main())
except ImportError as e:
    print(f"Error importing optimization modules: {e}")
    print("Please ensure all required modules are installed. See requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error running optimization: {e}")
    sys.exit(1) 