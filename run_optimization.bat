@echo off
REM Batch script to run the trading strategy optimization system

echo Starting Trading Strategy Optimization System...

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "optimization_checkpoints" mkdir optimization_checkpoints
if not exist "optimization_checkpoints\ml" mkdir optimization_checkpoints\ml
if not exist "optimization_checkpoints\medallion" mkdir optimization_checkpoints\medallion

REM Default optimization (both strategies)
echo.
echo Run option: 
echo 1. Optimize both strategies (ML and Medallion)
echo 2. Optimize ML strategy only
echo 3. Optimize Medallion strategy only
echo 4. Run with dashboard
echo 5. Quick test run (10 iterations)
echo 6. Advanced configuration
echo.

set /p choice="Select option (1-6): "

if "%choice%"=="1" (
    echo Running optimization for both strategies...
    python run_strategy_optimization.py --strategies ml,medallion --symbol EURUSD --primary_timeframe H1 --secondary_timeframes "H4,D1" --start_date 2023-01-01 --end_date 2023-12-31 --balance 10000 --algorithm bayesian --iterations 100
) else if "%choice%"=="2" (
    echo Running optimization for ML strategy only...
    python run_strategy_optimization.py --strategies ml --symbol EURUSD --primary_timeframe H1 --secondary_timeframes "H4,D1" --start_date 2023-01-01 --end_date 2023-12-31 --balance 10000 --algorithm bayesian --iterations 100
) else if "%choice%"=="3" (
    echo Running optimization for Medallion strategy only...
    python run_strategy_optimization.py --strategies medallion --symbol EURUSD --primary_timeframe H1 --secondary_timeframes "H4,D1" --start_date 2023-01-01 --end_date 2023-12-31 --balance 10000 --algorithm bayesian --iterations 100
) else if "%choice%"=="4" (
    echo Running optimization with dashboard...
    python run_strategy_optimization.py --strategies ml,medallion --symbol EURUSD --primary_timeframe H1 --secondary_timeframes "H4,D1" --start_date 2023-01-01 --end_date 2023-12-31 --balance 10000 --algorithm bayesian --iterations 100 --dashboard
) else if "%choice%"=="5" (
    echo Running quick test optimization (10 iterations)...
    python run_strategy_optimization.py --strategies ml,medallion --symbol EURUSD --primary_timeframe H1 --secondary_timeframes "H4,D1" --start_date 2023-01-01 --end_date 2023-01-31 --balance 10000 --algorithm random --iterations 10
) else if "%choice%"=="6" (
    echo Advanced configuration selected.
    
    echo.
    set /p strategies="Enter strategies (comma-separated, ml,medallion): "
    set /p symbol="Enter symbol (default: EURUSD): "
    set /p primary_tf="Enter primary timeframe (default: H1): "
    set /p secondary_tf="Enter secondary timeframes (comma-separated, default: H4,D1): "
    set /p start_date="Enter start date (YYYY-MM-DD, default: 2023-01-01): "
    set /p end_date="Enter end date (YYYY-MM-DD, default: 2023-12-31): "
    set /p balance="Enter initial balance (default: 10000): "
    set /p algorithm="Enter algorithm (bayesian, genetic, random, grid, optuna, default: bayesian): "
    set /p iterations="Enter number of iterations (default: 100): "
    set /p dashboard="Run with dashboard? (y/n, default: n): "
    
    set cmd=python run_strategy_optimization.py
    
    if not "%strategies%"=="" set cmd=%cmd% --strategies %strategies%
    if not "%symbol%"=="" set cmd=%cmd% --symbol %symbol%
    if not "%primary_tf%"=="" set cmd=%cmd% --primary_timeframe %primary_tf%
    if not "%secondary_tf%"=="" set cmd=%cmd% --secondary_timeframes "%secondary_tf%"
    if not "%start_date%"=="" set cmd=%cmd% --start_date %start_date%
    if not "%end_date%"=="" set cmd=%cmd% --end_date %end_date%
    if not "%balance%"=="" set cmd=%cmd% --balance %balance%
    if not "%algorithm%"=="" set cmd=%cmd% --algorithm %algorithm%
    if not "%iterations%"=="" set cmd=%cmd% --iterations %iterations%
    if /i "%dashboard%"=="y" set cmd=%cmd% --dashboard
    
    echo Running command: %cmd%
    %cmd%
) else (
    echo Invalid option selected. Exiting.
    goto :end
)

:end
echo.
echo Optimization complete. Results are available in the optimization_checkpoints directory.
echo Log files are available in the logs directory.

pause 