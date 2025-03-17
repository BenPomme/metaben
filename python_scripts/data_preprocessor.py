import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import pytz
from typing import Dict, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_preprocessor")

class MTDataPreprocessor:
    """
    Preprocessor for MetaTrader 5 data to handle gaps and ensure data quality
    """
    
    def __init__(self, interpolate_gaps=True, max_gap_to_interpolate=None, validate_quality=True):
        """
        Initialize the preprocessor
        
        Args:
            interpolate_gaps: Whether to interpolate values for small gaps
            max_gap_to_interpolate: Dictionary of maximum gap sizes to interpolate by timeframe, 
                                   e.g. {'H1': 3, 'H4': 2, 'D1': 1}
            validate_quality: Whether to validate data quality after preprocessing
        """
        self.interpolate_gaps = interpolate_gaps
        self.max_gap_to_interpolate = max_gap_to_interpolate or {
            'M1': 5,      # 5 minutes
            'M5': 3,      # 15 minutes
            'M15': 2,     # 30 minutes
            'M30': 2,     # 1 hour
            'H1': 3,      # 3 hours
            'H4': 2,      # 8 hours
            'D1': 1,      # 1 day
            'W1': 0,      # Don't interpolate weekly data
        }
        self.validate_quality = validate_quality
        
        # Expected time differences by timeframe
        self.expected_diffs = {
            'M1': pd.Timedelta(minutes=1),
            'M5': pd.Timedelta(minutes=5),
            'M15': pd.Timedelta(minutes=15),
            'M30': pd.Timedelta(minutes=30),
            'H1': pd.Timedelta(hours=1),
            'H4': pd.Timedelta(hours=4),
            'D1': pd.Timedelta(days=1),
            'W1': pd.Timedelta(weeks=1),
        }
        
        # Market session hours for identifying expected gaps
        # Using a simplified version - forex markets typically close on weekend
        self.weekend_days = [5, 6]  # Saturday (5) and Sunday (6)
        
    def preprocess(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess MT5 data for all timeframes
        
        Args:
            data: Dictionary of DataFrames by timeframe
            
        Returns:
            Dictionary of preprocessed DataFrames by timeframe
        """
        logger.info("Starting data preprocessing...")
        
        processed_data = {}
        
        for timeframe, df in data.items():
            if df is None or df.empty:
                logger.warning(f"No data for {timeframe} timeframe")
                processed_data[timeframe] = df
                continue
                
            logger.info(f"Preprocessing {timeframe} data with {len(df)} rows")
            
            # 1. Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'time' in df.columns:
                    df.set_index('time', inplace=True)
                    logger.info(f"Set 'time' column as index for {timeframe}")
                else:
                    logger.warning(f"Cannot set datetime index for {timeframe}, skipping")
                    processed_data[timeframe] = df
                    continue
            
            # 2. Sort by time index
            df = df.sort_index()
            
            # 3. Handle duplicate timestamps
            if df.index.duplicated().any():
                logger.warning(f"Found duplicate timestamps in {timeframe}, keeping last values")
                df = df[~df.index.duplicated(keep='last')]
            
            # 4. Find and fix gaps
            processed_df = self._handle_gaps(df, timeframe)
            
            # 5. Validate data quality
            if self.validate_quality:
                quality_report = self._validate_data_quality(processed_df, timeframe)
                if quality_report['critical_issues']:
                    logger.warning(f"Critical data quality issues found for {timeframe}:\n{quality_report}")
            
            processed_data[timeframe] = processed_df
            logger.info(f"Completed preprocessing {timeframe} data, output has {len(processed_df)} rows")
            
        return processed_data
    
    def _handle_gaps(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Find and handle gaps in the data
        
        Args:
            df: DataFrame with price data
            timeframe: Timeframe string ('H1', 'D1', etc.)
            
        Returns:
            DataFrame with gaps handled
        """
        if timeframe not in self.expected_diffs:
            logger.warning(f"Unknown timeframe {timeframe}, using defaults")
            expected_diff = pd.Timedelta(hours=1)
        else:
            expected_diff = self.expected_diffs[timeframe]
        
        # Calculate time differences between consecutive rows
        time_diff = df.index.to_series().diff()
        
        # Filter out weekend gaps for forex data
        if timeframe in ['H1', 'H4', 'D1']:
            # Don't treat weekend gaps as missing data
            weekend_gaps = []
            for idx, diff in time_diff.items():
                if diff is not pd.NaT and diff > expected_diff:
                    # Check if gap spans weekend
                    start_date = idx - diff
                    weekend_days_in_gap = 0
                    current_date = start_date
                    while current_date < idx:
                        if current_date.weekday() in self.weekend_days:
                            weekend_days_in_gap += 1
                        current_date += timedelta(days=1)
                    
                    # If gap includes weekend, adjust the expected diff
                    if weekend_days_in_gap > 0:
                        adjusted_diff = diff - pd.Timedelta(days=weekend_days_in_gap)
                        if adjusted_diff <= expected_diff * 1.5:  # Allow some tolerance
                            weekend_gaps.append(idx)
        
            # Create a mask for non-weekend gaps
            unexpected_gaps = time_diff[time_diff > expected_diff].index.difference(pd.Index(weekend_gaps))
        else:
            unexpected_gaps = time_diff[time_diff > expected_diff].index
        
        if len(unexpected_gaps) > 0:
            logger.info(f"Found {len(unexpected_gaps)} unexpected gaps in {timeframe} data")
            
            # Interpolate gaps if configured
            if self.interpolate_gaps and self.max_gap_to_interpolate.get(timeframe, 0) > 0:
                # Process each gap for possible interpolation
                interpolated_rows = 0
                max_units = self.max_gap_to_interpolate.get(timeframe, 0)
                
                # Create a copy of the dataframe for interpolation
                interpolated_df = df.copy()
                
                for gap_end in unexpected_gaps:
                    gap_start_idx = df.index.get_loc(gap_end) - 1
                    if gap_start_idx < 0:
                        continue  # Skip if it's the first row
                    
                    gap_start = df.index[gap_start_idx]
                    gap_size = gap_end - gap_start
                    
                    # Convert gap size to appropriate units for timeframe
                    if timeframe.startswith('M'):
                        minutes = gap_size.total_seconds() / 60
                        expected_minutes = self.expected_diffs[timeframe].total_seconds() / 60
                        gap_units = minutes / expected_minutes
                    elif timeframe.startswith('H'):
                        hours = gap_size.total_seconds() / 3600
                        expected_hours = self.expected_diffs[timeframe].total_seconds() / 3600
                        gap_units = hours / expected_hours
                    elif timeframe == 'D1':
                        gap_units = gap_size.days
                    else:  # W1
                        gap_units = gap_size.days / 7
                    
                    # Only interpolate if gap is small enough
                    if gap_units <= max_units:
                        # Create a time range for missing points
                        missing_times = pd.date_range(
                            start=gap_start + self.expected_diffs[timeframe], 
                            end=gap_end - self.expected_diffs[timeframe],
                            freq=self._get_freq_from_timeframe(timeframe)
                        )
                        
                        if len(missing_times) > 0:
                            # Create a DataFrame with missing points
                            before_row = df.loc[gap_start]
                            after_row = df.loc[gap_end]
                            
                            # For each missing point, linearly interpolate values
                            for missing_time in missing_times:
                                # Calculate the position of this time between gap_start and gap_end (0 to 1)
                                position = (missing_time - gap_start) / gap_size
                                
                                # Interpolate values for this point
                                new_row = before_row + position * (after_row - before_row)
                                interpolated_df.loc[missing_time] = new_row
                                interpolated_rows += 1
                
                # Sort the dataframe with interpolated values
                if interpolated_rows > 0:
                    logger.info(f"Interpolated {interpolated_rows} rows for {timeframe}")
                    interpolated_df = interpolated_df.sort_index()
                    df = interpolated_df
        
        return df
    
    def _get_freq_from_timeframe(self, timeframe: str) -> str:
        """Convert timeframe string to pandas frequency string"""
        freq_map = {
            'M1': 'T',     # minute
            'M5': '5T',    # 5 minutes
            'M15': '15T',  # 15 minutes
            'M30': '30T',  # 30 minutes
            'H1': 'H',     # hour
            'H4': '4H',    # 4 hours
            'D1': 'D',     # day
            'W1': 'W',     # week
        }
        return freq_map.get(timeframe, 'H')
    
    def _validate_data_quality(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """
        Validate data quality and return a report
        
        Args:
            df: DataFrame with price data
            timeframe: Timeframe string
            
        Returns:
            Dictionary with quality metrics and issues
        """
        quality_report = {
            'timeframe': timeframe,
            'row_count': len(df),
            'missing_values': df.isna().sum().to_dict(),
            'critical_issues': [],
            'warnings': [],
        }
        
        # Check for remaining gaps
        if df.index.to_series().diff().max() > self.expected_diffs.get(timeframe, pd.Timedelta(hours=1)) * 1.5:
            quality_report['critical_issues'].append('Data still contains significant gaps')
        
        # Check for price anomalies (zero or negative prices)
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                if (df[col] <= 0).any():
                    quality_report['critical_issues'].append(f'Found zero or negative values in {col}')
        
        # Check for high-low price inconsistencies
        if all(col in df.columns for col in ['high', 'low']):
            if (df['high'] < df['low']).any():
                quality_report['critical_issues'].append('Found high price less than low price')
        
        # Check for unrealistic price changes
        if 'close' in df.columns:
            pct_change = df['close'].pct_change().abs()
            extreme_changes = pct_change[pct_change > 0.1]  # 10% change threshold
            if not extreme_changes.empty:
                quality_report['warnings'].append(f'Found {len(extreme_changes)} extreme price changes > 10%')
        
        return quality_report


# Function to detect and fix common data issues
def preprocess_mt5_data(data, interpolate_gaps=True, validate=True):
    """
    Preprocess MT5 data to handle common issues
    
    Args:
        data: Dictionary of DataFrames by timeframe
        interpolate_gaps: Whether to interpolate small gaps
        validate: Whether to validate data quality
        
    Returns:
        Dictionary of preprocessed DataFrames by timeframe
    """
    preprocessor = MTDataPreprocessor(
        interpolate_gaps=interpolate_gaps,
        validate_quality=validate
    )
    return preprocessor.preprocess(data)


if __name__ == "__main__":
    # Example usage
    import MetaTrader5 as mt5
    from datetime import datetime, timedelta
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        import sys
        sys.exit(1)
    
    # Define timeframes
    timeframe_dict = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1
    }
    
    # Get data for EURUSD
    symbol = "EURUSD"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Get data for different timeframes
    data = {}
    for tf, mt5_tf in timeframe_dict.items():
        if tf in ['H1', 'H4', 'D1']:  # Only get these timeframes for the example
            # Convert dates to UTC timezone
            timezone = pytz.timezone("UTC")
            start = timezone.localize(start_date)
            end = timezone.localize(end_date)
            
            # Get rates
            rates = mt5.copy_rates_range(symbol, mt5_tf, start, end)
            
            if rates is not None and len(rates) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                data[tf] = df
    
    # Preprocess data
    preprocessed_data = preprocess_mt5_data(data)
    
    # Print results
    for tf, df in preprocessed_data.items():
        print(f"\n{tf} data:")
        print(f"Original: {len(data[tf])} rows")
        print(f"Processed: {len(df)} rows")
        
    # Shutdown MT5
    mt5.shutdown() 