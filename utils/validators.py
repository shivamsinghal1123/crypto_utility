"""
Validators Module
Data validation functions
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

from config.settings import VALIDATION_SETTINGS

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data from various sources."""
    
    def __init__(self):
        self.settings = VALIDATION_SETTINGS
    
    def validate_ohlcv_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate OHLCV DataFrame.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if df is None or df.empty:
            return False, ["DataFrame is None or empty"]
        
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return False, errors
        
        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            errors.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Validate price relationships (High >= Low, High >= Open, High >= Close, etc.)
        invalid_highs = (df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close'])
        if invalid_highs.any():
            errors.append(f"Invalid high prices at indices: {df[invalid_highs].index.tolist()}")
        
        invalid_lows = (df['low'] > df['high']) | (df['low'] > df['open']) | (df['low'] > df['close'])
        if invalid_lows.any():
            errors.append(f"Invalid low prices at indices: {df[invalid_lows].index.tolist()}")
        
        # Check for negative prices
        negative_prices = (df['open'] < 0) | (df['high'] < 0) | (df['low'] < 0) | (df['close'] < 0)
        if negative_prices.any():
            errors.append(f"Negative prices found at indices: {df[negative_prices].index.tolist()}")
        
        # Check for negative volume
        negative_volume = df['volume'] < 0
        if negative_volume.any():
            errors.append(f"Negative volume found at indices: {df[negative_volume].index.tolist()}")
        
        # Check for abnormal price changes (possible data errors)
        if len(df) > 1:
            price_changes = df['close'].pct_change().abs()
            max_change = self.settings['MAX_PRICE_CHANGE']
            abnormal_changes = price_changes > max_change
            
            if abnormal_changes.any():
                errors.append(
                    f"Abnormal price changes (>{max_change*100}%) at indices: "
                    f"{df[abnormal_changes].index.tolist()}"
                )
        
        # Check for gaps in timestamps
        if self.settings['REQUIRE_COMPLETE_CANDLES'] and len(df) > 1:
            time_diffs = df['timestamp'].diff()
            # Skip first NaT
            time_diffs = time_diffs[1:]
            
            # Check if all time differences are consistent (within reason)
            if not time_diffs.empty:
                mode_diff = time_diffs.mode()[0] if not time_diffs.mode().empty else None
                if mode_diff:
                    # Allow some tolerance for slight variations
                    inconsistent = abs(time_diffs - mode_diff) > pd.Timedelta(minutes=1)
                    if inconsistent.any():
                        errors.append(f"Inconsistent time intervals detected")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_price_data(self, price_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate price data dictionary.
        
        Args:
            price_data: Price data dictionary
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if not price_data:
            return False, ["Price data is None or empty"]
        
        # Check required fields
        required_fields = ['symbol', 'price']
        missing_fields = [field for field in required_fields if field not in price_data]
        
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
            return False, errors
        
        # Validate price
        price = price_data.get('price')
        if price is None or price < self.settings['MIN_PRICE']:
            errors.append(f"Invalid price: {price}")
        
        # Validate volume if present
        if 'volume_24h' in price_data:
            volume = price_data['volume_24h']
            if volume is not None and volume < self.settings['MIN_VOLUME']:
                logger.warning(f"Low volume: {volume}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_news_data(self, news_list: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Validate news data list.
        
        Args:
            news_list: List of news articles
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if not news_list:
            logger.warning("News list is empty")
            return True, []  # Empty is valid, just no data
        
        required_fields = ['title', 'published']
        
        for idx, article in enumerate(news_list):
            missing_fields = [field for field in required_fields if field not in article]
            if missing_fields:
                errors.append(f"Article {idx} missing fields: {missing_fields}")
            
            # Validate sentiment score if present
            if 'sentiment' in article:
                sentiment = article['sentiment']
                if sentiment < -1 or sentiment > 1:
                    errors.append(f"Article {idx} has invalid sentiment: {sentiment}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_symbol(self, symbol: str) -> Tuple[bool, str]:
        """
        Validate cryptocurrency symbol.
        
        Args:
            symbol: Symbol to validate
        
        Returns:
            Tuple of (is_valid, error message or empty string)
        """
        if not symbol:
            return False, "Symbol is empty"
        
        if not isinstance(symbol, str):
            return False, "Symbol must be a string"
        
        # Basic validation - should be alphanumeric
        if not symbol.replace('USDT', '').replace('USD', '').replace('BUSD', '').isalnum():
            return False, "Symbol contains invalid characters"
        
        if len(symbol) < 2:
            return False, "Symbol is too short"
        
        if len(symbol) > 20:
            return False, "Symbol is too long"
        
        return True, ""
    
    def validate_analysis_results(self, results: Dict, required_keys: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate analysis results dictionary.
        
        Args:
            results: Results dictionary
            required_keys: List of required keys
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if not results:
            return False, ["Results dictionary is None or empty"]
        
        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            errors.append(f"Missing required keys: {missing_keys}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize DataFrame by removing/fixing invalid data.
        
        Args:
            df: DataFrame to sanitize
        
        Returns:
            Sanitized DataFrame
        """
        if df is None or df.empty:
            return df
        
        # Remove rows with null critical values
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Fix negative prices (replace with forward fill)
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df.loc[df[col] < 0, col] = None
        
        df[price_columns] = df[price_columns].fillna(method='ffill')
        
        # Fix volume (replace negatives with 0)
        if 'volume' in df.columns:
            df.loc[df['volume'] < 0, 'volume'] = 0
        
        # Remove duplicate timestamps
        if 'timestamp' in df.columns:
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        logger.info(f"Sanitized DataFrame: {len(df)} rows remaining")
        return df
