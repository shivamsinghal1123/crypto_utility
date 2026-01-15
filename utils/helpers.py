"""
Helper Functions
Utility functions and decorators
"""

import time
import requests
from functools import wraps
from typing import Callable, Any
import logging

from config.settings import REQUEST_CONFIG

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int, period: int):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Remove old calls outside the period
        self.calls = [call_time for call_time in self.calls if now - call_time < self.period]
        
        # Check if we need to wait
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self.calls = []
        
        # Add current call
        self.calls.append(time.time())


def handle_api_errors(func: Callable) -> Callable:
    """
    Decorator to handle API errors with retry logic.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        max_retries = REQUEST_CONFIG['MAX_RETRIES']
        backoff_factor = REQUEST_CONFIG['BACKOFF_FACTOR']
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout in {func.__name__}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = backoff_factor * (2 ** attempt)
                    logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Max retries reached for {func.__name__}")
                    return None
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error in {func.__name__}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = backoff_factor * (2 ** attempt)
                    time.sleep(sleep_time)
                else:
                    return None
            except ValueError as e:
                logger.error(f"Data validation error in {func.__name__}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                return None
        
        return None
    
    return wrapper


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format number with thousand separators.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
    
    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"
    
    return f"{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format percentage value.
    
    Args:
        value: Percentage value
        decimals: Number of decimal places
    
    Returns:
        Formatted string with % sign
    """
    if value is None:
        return "N/A"
    
    return f"{value:.{decimals}f}%"


def format_large_number(value: float) -> str:
    """
    Format large numbers with K, M, B suffixes.
    
    Args:
        value: Number to format
    
    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"
    
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"${value / 1_000:.2f}K"
    else:
        return f"${value:.2f}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
    
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0
    
    return ((new_value - old_value) / old_value) * 100


def normalize_symbol(symbol: str) -> str:
    """
    Normalize cryptocurrency symbol.
    
    Args:
        symbol: Symbol to normalize
    
    Returns:
        Normalized symbol (uppercase, with USDT if needed)
    """
    symbol = symbol.upper().strip()
    
    # Add USDT if not present
    if not symbol.endswith('USDT') and not symbol.endswith('USD') and not symbol.endswith('BUSD'):
        symbol = f"{symbol}USDT"
    
    return symbol


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails
    
    Returns:
        Result or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated string
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def get_coin_id_mapping(symbol: str) -> str:
    """
    Get CoinGecko coin ID from symbol.
    
    Args:
        symbol: Cryptocurrency symbol
    
    Returns:
        CoinGecko coin ID
    """
    # Common mappings
    mapping = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'BNB': 'binancecoin',
        'ADA': 'cardano',
        'SOL': 'solana',
        'XRP': 'ripple',
        'DOT': 'polkadot',
        'DOGE': 'dogecoin',
        'AVAX': 'avalanche-2',
        'MATIC': 'matic-network',
        'LINK': 'chainlink',
        'UNI': 'uniswap',
        'ATOM': 'cosmos',
        'LTC': 'litecoin',
        'BCH': 'bitcoin-cash',
        'ALGO': 'algorand',
        'VET': 'vechain',
        'FIL': 'filecoin',
        'TRX': 'tron',
        'ETC': 'ethereum-classic'
    }
    
    # Remove USDT/USD suffix
    clean_symbol = symbol.replace('USDT', '').replace('USD', '').replace('BUSD', '')
    
    return mapping.get(clean_symbol, clean_symbol.lower())


def setup_logging(log_file: str = None, log_level: str = 'INFO'):
    """
    Setup logging configuration.
    
    Args:
        log_file: Log file path
        log_level: Logging level
    """
    from config.settings import LOGGING_CONFIG
    
    log_file = log_file or LOGGING_CONFIG['LOG_FILE']
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(LOGGING_CONFIG['LOG_FORMAT'])
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logger.info("Logging configured")
