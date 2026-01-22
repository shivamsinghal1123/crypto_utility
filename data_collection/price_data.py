"""
Price Data Collection Module
Collects OHLCV data, order book, and trading information from exchanges
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import API_ENDPOINTS, RATE_LIMITS, REQUEST_CONFIG
from utils.helpers import handle_api_errors, RateLimiter

logger = logging.getLogger(__name__)


class PriceDataCollector:
    """Collects price and volume data from cryptocurrency exchanges."""
    
    def __init__(self):
        self.binance_base = API_ENDPOINTS['BINANCE_BASE']
        self.coingecko_base = API_ENDPOINTS['COINGECKO_BASE']
        self.rate_limiter = RateLimiter(RATE_LIMITS['BINANCE'], 60)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': REQUEST_CONFIG['USER_AGENT']})
        # Disable SSL verification for environments with self-signed certificates
        self.session.verify = False
        # Suppress SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    @handle_api_errors
    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
    )
    def get_ohlcv_data(self, symbol: str, interval: str = '1m', limit: int = 100000) -> Optional[pd.DataFrame]:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data from Binance.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Candlestick interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to retrieve (max 100000)
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.binance_base}klines"
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': min(limit, 100000)
        }
        
        response = self.session.get(
            url, 
            params=params, 
            timeout=REQUEST_CONFIG['TIMEOUT']
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)
        
        df['trades'] = df['trades'].astype(int)
        
        logger.info(f"Retrieved {len(df)} candles for {symbol} ({interval})")
        return df
    
    @handle_api_errors
    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
    )
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current price and 24h statistics.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
        
        Returns:
            Dictionary with price statistics or None if failed
        """
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.binance_base}ticker/24hr"
        params = {'symbol': symbol.upper()}
        
        response = self.session.get(
            url, 
            params=params, 
            timeout=REQUEST_CONFIG['TIMEOUT']
        )
        response.raise_for_status()
        
        data = response.json()
        
        return {
            'symbol': data['symbol'],
            'price': float(data['lastPrice']),
            'price_change': float(data['priceChange']),
            'price_change_percent': float(data['priceChangePercent']),
            'high_24h': float(data['highPrice']),
            'low_24h': float(data['lowPrice']),
            'volume_24h': float(data['volume']),
            'quote_volume_24h': float(data['quoteVolume']),
            'trades_24h': int(data['count']),
            'timestamp': datetime.fromtimestamp(data['closeTime'] / 1000)
        }
    
    @handle_api_errors
    def get_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """
        Get order book depth data.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of price levels (5, 10, 20, 50, 100, 500, 1000)
        
        Returns:
            Dictionary with bid/ask data or None if failed
        """
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.binance_base}depth"
        params = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        
        response = self.session.get(
            url, 
            params=params, 
            timeout=REQUEST_CONFIG['TIMEOUT']
        )
        response.raise_for_status()
        
        data = response.json()
        
        bids_df = pd.DataFrame(data['bids'], columns=['price', 'quantity'])
        asks_df = pd.DataFrame(data['asks'], columns=['price', 'quantity'])
        
        bids_df = bids_df.astype(float)
        asks_df = asks_df.astype(float)
        
        return {
            'bids': bids_df,
            'asks': asks_df,
            'best_bid': float(data['bids'][0][0]) if data['bids'] else None,
            'best_ask': float(data['asks'][0][0]) if data['asks'] else None,
            'spread': float(data['asks'][0][0]) - float(data['bids'][0][0]) if data['bids'] and data['asks'] else None,
            'timestamp': datetime.now()
        }
    
    @handle_api_errors
    def get_recent_trades(self, symbol: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Get recent trades.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of trades (max 1000)
        
        Returns:
            DataFrame with recent trades or None if failed
        """
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.binance_base}trades"
        params = {
            'symbol': symbol.upper(),
            'limit': min(limit, 1000)
        }
        
        response = self.session.get(
            url, 
            params=params, 
            timeout=REQUEST_CONFIG['TIMEOUT']
        )
        response.raise_for_status()
        
        data = response.json()
        
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['price'] = df['price'].astype(float)
        df['qty'] = df['qty'].astype(float)
        
        return df
    
    def get_multiple_timeframes(self, symbol: str, intervals: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV data for multiple timeframes.
        
        Args:
            symbol: Trading pair symbol
            intervals: List of intervals (default from settings)
        
        Returns:
            Dictionary mapping intervals to DataFrames
        """
        if intervals is None:
            from config.settings import ANALYSIS_PARAMS
            intervals = ANALYSIS_PARAMS['TECHNICAL_TIMEFRAMES']
        
        result = {}
        for interval in intervals:
            df = self.get_ohlcv_data(symbol, interval)
            if df is not None:
                result[interval] = df
            time.sleep(0.1)  # Small delay between requests
        
        logger.info(f"Retrieved data for {len(result)} timeframes")
        return result
    
    @handle_api_errors
    def get_coingecko_data(self, coin_id: str) -> Optional[Dict]:
        """
        Get additional data from CoinGecko as backup.
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
        
        Returns:
            Dictionary with market data or None if failed
        """
        url = f"{self.coingecko_base}coins/{coin_id}"
        params = {
            'localization': 'false',
            'tickers': 'false',
            'community_data': 'true',
            'developer_data': 'true'
        }
        
        response = self.session.get(
            url, 
            params=params, 
            timeout=REQUEST_CONFIG['TIMEOUT']
        )
        response.raise_for_status()
        
        data = response.json()
        
        return {
            'name': data.get('name'),
            'symbol': data.get('symbol'),
            'market_cap_rank': data.get('market_cap_rank'),
            'current_price': data.get('market_data', {}).get('current_price', {}).get('usd'),
            'market_cap': data.get('market_data', {}).get('market_cap', {}).get('usd'),
            'total_volume': data.get('market_data', {}).get('total_volume', {}).get('usd'),
            'ath': data.get('market_data', {}).get('ath', {}).get('usd'),
            'ath_change_percentage': data.get('market_data', {}).get('ath_change_percentage', {}).get('usd'),
            'circulating_supply': data.get('market_data', {}).get('circulating_supply'),
            'total_supply': data.get('market_data', {}).get('total_supply'),
            'max_supply': data.get('market_data', {}).get('max_supply'),
        }
    
    def calculate_volume_profile(self, df: pd.DataFrame, num_bins: int = 20) -> Dict:
        """
        Calculate volume profile from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            num_bins: Number of price bins
        
        Returns:
            Dictionary with volume profile data
        """
        if df is None or len(df) == 0:
            return {}
        
        price_min = df['low'].min()
        price_max = df['high'].max()
        
        bins = np.linspace(price_min, price_max, num_bins + 1)
        bin_volume = np.zeros(num_bins)
        
        for _, row in df.iterrows():
            # Distribute volume across price range of the candle
            low_bin = np.digitize(row['low'], bins) - 1
            high_bin = np.digitize(row['high'], bins) - 1
            
            for i in range(max(0, low_bin), min(num_bins, high_bin + 1)):
                bin_volume[i] += row['volume'] / (high_bin - low_bin + 1)
        
        # Find Point of Control (POC) - highest volume price level
        poc_index = np.argmax(bin_volume)
        poc_price = (bins[poc_index] + bins[poc_index + 1]) / 2
        
        # Find Value Area (70% of volume)
        sorted_indices = np.argsort(bin_volume)[::-1]
        cumsum = 0
        value_area_indices = []
        
        for idx in sorted_indices:
            cumsum += bin_volume[idx]
            value_area_indices.append(idx)
            if cumsum >= 0.7 * bin_volume.sum():
                break
        
        vah = bins[max(value_area_indices) + 1]  # Value Area High
        val = bins[min(value_area_indices)]       # Value Area Low
        
        return {
            'poc': poc_price,
            'vah': vah,
            'val': val,
            'price_bins': bins,
            'volume_bins': bin_volume
        }
