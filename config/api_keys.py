"""
API Keys and Authentication Configuration
Store your API keys here or in environment variables for security
"""

import os
from typing import Dict

class APIKeys:
    """Centralized API key management."""
    
    def __init__(self):
        # Binance API (No authentication required for public endpoints)
        self.BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
        self.BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
        
        # CoinGecko API
        self.COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY', '')
        
        # CoinMarketCap API
        self.COINMARKETCAP_API_KEY = os.getenv('COINMARKETCAP_API_KEY', '')
        
        # CryptoPanic API
        self.CRYPTOPANIC_API_KEY = os.getenv('CRYPTOPANIC_API_KEY', '')
        
        # Twitter API v2
        self.TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', '')
        
        # Reddit API
        self.REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
        self.REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
        self.REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'CryptoAnalyzer/1.0')
        
        # Etherscan API
        self.ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY', '')
        
        # BSCScan API
        self.BSCSCAN_API_KEY = os.getenv('BSCSCAN_API_KEY', '')
    
    def get_api_config(self) -> Dict[str, str]:
        """Return all API configurations as a dictionary."""
        return {
            'binance_api_key': self.BINANCE_API_KEY,
            'coingecko_api_key': self.COINGECKO_API_KEY,
            'coinmarketcap_api_key': self.COINMARKETCAP_API_KEY,
            'cryptopanic_api_key': self.CRYPTOPANIC_API_KEY,
            'twitter_bearer_token': self.TWITTER_BEARER_TOKEN,
            'etherscan_api_key': self.ETHERSCAN_API_KEY,
            'bscscan_api_key': self.BSCSCAN_API_KEY
        }
    
    def validate_keys(self) -> Dict[str, bool]:
        """Validate which API keys are configured."""
        return {
            'binance': bool(self.BINANCE_API_KEY),
            'coingecko': bool(self.COINGECKO_API_KEY),
            'coinmarketcap': bool(self.COINMARKETCAP_API_KEY),
            'cryptopanic': bool(self.CRYPTOPANIC_API_KEY),
            'twitter': bool(self.TWITTER_BEARER_TOKEN),
            'etherscan': bool(self.ETHERSCAN_API_KEY),
            'bscscan': bool(self.BSCSCAN_API_KEY)
        }
