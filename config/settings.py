"""
Global Settings and Configuration
Central configuration file for the crypto analyzer framework
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# API Endpoints
API_ENDPOINTS = {
    'BINANCE_BASE': 'https://api.binance.com/api/v3/',
    'COINGECKO_BASE': 'https://api.coingecko.com/api/v3/',
    'COINMARKETCAP_BASE': 'https://pro-api.coinmarketcap.com/v1/',
    'DEFILLAMA_BASE': 'https://api.llama.fi/',
    'CRYPTOPANIC_BASE': 'https://cryptopanic.com/api/v1/',
    'ETHERSCAN_BASE': 'https://api.etherscan.io/api',
    'BSCSCAN_BASE': 'https://api.bscscan.com/api',
    'COINDESK_RSS': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'COINTELEGRAPH_RSS': 'https://cointelegraph.com/rss'
}

# Data Storage Configuration
DATA_STORAGE = {
    'DATABASE_PATH': str(DATA_DIR / 'crypto_analysis.db'),
    'CACHE_DURATION': 300,  # 5 minutes in seconds
    'HISTORY_DAYS': 365,
    'MAX_CACHE_SIZE': 1000  # Maximum number of cached items
}

# Analysis Parameters
ANALYSIS_PARAMS = {
    'TECHNICAL_TIMEFRAMES': ['1h', '4h', '1d', '5m', '1m'],
    'NEWS_LOOKBACK_DAYS': 7,
    'SENTIMENT_WEIGHT': 0.3,
    'FUNDAMENTAL_WEIGHT': 0.4,
    'TECHNICAL_WEIGHT': 0.3,
    'MIN_CONFIDENCE_SCORE': 0.6
}

# Technical Analysis Settings
TECHNICAL_SETTINGS = {
    'RSI_PERIOD': 14,
    'RSI_OVERBOUGHT': 70,
    'RSI_OVERSOLD': 30,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'BB_PERIOD': 20,
    'BB_STD_DEV': 2,
    'SMA_PERIODS': [20, 50, 100, 200],
    'EMA_PERIODS': [12, 26, 50],
    'ATR_PERIOD': 14,
    'VOLUME_SMA_PERIOD': 20
}

# Rate Limiting (requests per minute)
RATE_LIMITS = {
    'BINANCE': 1200,
    'COINGECKO': 50,
    'COINMARKETCAP': 30,
    'ETHERSCAN': 5,
    'BSCSCAN': 5,
    'CRYPTOPANIC': 100,
    'REDDIT': 60,
    'TWITTER': 300
}

# Request Configuration
REQUEST_CONFIG = {
    'TIMEOUT': 30,  # seconds
    'MAX_RETRIES': 3,
    'BACKOFF_FACTOR': 0.3,
    'USER_AGENT': 'CryptoAnalyzer/1.0'
}

# Logging Configuration
LOGGING_CONFIG = {
    'LOG_FILE': str(LOGS_DIR / 'crypto_analyzer.log'),
    'LOG_LEVEL': 'INFO',
    'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'MAX_LOG_SIZE': 10 * 1024 * 1024,  # 10 MB
    'BACKUP_COUNT': 5
}

# Support/Resistance Calculation Settings
SR_SETTINGS = {
    'LOOKBACK_PERIODS': 100,
    'MIN_TOUCHES': 2,
    'TOLERANCE_PERCENTAGE': 0.5,
    'PIVOT_TIMEFRAME': '1d',
    'FIBONACCI_LEVELS': [0.236, 0.382, 0.5, 0.618, 0.786]
}

# Sentiment Analysis Settings
SENTIMENT_SETTINGS = {
    'MIN_NEWS_SCORE': -1.0,
    'MAX_NEWS_SCORE': 1.0,
    'SOURCE_WEIGHTS': {
        'coindesk': 1.0,
        'cointelegraph': 0.9,
        'reddit': 0.7,
        'twitter': 0.6,
        'other': 0.5
    },
    'TIME_DECAY_FACTOR': 0.1  # Per day
}

# Fundamental Analysis Settings
FUNDAMENTAL_SETTINGS = {
    'MIN_GITHUB_COMMITS': 10,  # Per month
    'MIN_TEAM_SCORE': 5.0,
    'MAX_INFLATION_RATE': 10.0,  # Percentage
    'VALUATION_METRICS': ['market_cap', 'fdv', 'nvt_ratio']
}

# Output Settings
OUTPUT_SETTINGS = {
    'REPORT_FORMAT': 'json',  # json, html, pdf
    'CHART_DPI': 300,
    'CHART_STYLE': 'seaborn-v0_8',  # Updated for matplotlib 3.6+
    'INCLUDE_CHARTS': True,
    'SAVE_REPORTS': True,
    'REPORTS_DIR': str(DATA_DIR / 'reports')
}

# Create reports directory
Path(OUTPUT_SETTINGS['REPORTS_DIR']).mkdir(exist_ok=True)

# Portfolio Settings (for mock trading)
PORTFOLIO_SETTINGS = {
    'INITIAL_CAPITAL': 10000,  # USD
    'MAX_POSITION_SIZE': 0.2,  # 20% of portfolio
    'STOP_LOSS_PERCENTAGE': 0.05,  # 5%
    'TAKE_PROFIT_LEVELS': [0.1, 0.2, 0.3],  # 10%, 20%, 30%
    'MAX_OPEN_POSITIONS': 5
}

# Validation Settings
VALIDATION_SETTINGS = {
    'MIN_PRICE': 0.0,
    'MAX_PRICE_CHANGE': 0.5,  # 50% change filter for anomaly detection
    'MIN_VOLUME': 0.0,
    'REQUIRE_COMPLETE_CANDLES': True
}

# Feature Flags
FEATURES = {
    'ENABLE_MACHINE_LEARNING': True,
    'ENABLE_NEWS_SCRAPING': True,
    'ENABLE_SOCIAL_SENTIMENT': True,
    'ENABLE_ONCHAIN_ANALYSIS': True,
    'ENABLE_BACKTESTING': True,
    'ENABLE_REAL_TIME_UPDATES': False
}
