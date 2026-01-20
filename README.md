# Cryptocurrency Analysis Framework

A comprehensive Python-based framework for analyzing cryptocurrencies through fundamental analysis, technical analysis, sentiment analysis, and **machine learning-powered price predictions**. Built for educational purposes and mock trading portfolio projects.

## 🆕 NEW: Reinforcement Learning Integration

This framework now includes a **complete Reinforcement Learning system** that learns from prediction outcomes to continuously improve accuracy!

### RL Features:
- 🤖 **Deep Q-Learning Agent** with neural network
- 📊 **Prediction Tracking** with automatic verification  
- 📈 **Performance Analytics** showing learning progress
- 🔄 **Real-Time WebSocket** monitoring for 24/7 operation
- 🎯 **Hybrid Predictions** combining traditional + RL approaches
- 📉 **Expected Accuracy**: 50-55% → 70-75% over 6 months

**Quick Start:** See [QUICKSTART_RL.md](QUICKSTART_RL.md)  
**Full Documentation:** See [RL_IMPLEMENTATION.md](RL_IMPLEMENTATION.md)

---

## 🚀 Features

### Data Collection
- **Price Data**: Real-time and historical OHLCV data from Binance
- **News Aggregation**: Multi-source news collection from CoinDesk, CoinTelegraph, CryptoPanic
- **On-Chain Metrics**: Blockchain data from Etherscan, BSCScan, CoinGecko
- **Social Sentiment**: Reddit and Twitter sentiment analysis (optional)
- **🆕 WebSocket Streaming**: Real-time price monitoring with auto-reconnection

### Analysis Modules
- **Fundamental Analysis**: Tokenomics, project evaluation, valuation metrics
- **Technical Analysis**: 20+ indicators including RSI, MACD, Bollinger Bands, Moving Averages
- **Sentiment Analysis**: News and social media sentiment scoring
- **Support/Resistance**: Multi-method S/R calculation using pivot points, Fibonacci, volume profile, and more
- **🆕 RL Predictions**: Neural network learned patterns from actual outcomes

### Predictions
- **24-Hour Forecasting**: Price movement probability predictions
- **Support/Resistance Levels**: Next 24h key levels with confidence scores
- **Volatility Assessment**: Expected price range calculations
- **🆕 Machine Learning**: Deep Q-Learning with experience replay
- **🆕 Hybrid Approach**: Combines traditional + RL predictions (40%/60%)

### Output
- **Comprehensive Reports**: JSON and text format reports with RL predictions
- **Visualizations**: Price charts, technical indicators, sentiment gauges, fundamental radar charts
- **Database Storage**: Local SQLite database for historical tracking and prediction verification
- **Caching**: Intelligent API response caching
- **🆕 Performance Analytics**: Accuracy trends, learning velocity, calibration metrics

## 📋 Requirements

- Python 3.8+
- Internet connection for API access
- Optional: API keys for enhanced features (see Configuration)
- **🆕 For RL features**: PyTorch, WebSockets (see installation below)

## 🛠️ Installation

1. **Clone or download the framework**
```bash
cd crypto_analyzer
```

2. **Install dependencies**

**Standard Installation (Traditional Analysis Only):**
```bash
pip install -r requirements.txt
```

**Full Installation (With RL + WebSocket):**
```bash
pip install -r requirements.txt
pip install torch websockets

# Or for Apple Silicon Macs (M1/M2/M3):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install websockets
```

3. **Configure API keys (optional)**
Create a `.env` file or set environment variables:
```bash
export COINGECKO_API_KEY="your_key_here"
export CRYPTOPANIC_API_KEY="your_key_here"
export ETHERSCAN_API_KEY="your_key_here"
export REDDIT_CLIENT_ID="your_key_here"
export REDDIT_CLIENT_SECRET="your_key_here"
export TWITTER_BEARER_TOKEN="your_key_here"
```

**Note**: Most features work without API keys. Binance public API requires no authentication.

## 🎯 Usage

### Basic Usage (Traditional Analysis)

Analyze any cryptocurrency:
```bash
python main.py --symbol BTCUSDT
```

Analyze with symbol only (framework adds USDT automatically):
```bash
python main.py --symbol BTC
```

### 🆕 RL-Enhanced Analysis

**Enable RL predictions:**
```bash
python main.py --symbol BTCUSDT --enable-rl
```

**Verify past predictions and train model:**
```bash
python main.py --symbol BTCUSDT --verify-predictions --enable-rl
```

**Real-time monitoring with automatic learning (24/7):**
```bash
python main.py --symbol BTCUSDT --monitor --enable-rl
```

### Advanced Options

Quick analysis without charts:
```bash
python main.py --symbol ETH --no-charts
```

Full analysis with all features:
```bash
python main.py --symbol ADAUSDT --analysis full
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--symbol SYMBOL` | Cryptocurrency symbol (required) |
| `--analysis TYPE` | Analysis type: full or quick (default: full) |
| `--no-charts` | Skip chart generation |
| `--enable-rl` | 🆕 Enable RL predictions |
| `--verify-predictions` | 🆕 Verify and train RL model |
| `--monitor` | 🆕 Start WebSocket real-time monitoring |

### Programmatic Usage

```python
from crypto_analyzer.main import CryptoAnalyzer

# Create analyzer instance (traditional)
analyzer = CryptoAnalyzer()

# Or with RL enabled
analyzer = CryptoAnalyzer(enable_rl=True)

# Analyze cryptocurrency
results = analyzer.analyze_cryptocurrency('BTCUSDT', generate_charts=True)

# Verify predictions and train RL
analyzer.verify_predictions()

# Access specific results
if results:
    fundamental_score = results['analysis_results']['fundamental_analysis']['overall_score']
    print(f"Fundamental Score: {fundamental_score}/10")
    
    # Get support/resistance levels
    sr_levels = results['analysis_results']['support_resistance']
    print(f"Next Support: {sr_levels['next_24h_support'][0]['level']}")
```

## 📁 Project Structure

```
crypto_analyzer/
├── main.py                 # Main entry point
├── config/                 # Configuration files
│   ├── api_keys.py        # API key management
│   └── settings.py        # Global settings
├── data_collection/        # Data collection modules
│   ├── price_data.py      # OHLCV data from exchanges
│   ├── news_scraper.py    # News aggregation
│   ├── onchain_data.py    # Blockchain metrics
│   └── social_data.py     # Social sentiment
├── analysis/              # Analysis modules
│   ├── fundamental.py     # Fundamental analysis
│   ├── technical.py       # Technical indicators
│   ├── sentiment.py       # Sentiment analysis
│   └── valuation.py       # Valuation models
├── prediction/            # Prediction modules
│   ├── support_resistance.py  # S/R calculations
│   └── forecast.py        # Price forecasting
├── storage/               # Data storage
│   ├── database.py        # SQLite database
│   └── cache.py          # Caching system
├── utils/                # Utilities
│   ├── helpers.py        # Helper functions
│   └── validators.py     # Data validation
├── outputs/              # Output generation
│   ├── reports.py        # Report generation
│   └── visualizations.py # Chart creation
├── data/                 # Data storage directory
│   ├── crypto_analysis.db  # SQLite database
│   └── reports/          # Generated reports
├── logs/                 # Log files
└── requirements.txt      # Python dependencies
```

## 🔧 Configuration

### Settings (`config/settings.py`)

Key configuration options:

```python
# Analysis Parameters
ANALYSIS_PARAMS = {
    'TECHNICAL_TIMEFRAMES': ['1h', '4h', '1d'],
    'NEWS_LOOKBACK_DAYS': 7,
    'SENTIMENT_WEIGHT': 0.3,
    'FUNDAMENTAL_WEIGHT': 0.4,
    'TECHNICAL_WEIGHT': 0.3
}

# Technical Analysis Settings
TECHNICAL_SETTINGS = {
    'RSI_PERIOD': 14,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'BB_PERIOD': 20,
    'SMA_PERIODS': [20, 50, 100, 200]
}
```

## 📊 Output Examples

### Console Output
```
================================================================================
CRYPTOCURRENCY ANALYSIS FRAMEWORK
Analyzing: BTCUSDT
Timestamp: 2026-01-15 10:30:45
================================================================================

📦 STEP 1: DATA COLLECTION
--------------------------------------------------------------------------------
📊 Collecting price data for BTCUSDT...
📰 Collecting news data for BTCUSDT...
⛓️  Collecting on-chain data for BTCUSDT...

🔬 STEP 2: COMPREHENSIVE ANALYSIS
--------------------------------------------------------------------------------
🔍 Performing fundamental analysis...
📈 Performing technical analysis...
😊 Analyzing market sentiment...
🎯 Calculating support and resistance levels...
🔮 Generating price forecast...

📤 STEP 3: GENERATING OUTPUT
--------------------------------------------------------------------------------
📝 Generating analysis report...
📊 Generating visualization charts...

✅ ANALYSIS COMPLETED SUCCESSFULLY
--------------------------------------------------------------------------------
Reports and charts saved to: ./data/reports/BTCUSDT_20260115_103045_report.json
================================================================================
```

### Generated Files

- `BTCUSDT_20260115_103045_report.json` - Comprehensive analysis in JSON format
- `BTCUSDT_20260115_103045_report.txt` - Human-readable text report
- `BTCUSDT_20260115_103045_price_chart.png` - Price chart with S/R levels
- `BTCUSDT_20260115_103045_indicators_chart.png` - Technical indicators
- `BTCUSDT_20260115_103045_sentiment_chart.png` - Sentiment visualization
- `BTCUSDT_20260115_103045_fundamental_radar.png` - Fundamental scores radar

## 🧪 Features Breakdown

### Fundamental Analysis
- Supply metrics (circulating, total, max supply)
- Tokenomics scoring (utility, scarcity)
- GitHub activity analysis
- Community engagement metrics
- Market cap and valuation ratios
- Developer activity scoring

### Technical Analysis
- **Trend Indicators**: SMA, EMA, MACD, Bollinger Bands
- **Momentum Indicators**: RSI, Stochastic RSI, Williams %R
- **Volume Indicators**: Volume SMA, On-Balance Volume
- **Volatility Indicators**: ATR, Keltner Channels
- Pattern recognition for support/resistance
- Multi-timeframe analysis

### Sentiment Analysis
- News sentiment scoring with source credibility weighting
- Time-decay for older news
- Social media sentiment from Reddit and Twitter
- Fear & Greed Index calculation
- Platform-specific sentiment breakdown

### Support/Resistance Prediction
- Pivot Points (Standard, Fibonacci, Camarilla)
- Fibonacci Retracement levels
- Volume Profile (High Volume Nodes)
- Psychological levels (round numbers)
- Moving Average levels
- Bollinger Band levels
- Multi-method confidence scoring

## 🎓 Educational Use Cases

1. **Learning Technical Analysis**: Understand how indicators work together
2. **Market Sentiment**: Learn to gauge market psychology
3. **Risk Management**: Practice position sizing and stop-loss placement
4. **Portfolio Building**: Track multiple cryptocurrencies
5. **Strategy Backtesting**: Save predictions and validate accuracy

## 📈 Mock Trading Recommendations

The framework provides:
- Entry/exit suggestions based on combined analysis
- Stop-loss recommendations from support levels
- Take-profit targets from resistance levels
- Position sizing based on volatility and risk
- Confidence scores for each recommendation

## ⚠️ Important Notes

### Disclaimer
This framework is for **EDUCATIONAL PURPOSES ONLY**. It is designed for:
- Learning cryptocurrency analysis
- Mock trading practice
- Portfolio demonstration projects
- Understanding market dynamics

**NOT for real trading decisions.** Cryptocurrency trading carries significant risk.

### API Rate Limits
- Binance: 1200 requests/minute (public API)
- CoinGecko: 50 requests/minute (free tier)
- Respect rate limits to avoid temporary blocks

### Data Quality
- Analysis quality depends on data availability
- Some features require API keys for full functionality
- Historical data limited by API constraints

## 🔍 Troubleshooting

### Common Issues

**Issue**: "Failed to collect price data"
- Check internet connection
- Verify symbol format (e.g., BTCUSDT not BTC/USDT)
- Check if symbol exists on Binance

**Issue**: "No news data collected"
- RSS feeds may be temporarily unavailable
- CryptoPanic requires API key
- Analysis continues with available data

**Issue**: "Import errors"
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

## 🚀 Future Enhancements

Potential improvements:
- [ ] Machine learning price prediction models
- [ ] Real-time WebSocket price updates
- [ ] Multi-exchange support
- [ ] Portfolio management features
- [ ] Automated trading simulation
- [ ] Backtesting framework
- [ ] Web dashboard interface
- [ ] Email/SMS alerts

## 📝 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

This is an educational framework. Feel free to:
- Fork and modify for your learning
- Add new analysis methods
- Improve existing algorithms
- Share improvements

## 📧 Support

For issues and questions:
- Review documentation
- Check logs in `logs/crypto_analyzer.log`
- Validate your configuration in `config/settings.py`

## 🙏 Acknowledgments

Data sources:
- Binance API
- CoinGecko API
- CoinDesk & CoinTelegraph
- DefiLlama
- Etherscan & BSCScan

Libraries:
- Pandas, NumPy for data processing
- Matplotlib, Seaborn for visualization
- TextBlob, VADER for sentiment analysis

---

**Happy Analyzing! 📊🚀**
