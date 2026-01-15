"""
Cryptocurrency Analysis Framework
Main entry point for the application
"""

import sys
import argparse
from datetime import datetime
import logging

# Import all modules
from config.api_keys import APIKeys
from config.settings import LOGGING_CONFIG, ANALYSIS_PARAMS

from data_collection.price_data import PriceDataCollector
from data_collection.news_scraper import NewsCollector
from data_collection.onchain_data import OnChainDataCollector
from data_collection.social_data import SocialDataCollector

from analysis.fundamental import FundamentalAnalyzer
from analysis.technical import TechnicalAnalyzer
from analysis.sentiment import SentimentAnalyzer
from analysis.valuation import ValuationAnalyzer

from prediction.support_resistance import SupportResistanceCalculator
from prediction.forecast import PriceForecaster

from storage.database import DatabaseManager
from storage.cache import get_cache

from outputs.reports import ReportGenerator
from outputs.visualizations import ChartGenerator

from utils.helpers import setup_logging, normalize_symbol, get_coin_id_mapping
from utils.validators import DataValidator

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class CryptoAnalyzer:
    """Main cryptocurrency analysis framework."""
    
    def __init__(self):
        """Initialize all components."""
        logger.info("Initializing Crypto Analyzer Framework")
        
        # API Keys
        self.api_keys = APIKeys()
        
        # Data Collectors
        self.price_collector = PriceDataCollector()
        self.news_collector = NewsCollector()
        self.onchain_collector = OnChainDataCollector(
            etherscan_api_key=self.api_keys.ETHERSCAN_API_KEY,
            bscscan_api_key=self.api_keys.BSCSCAN_API_KEY
        )
        self.social_collector = SocialDataCollector(
            reddit_client_id=self.api_keys.REDDIT_CLIENT_ID,
            reddit_client_secret=self.api_keys.REDDIT_CLIENT_SECRET,
            twitter_bearer_token=self.api_keys.TWITTER_BEARER_TOKEN
        )
        
        # Analyzers
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.valuation_analyzer = ValuationAnalyzer()
        
        # Predictors
        self.sr_calculator = SupportResistanceCalculator()
        self.price_forecaster = PriceForecaster()
        
        # Storage
        self.db = DatabaseManager()
        self.cache = get_cache()
        
        # Output
        self.report_generator = ReportGenerator()
        self.chart_generator = ChartGenerator()
        
        # Validator
        self.validator = DataValidator()
        
        logger.info("Crypto Analyzer Framework initialized successfully")
    
    def collect_data(self, symbol: str) -> dict:
        """
        Collect all data for analysis.
        
        Args:
            symbol: Cryptocurrency symbol
        
        Returns:
            Dictionary with collected data
        """
        logger.info(f"Collecting data for {symbol}")
        
        data = {}
        
        # Price Data
        print(f"📊 Collecting price data for {symbol}...")
        price_df = self.price_collector.get_ohlcv_data(symbol, interval='1h', limit=500)
        
        if price_df is not None:
            is_valid, errors = self.validator.validate_ohlcv_data(price_df)
            if not is_valid:
                logger.warning(f"Price data validation errors: {errors}")
                price_df = self.validator.sanitize_dataframe(price_df)
            
            data['price_data_df'] = price_df
            
            # Save to database
            self.db.save_price_data(symbol, price_df)
        else:
            logger.error("Failed to collect price data")
            return None
        
        # Current price and 24h stats
        current_price_data = self.price_collector.get_current_price(symbol)
        if current_price_data:
            data['current_price_data'] = current_price_data
        
        # News Data
        print(f"📰 Collecting news data for {symbol}...")
        news_data = self.news_collector.collect_news(
            symbol, 
            days_back=ANALYSIS_PARAMS['NEWS_LOOKBACK_DAYS'],
            api_key=self.api_keys.CRYPTOPANIC_API_KEY
        )
        
        if news_data:
            is_valid, errors = self.validator.validate_news_data(news_data)
            if is_valid:
                data['news_data'] = news_data
                self.db.save_news_data(symbol, news_data)
            else:
                logger.warning(f"News data validation errors: {errors}")
        
        # On-chain Data
        print(f"⛓️  Collecting on-chain data for {symbol}...")
        coin_id = get_coin_id_mapping(symbol)
        onchain_data = self.onchain_collector.get_onchain_metrics(symbol, coin_id=coin_id)
        
        if onchain_data:
            # Calculate network metrics
            network_metrics = self.onchain_collector.calculate_network_metrics(onchain_data)
            onchain_data.update(network_metrics)
            data['onchain_data'] = onchain_data
        
        # Social Data (optional, may not have API keys)
        if self.api_keys.REDDIT_CLIENT_ID or self.api_keys.TWITTER_BEARER_TOKEN:
            print(f"💬 Collecting social media data for {symbol}...")
            social_data = self.social_collector.collect_social_data(symbol)
            if social_data:
                data['social_data'] = social_data
        
        logger.info(f"Data collection completed for {symbol}")
        return data
    
    def perform_analysis(self, symbol: str, collected_data: dict) -> dict:
        """
        Perform comprehensive analysis.
        
        Args:
            symbol: Cryptocurrency symbol
            collected_data: Data collected from various sources
        
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Performing analysis for {symbol}")
        
        analysis_results = {}
        
        # Fundamental Analysis
        print(f"🔍 Performing fundamental analysis...")
        fundamental_results = self.fundamental_analyzer.analyze(
            symbol,
            collected_data.get('current_price_data', {}),
            collected_data.get('onchain_data')
        )
        analysis_results['fundamental_analysis'] = fundamental_results
        self.db.save_analysis_results(symbol, 'fundamental', fundamental_results)
        
        # Technical Analysis
        print(f"📈 Performing technical analysis...")
        price_df = collected_data.get('price_data_df')
        if price_df is not None and not price_df.empty:
            technical_results = self.technical_analyzer.analyze(price_df)
            analysis_results['technical_analysis'] = technical_results
            self.db.save_analysis_results(symbol, 'technical', technical_results)
        
        # Sentiment Analysis
        print(f"😊 Analyzing market sentiment...")
        sentiment_results = self.sentiment_analyzer.analyze(
            news_data=collected_data.get('news_data'),
            social_data=collected_data.get('social_data'),
            price_data=collected_data.get('current_price_data'),
            volatility_data=technical_results.get('volatility') if 'technical_results' in locals() else None
        )
        analysis_results['sentiment_analysis'] = sentiment_results
        self.db.save_analysis_results(symbol, 'sentiment', sentiment_results)
        
        # Support/Resistance Calculation
        print(f"🎯 Calculating support and resistance levels...")
        if price_df is not None and not price_df.empty:
            sr_results = self.sr_calculator.calculate_24h_levels(price_df)
            analysis_results['support_resistance'] = sr_results
            self.db.save_analysis_results(symbol, 'support_resistance', sr_results)
        
        # Price Forecast
        print(f"🔮 Generating price forecast...")
        if price_df is not None and not price_df.empty:
            forecast_results = self.price_forecaster.generate_forecast(
                price_df,
                technical_results if 'technical_results' in locals() else {},
                sentiment_results
            )
            analysis_results['predictions'] = forecast_results
            self.db.save_analysis_results(symbol, 'forecast', forecast_results)
        
        logger.info(f"Analysis completed for {symbol}")
        return analysis_results
    
    def generate_output(self, symbol: str, collected_data: dict, 
                       analysis_results: dict, generate_charts: bool = True) -> dict:
        """
        Generate reports and visualizations.
        
        Args:
            symbol: Cryptocurrency symbol
            collected_data: Collected data
            analysis_results: Analysis results
            generate_charts: Whether to generate charts
        
        Returns:
            Dictionary with output file paths
        """
        logger.info(f"Generating output for {symbol}")
        
        output_files = {}
        
        # Combine all data for report
        complete_data = {
            **collected_data,
            **analysis_results
        }
        
        # Generate comprehensive report
        print(f"📝 Generating analysis report...")
        report = self.report_generator.generate_comprehensive_report(symbol, complete_data)
        
        # Set the session folder for chart generator
        self.chart_generator.set_session_folder(self.report_generator.session_folder)
        
        # Save report
        json_report_path = self.report_generator.save_report(symbol, report, format='json')
        txt_report_path = self.report_generator.save_report(symbol, report, format='txt')
        
        output_files['json_report'] = json_report_path
        output_files['txt_report'] = txt_report_path
        
        # Display results
        self.report_generator.display_results(report)
        
        # Generate charts
        if generate_charts and 'price_data_df' in collected_data:
            print(f"📊 Generating visualization charts...")
            
            price_df = collected_data['price_data_df']
            
            # Price chart with S/R levels
            price_chart = self.chart_generator.create_price_chart(
                price_df,
                symbol,
                sr_levels=analysis_results.get('support_resistance')
            )
            if price_chart:
                output_files['price_chart'] = price_chart
            
            # Technical indicators chart
            if 'technical_analysis' in analysis_results:
                tech_chart = self.chart_generator.create_technical_indicators_chart(
                    price_df,
                    analysis_results['technical_analysis'].get('indicators', {}),
                    symbol
                )
                if tech_chart:
                    output_files['technical_chart'] = tech_chart
            
            # Sentiment chart
            if 'sentiment_analysis' in analysis_results:
                sentiment_chart = self.chart_generator.create_sentiment_chart(
                    analysis_results['sentiment_analysis'],
                    symbol
                )
                if sentiment_chart:
                    output_files['sentiment_chart'] = sentiment_chart
            
            # Fundamental radar chart
            if 'fundamental_analysis' in analysis_results:
                radar_chart = self.chart_generator.create_fundamental_radar(
                    analysis_results['fundamental_analysis'],
                    symbol
                )
                if radar_chart:
                    output_files['fundamental_radar'] = radar_chart
        
        logger.info(f"Output generation completed for {symbol}")
        return output_files
    
    def analyze_cryptocurrency(self, symbol: str, generate_charts: bool = True):
        """
        Main analysis workflow.
        
        Args:
            symbol: Cryptocurrency symbol
            generate_charts: Whether to generate visualization charts
        
        Returns:
            Complete analysis report
        """
        print("\n" + "="*80)
        print(f"CRYPTOCURRENCY ANALYSIS FRAMEWORK")
        print(f"Analyzing: {symbol}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
        # Normalize symbol
        symbol = normalize_symbol(symbol)
        logger.info(f"Starting comprehensive analysis for {symbol}")
        
        try:
            # Step 1: Data Collection
            print("\n📦 STEP 1: DATA COLLECTION")
            print("-" * 80)
            collected_data = self.collect_data(symbol)
            
            if not collected_data or 'price_data_df' not in collected_data:
                logger.error("Failed to collect required data")
                print("\n❌ Analysis failed: Unable to collect required price data")
                return None
            
            # Step 2: Analysis
            print("\n🔬 STEP 2: COMPREHENSIVE ANALYSIS")
            print("-" * 80)
            analysis_results = self.perform_analysis(symbol, collected_data)
            
            # Step 3: Generate Output
            print("\n📤 STEP 3: GENERATING OUTPUT")
            print("-" * 80)
            output_files = self.generate_output(
                symbol, 
                collected_data, 
                analysis_results,
                generate_charts
            )
            
            # Summary
            print("\n✅ ANALYSIS COMPLETED SUCCESSFULLY")
            print("-" * 80)
            print(f"All artifacts saved to: {self.report_generator.session_folder}")
            print("="*80 + "\n")
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'collected_data': collected_data,
                'analysis_results': analysis_results,
                'output_files': output_files
            }
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}", exc_info=True)
            print(f"\n❌ Analysis failed: {e}")
            return None


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Cryptocurrency Analysis Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --symbol BTCUSDT
  python main.py --symbol ETH --no-charts
  python main.py --symbol ADAUSDT --analysis full
        """
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Cryptocurrency symbol (e.g., BTCUSDT, ETH, ADA)'
    )
    
    parser.add_argument(
        '--analysis',
        type=str,
        choices=['full', 'quick'],
        default='full',
        help='Analysis type (default: full)'
    )
    
    parser.add_argument(
        '--no-charts',
        action='store_true',
        help='Skip chart generation'
    )
    
    parser.add_argument(
        '--export-report',
        action='store_true',
        help='Export report (always enabled)'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = CryptoAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_cryptocurrency(
        symbol=args.symbol,
        generate_charts=not args.no_charts
    )
    
    if results:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
