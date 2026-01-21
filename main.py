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
from prediction.trading_environment import TradingEnvironment
from prediction.rl_agent import RLAgent

from storage.database import DatabaseManager
from storage.cache import get_cache
from storage.prediction_tracker import PredictionTracker
from storage.performance_analytics import PerformanceAnalytics

from outputs.reports import ReportGenerator
from outputs.visualizations import ChartGenerator

from tests.excel_report_generator import ExcelReportGenerator

from utils.helpers import setup_logging, normalize_symbol, get_coin_id_mapping
from utils.validators import DataValidator

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class CryptoAnalyzer:
    """Main cryptocurrency analysis framework."""
    
    def __init__(self, enable_rl: bool = False):
        """
        Initialize all components.
        
        Args:
            enable_rl: Whether to enable RL-based predictions
        """
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
        
        # RL Components
        self.enable_rl = enable_rl
        self.prediction_tracker = PredictionTracker()
        self.performance_analytics = PerformanceAnalytics()
        
        if enable_rl:
            logger.info("Initializing RL components...")
            self.rl_env = TradingEnvironment(state_dim=50)
            self.rl_agent = RLAgent(
                state_dim=50,
                action_dim=3,  # up, down, neutral
                learning_rate=0.001,
                gamma=0.95
            )
            # Try to load existing RL model
            self.rl_agent.load_model()
            logger.info("RL components initialized")
        else:
            self.rl_env = None
            self.rl_agent = None
        
        # Output
        self.report_generator = ReportGenerator()
        self.chart_generator = ChartGenerator()
        self.excel_generator = ExcelReportGenerator()
        
        # Validator
        self.validator = DataValidator()
        
        logger.info(f"Crypto Analyzer Framework initialized successfully (RL: {enable_rl})")
    
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
            
            # RL-Enhanced Prediction (if enabled)
            if self.enable_rl and self.rl_agent:
                print(f"🤖 Generating RL-enhanced prediction...")
                rl_prediction = self._get_rl_prediction(collected_data, analysis_results)
                
                # Combine traditional and RL predictions
                forecast_results['rl_prediction'] = rl_prediction
                forecast_results['combined_prediction'] = self._combine_predictions(
                    forecast_results, rl_prediction
                )
                
                # Track prediction for future learning
                self._track_prediction(symbol, forecast_results, analysis_results)
            
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
        
        # Generate/Update Excel report
        print(f"📊 Updating Excel report...")
        try:
            excel_path = self.excel_generator.update_excel_with_new_report(symbol, report)
            if excel_path:
                output_files['excel_report'] = excel_path
                print(f"   ✓ Excel report updated: {excel_path}")
        except Exception as e:
            logger.warning(f"Failed to generate Excel report: {e}")
        
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
    
    def _get_rl_prediction(self, collected_data: dict, analysis_results: dict) -> dict:
        """
        Generate RL-based prediction.
        
        Args:
            collected_data: Collected market data
            analysis_results: Analysis results
            
        Returns:
            RL prediction dictionary
        """
        # Encode current state
        complete_data = {**collected_data, **analysis_results}
        state = self.rl_env.encode_state(complete_data)
        
        # Get action from RL agent
        action, confidence = self.rl_agent.select_action(state, explore=False)
        predicted_direction = self.rl_env.action_space[action]
        
        # Get current price
        current_price = collected_data.get('current_price_data', {}).get('price', 0)
        
        # Estimate price based on direction and historical volatility
        technical = analysis_results.get('technical_analysis', {})
        volatility_data = technical.get('volatility', {})
        
        # Extract numeric volatility value
        if isinstance(volatility_data, dict):
            # Try ATR first, then other metrics
            volatility = volatility_data.get('atr', 0)
            if hasattr(volatility, 'item'):  # numpy scalar
                volatility = float(volatility.item())
            elif not isinstance(volatility, (int, float)):
                volatility = 0.02  # default fallback
        else:
            volatility = 0.02
        
        # Normalize volatility to percentage if it's absolute value
        if volatility > 1:
            volatility = volatility / current_price if current_price > 0 else 0.02
        
        if predicted_direction == 'up':
            predicted_price = current_price * (1 + volatility * confidence)
        elif predicted_direction == 'down':
            predicted_price = current_price * (1 - volatility * confidence)
        else:
            predicted_price = current_price
        
        return {
            'direction': predicted_direction,
            'predicted_price': predicted_price,
            'confidence': confidence,
            'method': 'reinforcement_learning'
        }
    
    def _combine_predictions(self, traditional_forecast: dict, rl_prediction: dict) -> dict:
        """
        Combine traditional and RL predictions.
        
        Args:
            traditional_forecast: Traditional forecast results
            rl_prediction: RL prediction results
            
        Returns:
            Combined prediction
        """
        # Weight: 40% traditional, 60% RL (as RL improves over time)
        trad_weight = 0.4
        rl_weight = 0.6
        
        trad_price = traditional_forecast.get('predicted_price_24h', 0)
        rl_price = rl_prediction.get('predicted_price', 0)
        
        combined_price = trad_weight * trad_price + rl_weight * rl_price
        
        # Direction: Use RL if high confidence, otherwise traditional
        if rl_prediction['confidence'] > 0.7:
            combined_direction = rl_prediction['direction']
        else:
            combined_direction = traditional_forecast.get('direction', 'neutral')
        
        return {
            'predicted_price': combined_price,
            'direction': combined_direction,
            'confidence': (trad_weight * traditional_forecast.get('confidence', 0.5) + 
                         rl_weight * rl_prediction['confidence']),
            'method': 'hybrid_traditional_rl'
        }
    
    def _track_prediction(self, symbol: str, forecast: dict, analysis: dict):
        """
        Track prediction for future verification and learning.
        
        Args:
            symbol: Cryptocurrency symbol
            forecast: Forecast results
            analysis: Analysis results
        """
        try:
            # Extract states for tracking
            technical_state = analysis.get('technical_analysis', {})
            sentiment_state = analysis.get('sentiment_analysis', {})
            fundamental_state = analysis.get('fundamental_analysis', {})
            
            # Current market conditions
            market_conditions = {
                'rsi': technical_state.get('rsi', 50),
                'trend': technical_state.get('trend', 'neutral'),
                'sentiment': sentiment_state.get('overall_score', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Get prediction details
            if 'rl_prediction' in forecast:
                pred = forecast['rl_prediction']
            else:
                pred = forecast
            
            # Save prediction to tracker
            prediction_id = self.prediction_tracker.save_prediction(
                symbol=symbol,
                predicted_price=pred.get('predicted_price', 0),
                predicted_direction=pred.get('direction', 'neutral'),
                confidence=pred.get('confidence', 0.5),
                technical_state=technical_state,
                sentiment_state=sentiment_state,
                fundamental_state=fundamental_state,
                market_conditions=market_conditions
            )
            
            logger.info(f"Prediction {prediction_id} tracked for {symbol}")
        except Exception as e:
            logger.error(f"Failed to track prediction: {e}")
    
    def verify_predictions(self):
        """Verify past predictions and train RL agent."""
        if not self.enable_rl:
            logger.info("RL not enabled, skipping prediction verification")
            return
        
        logger.info("Verifying past predictions...")
        
        # Get unverified predictions (24h+ old)
        unverified = self.prediction_tracker.get_unverified_predictions(hours_old=24)
        
        if not unverified:
            logger.info("No predictions to verify")
            return
        
        print(f"\n🔍 Verifying {len(unverified)} predictions...")
        
        verified_count = 0
        for prediction in unverified:
            try:
                symbol = prediction['symbol']
                
                # Get current price
                current_data = self.price_collector.get_current_price(symbol)
                if not current_data:
                    continue
                
                actual_price = current_data.get('price', 0)
                predicted_price = prediction['predicted_price']
                
                # Determine actual direction
                if actual_price > predicted_price * 1.01:
                    actual_direction = 'up'
                elif actual_price < predicted_price * 0.99:
                    actual_direction = 'down'
                else:
                    actual_direction = 'neutral'
                
                # Verify prediction
                accuracy = self.prediction_tracker.verify_prediction(
                    prediction['id'],
                    actual_price,
                    actual_direction
                )
                
                # Train RL agent with this experience
                if self.rl_agent:
                    # Reconstruct state from saved data
                    state_data = {
                        'technical_analysis': prediction['technical_state'],
                        'sentiment_analysis': prediction['sentiment_state'],
                        'fundamental_analysis': prediction['fundamental_state'],
                        'price_data': prediction['market_conditions']
                    }
                    
                    state = self.rl_env.encode_state(state_data)
                    action = self.rl_env.action_space.index(prediction['predicted_direction'])
                    
                    # Calculate reward
                    reward = self.rl_env._calculate_reward(
                        prediction['predicted_direction'],
                        actual_direction,
                        prediction['confidence'],
                        predicted_price,
                        actual_price
                    )
                    
                    # Store experience
                    self.rl_agent.store_experience(state, action, reward, state, True)
                    
                    # Train
                    loss = self.rl_agent.train_step()
                    if loss:
                        logger.debug(f"RL training loss: {loss:.4f}")
                
                verified_count += 1
                
            except Exception as e:
                logger.error(f"Failed to verify prediction {prediction['id']}: {e}")
        
        # Update target network periodically
        if verified_count > 0 and verified_count % 10 == 0:
            self.rl_agent.update_target_network()
        
        # Save RL model
        if verified_count > 0:
            self.rl_agent.save_model()
        
        print(f"✅ Verified {verified_count} predictions")
        
        # Show performance stats
        stats = self.prediction_tracker.get_performance_stats()
        print(f"\n📊 Performance Stats (Last 30 days):")
        print(f"   Total Predictions: {stats['total_predictions']}")
        print(f"   Average Accuracy: {stats['average_accuracy']:.2%}")
        print(f"   Direction Accuracy: {stats['direction_accuracy']:.2%}")
    
    def analyze_symbol(self, symbol: str):
        """
        Alias for analyze_cryptocurrency for WebSocket integration.
        
        Args:
            symbol: Cryptocurrency symbol
        """
        return self.analyze_cryptocurrency(symbol)


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
        required=False,
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
    
    parser.add_argument(
        '--enable-rl',
        action='store_true',
        help='Enable RL-based predictions and learning'
    )
    
    parser.add_argument(
        '--verify-predictions',
        action='store_true',
        help='Verify past predictions and train RL model'
    )
    
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Start real-time WebSocket monitoring'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = CryptoAnalyzer(enable_rl=args.enable_rl)
    
    # Verify predictions mode
    if args.verify_predictions:
        print("\n🔍 PREDICTION VERIFICATION MODE")
        print("="*80)
        analyzer.verify_predictions()
        sys.exit(0)
    
    # Real-time monitoring mode
    if args.monitor:
        if not args.symbol:
            parser.error("--symbol is required for monitoring mode")
            
        from data_collection.websocket_client import RealTimeAnalyzer
        
        print("\n📡 REAL-TIME MONITORING MODE")
        print("="*80)
        print(f"Starting WebSocket monitoring for {args.symbol}")
        
        rt_analyzer = RealTimeAnalyzer(analyzer, rl_enabled=args.enable_rl)
        
        try:
            rt_analyzer.start_monitoring([args.symbol])
            print("\n✅ Monitoring active. Press Ctrl+C to stop.")
            
            # Keep main thread alive
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping monitoring...")
            rt_analyzer.stop_monitoring()
            sys.exit(0)
    
    # Regular analysis mode requires symbol
    if not args.symbol:
        parser.error("--symbol is required for analysis")
    
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
