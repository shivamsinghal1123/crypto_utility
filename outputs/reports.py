"""
Reports Module
Generates analysis reports in various formats
"""

import json
from typing import Dict, List
from datetime import datetime
from pathlib import Path
import logging

from config.settings import OUTPUT_SETTINGS
from utils.helpers import format_number, format_percentage, format_large_number

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive analysis reports."""
    
    def __init__(self):
        self.settings = OUTPUT_SETTINGS
        self.reports_dir = Path(self.settings['REPORTS_DIR'])
        self.reports_dir.mkdir(exist_ok=True, parents=True)
        self.session_folder = None  # Will be set when generating report
    
    def generate_comprehensive_report(self, symbol: str, analysis_data: Dict) -> Dict:
        """
        Generate comprehensive analysis report.
        
        Args:
            symbol: Cryptocurrency symbol
            analysis_data: Complete analysis data
        
        Returns:
            Report dictionary
        """
        # Create session folder for this execution
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_folder = self.reports_dir / f"{symbol}_{timestamp}"
        self.session_folder.mkdir(exist_ok=True, parents=True)
        logger.info(f"Created session folder: {self.session_folder}")
        
        # Create README file in the session folder
        readme_path = self.session_folder / "README.txt"
        with open(readme_path, 'w') as f:
            f.write(f"Cryptocurrency Analysis Session\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session Folder: {self.session_folder.name}\n\n")
            f.write(f"Contents:\n")
            f.write(f"  - report.json: Comprehensive analysis report (JSON format)\n")
            f.write(f"  - report.txt: Human-readable analysis report\n")
            f.write(f"  - price_chart.png: Price chart with support/resistance levels\n")
            f.write(f"  - indicators_chart.png: Technical indicators visualization\n")
            f.write(f"  - sentiment_chart.png: Sentiment analysis visualization\n")
            f.write(f"  - fundamental_radar.png: Fundamental analysis radar chart\n")
            f.write(f"  - README.txt: This file\n")
        
        report = {
            'metadata': {
                'symbol': symbol,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_freshness': self._assess_data_freshness(analysis_data),
                'confidence_score': self._calculate_overall_confidence(analysis_data)
            },
            'fundamental_analysis': self._format_fundamental_analysis(
                analysis_data.get('fundamental_analysis', {})
            ),
            'technical_analysis': self._format_technical_analysis(
                analysis_data.get('technical_analysis', {})
            ),
            'sentiment_analysis': self._format_sentiment_analysis(
                analysis_data.get('sentiment_analysis', {})
            ),
            'predictions': self._format_predictions(
                analysis_data.get('predictions', {}),
                analysis_data.get('support_resistance', {})
            ),
            'risk_assessment': self._assess_risk(analysis_data),
            'recommendations': self._generate_recommendations(analysis_data)
        }
        
        logger.info(f"Generated comprehensive report for {symbol}")
        return report
    
    def _assess_data_freshness(self, analysis_data: Dict) -> Dict:
        """Assess freshness of data sources."""
        freshness = {}
        
        if 'price_data' in analysis_data:
            freshness['price_data'] = 'fresh'  # Assuming real-time
        
        if 'news_data' in analysis_data:
            news_data = analysis_data['news_data']
            if news_data and len(news_data) > 0:
                latest_news = max(n.get('published', datetime.min) for n in news_data)
                age_hours = (datetime.now() - latest_news).total_seconds() / 3600
                freshness['news_data'] = f"{age_hours:.1f} hours old"
            else:
                freshness['news_data'] = 'no data'
        
        return freshness
    
    def _calculate_overall_confidence(self, analysis_data: Dict) -> float:
        """Calculate overall confidence score."""
        scores = []
        
        # Prediction confidence
        predictions = analysis_data.get('predictions', {})
        if predictions and 'probabilities' in predictions:
            scores.append(predictions['probabilities'].get('confidence', 0))
        
        # Support/Resistance confidence
        sr = analysis_data.get('support_resistance', {})
        if sr:
            scores.append(sr.get('confidence_score', 0) / 100)
        
        # Data availability score
        if analysis_data.get('fundamental_analysis'):
            scores.append(0.8)
        if analysis_data.get('sentiment_analysis'):
            scores.append(0.7)
        
        return round(sum(scores) / len(scores) if scores else 0.5, 2)
    
    def _format_fundamental_analysis(self, fundamental_data: Dict) -> Dict:
        """Format fundamental analysis for report."""
        if not fundamental_data:
            return {}
        
        tokenomics = fundamental_data.get('tokenomics', {})
        fundamentals = fundamental_data.get('fundamentals', {})
        valuation = fundamental_data.get('valuation', {})
        
        return {
            'overall_score': fundamental_data.get('overall_score', 0),
            'tokenomics_score': tokenomics.get('utility_score', 0),
            'team_score': fundamentals.get('team_score', 0),
            'technology_score': fundamentals.get('technology_score', 0),
            'adoption_score': fundamentals.get('community_score', 0),
            'detailed_metrics': {
                'market_cap': format_large_number(valuation.get('market_cap', 0)),
                'supply_metrics': tokenomics.get('supply_metrics', {}),
                'valuation_ratios': {
                    'nvt_ratio': valuation.get('nvt_ratio', 'N/A')
                }
            }
        }
    
    def _format_technical_analysis(self, technical_data: Dict) -> Dict:
        """Format technical analysis for report."""
        if not technical_data:
            return {}
        
        trend = technical_data.get('trend', {})
        momentum = technical_data.get('momentum', {})
        volatility = technical_data.get('volatility', {})
        
        return {
            'trend_direction': trend.get('direction', 'unknown'),
            'trend_strength': trend.get('strength', 0),
            'momentum_score': momentum.get('momentum_score', 0),
            'volatility_assessment': volatility.get('volatility_level', 'unknown'),
            'key_levels': technical_data.get('support_resistance', {}),
            'indicators': {
                'rsi': momentum.get('rsi', {}),
                'macd': momentum.get('macd', {})
            }
        }
    
    def _format_sentiment_analysis(self, sentiment_data: Dict) -> Dict:
        """Format sentiment analysis for report."""
        if not sentiment_data:
            return {}
        
        news_sentiment = sentiment_data.get('news_sentiment', {})
        social_sentiment = sentiment_data.get('social_sentiment', {})
        
        return {
            'overall_sentiment': sentiment_data.get('overall_sentiment', 0),
            'news_sentiment': news_sentiment.get('overall_sentiment', 0),
            'social_sentiment': social_sentiment.get('overall_sentiment', 0),
            'market_fear_greed': sentiment_data.get('fear_greed_index', {})
        }
    
    def _format_predictions(self, predictions: Dict, sr_levels: Dict) -> Dict:
        """Format predictions for report."""
        return {
            'next_24h_support_levels': sr_levels.get('next_24h_support', []),
            'next_24h_resistance_levels': sr_levels.get('next_24h_resistance', []),
            'probability_up': predictions.get('probabilities', {}).get('probability_up', 0),
            'probability_down': predictions.get('probabilities', {}).get('probability_down', 0),
            'expected_volatility': predictions.get('expected_volatility', 0),
            'price_forecast': predictions.get('trend_forecast', {})
        }
    
    def _assess_risk(self, analysis_data: Dict) -> Dict:
        """Assess overall risk."""
        risk_factors = []
        
        # Volatility risk
        technical = analysis_data.get('technical_analysis', {})
        volatility = technical.get('volatility', {}).get('volatility_level', 'medium')
        
        if volatility == 'high':
            risk_factors.append('High volatility')
        
        # Sentiment risk
        sentiment = analysis_data.get('sentiment_analysis', {})
        overall_sentiment = sentiment.get('overall_sentiment', 0)
        
        if overall_sentiment < -0.3:
            risk_factors.append('Negative sentiment')
        
        # Determine overall risk
        if len(risk_factors) >= 2:
            overall_risk = 'high'
        elif len(risk_factors) == 1:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'
        
        return {
            'overall_risk': overall_risk,
            'liquidity_risk': 'medium',  # Placeholder
            'volatility_risk': volatility,
            'fundamental_risk': 'medium',  # Placeholder
            'risk_factors': risk_factors
        }
    
    def _generate_recommendations(self, analysis_data: Dict) -> Dict:
        """Generate trading recommendations."""
        # Simplified recommendation logic
        technical = analysis_data.get('technical_analysis', {})
        sentiment = analysis_data.get('sentiment_analysis', {})
        predictions = analysis_data.get('predictions', {})
        
        trend = technical.get('trend', {}).get('direction', 'neutral')
        sentiment_score = sentiment.get('overall_sentiment', 0)
        prob_up = predictions.get('probabilities', {}).get('probability_up', 0.5)
        
        # Calculate signals
        bullish_signals = 0
        bearish_signals = 0
        
        if trend == 'bullish':
            bullish_signals += 1
        elif trend == 'bearish':
            bearish_signals += 1
        
        if sentiment_score > 0.2:
            bullish_signals += 1
        elif sentiment_score < -0.2:
            bearish_signals += 1
        
        if prob_up > 0.6:
            bullish_signals += 1
        elif prob_up < 0.4:
            bearish_signals += 1
        
        # Generate recommendation
        if bullish_signals >= 2:
            suggestion = 'Consider long position'
            position_sizing = 'moderate'
        elif bearish_signals >= 2:
            suggestion = 'Consider short position or avoid'
            position_sizing = 'small'
        else:
            suggestion = 'Hold or wait for clearer signals'
            position_sizing = 'minimal'
        
        # Get support/resistance for stop loss/take profit
        sr = analysis_data.get('support_resistance', {})
        support_levels = sr.get('next_24h_support', [])
        resistance_levels = sr.get('next_24h_resistance', [])
        
        stop_loss = support_levels[0]['level'] if support_levels else None
        take_profit = [r['level'] for r in resistance_levels[:3]] if resistance_levels else []
        
        return {
            'mock_trading_suggestion': suggestion,
            'position_sizing': position_sizing,
            'stop_loss_suggestion': stop_loss,
            'take_profit_levels': take_profit,
            'confidence': round((bullish_signals + bearish_signals) / 6, 2)
        }
    
    def save_report(self, symbol: str, report: Dict, format: str = 'json') -> str:
        """
        Save report to file.
        
        Args:
            symbol: Cryptocurrency symbol
            report: Report dictionary
            format: Output format ('json', 'txt')
        
        Returns:
            Path to saved file
        """
        # Save inside the session folder
        filename = f"report.{format}"
        filepath = self.session_folder / filename
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format == 'txt':
            with open(filepath, 'w') as f:
                f.write(self._format_text_report(report))
        
        logger.info(f"Report saved to {filepath}")
        return str(filepath)
    
    def _format_text_report(self, report: Dict) -> str:
        """Format report as text."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"CRYPTOCURRENCY ANALYSIS REPORT")
        lines.append(f"Symbol: {report['metadata']['symbol']}")
        lines.append(f"Generated: {report['metadata']['analysis_timestamp']}")
        lines.append(f"Overall Confidence: {report['metadata']['confidence_score']}")
        lines.append("=" * 80)
        lines.append("")
        
        # Fundamental Analysis
        lines.append("FUNDAMENTAL ANALYSIS")
        lines.append("-" * 80)
        fund = report.get('fundamental_analysis', {})
        lines.append(f"Overall Score: {fund.get('overall_score', 'N/A')}/10")
        lines.append(f"Tokenomics Score: {fund.get('tokenomics_score', 'N/A')}/10")
        lines.append(f"Technology Score: {fund.get('technology_score', 'N/A')}/10")
        lines.append("")
        
        # Technical Analysis
        lines.append("TECHNICAL ANALYSIS")
        lines.append("-" * 80)
        tech = report.get('technical_analysis', {})
        lines.append(f"Trend: {tech.get('trend_direction', 'N/A').upper()}")
        lines.append(f"Momentum Score: {tech.get('momentum_score', 'N/A')}")
        lines.append(f"Volatility: {tech.get('volatility_assessment', 'N/A').upper()}")
        lines.append("")
        
        # Predictions
        lines.append("PREDICTIONS (24H)")
        lines.append("-" * 80)
        pred = report.get('predictions', {})
        lines.append(f"Probability Up: {format_percentage(pred.get('probability_up', 0) * 100)}")
        lines.append(f"Probability Down: {format_percentage(pred.get('probability_down', 0) * 100)}")
        lines.append("")
        
        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 80)
        rec = report.get('recommendations', {})
        lines.append(f"Suggestion: {rec.get('mock_trading_suggestion', 'N/A')}")
        lines.append(f"Position Size: {rec.get('position_sizing', 'N/A').upper()}")
        if rec.get('stop_loss_suggestion'):
            lines.append(f"Stop Loss: {rec['stop_loss_suggestion']}")
        lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def display_results(self, report: Dict):
        """Display results to console."""
        print("\n" + "=" * 80)
        print(f"ANALYSIS RESULTS FOR {report['metadata']['symbol']}")
        print("=" * 80)
        print(f"\nOverall Confidence Score: {report['metadata']['confidence_score']}")
        
        # Display key metrics
        fund = report.get('fundamental_analysis', {})
        print(f"\nFundamental Score: {fund.get('overall_score', 'N/A')}/10")
        
        tech = report.get('technical_analysis', {})
        print(f"Technical Trend: {tech.get('trend_direction', 'N/A').upper()}")
        
        sentiment = report.get('sentiment_analysis', {})
        print(f"Overall Sentiment: {sentiment.get('overall_sentiment', 0):.2f}")
        
        pred = report.get('predictions', {})
        print(f"\nProbability Up (24h): {pred.get('probability_up', 0):.1%}")
        print(f"Probability Down (24h): {pred.get('probability_down', 0):.1%}")
        
        rec = report.get('recommendations', {})
        print(f"\nRecommendation: {rec.get('mock_trading_suggestion', 'N/A')}")
        
        print("=" * 80 + "\n")
