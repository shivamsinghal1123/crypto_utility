"""
Sentiment Analysis Module
Analyzes market sentiment from news and social media
"""

from typing import Dict, List
from datetime import datetime, timedelta
import logging

from config.settings import SENTIMENT_SETTINGS

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzes market sentiment from various sources."""
    
    def __init__(self):
        self.settings = SENTIMENT_SETTINGS
        self.source_weights = self.settings['SOURCE_WEIGHTS']
    
    def calculate_time_decay(self, article_date: datetime, current_date: datetime = None) -> float:
        """
        Calculate time decay factor for news sentiment.
        
        Args:
            article_date: Date of the article
            current_date: Current date (default: now)
        
        Returns:
            Decay factor between 0 and 1
        """
        if current_date is None:
            current_date = datetime.now()
        
        days_old = (current_date - article_date).days
        decay_factor = self.settings['TIME_DECAY_FACTOR']
        
        # Exponential decay
        weight = max(0, 1 - (days_old * decay_factor))
        
        return weight
    
    def analyze_news_sentiment(self, news_data: List[Dict]) -> Dict:
        """
        Analyze sentiment from news articles.
        
        Args:
            news_data: List of news articles with sentiment scores
        
        Returns:
            Dictionary with aggregated news sentiment
        """
        if not news_data:
            return {
                'overall_sentiment': 0.0,
                'sentiment_trend': 'neutral',
                'news_volume': 0,
                'weighted_sentiment': 0.0,
                'sentiment_by_source': {}
            }
        
        weighted_sentiments = []
        weights = []
        sentiment_by_source = {}
        
        for article in news_data:
            sentiment = article.get('sentiment', 0)
            source = article.get('source', 'other').lower()
            published = article.get('published', datetime.now())
            relevance = article.get('relevance', 1.0)
            credibility = article.get('credibility_score', 0.5)
            
            # Get source weight
            source_weight = self.source_weights.get(source, self.source_weights['other'])
            
            # Calculate time decay
            time_weight = self.calculate_time_decay(published)
            
            # Combined weight
            total_weight = source_weight * time_weight * relevance * credibility
            
            weighted_sentiments.append(sentiment * total_weight)
            weights.append(total_weight)
            
            # Track by source
            if source not in sentiment_by_source:
                sentiment_by_source[source] = {
                    'sentiment': 0,
                    'count': 0,
                    'weight': 0
                }
            
            sentiment_by_source[source]['sentiment'] += sentiment
            sentiment_by_source[source]['count'] += 1
            sentiment_by_source[source]['weight'] += total_weight
        
        # Calculate weighted average
        if sum(weights) > 0:
            overall_sentiment = sum(weighted_sentiments) / sum(weights)
        else:
            overall_sentiment = 0.0
        
        # Determine trend
        if overall_sentiment > 0.2:
            trend = 'positive'
        elif overall_sentiment < -0.2:
            trend = 'negative'
        else:
            trend = 'neutral'
        
        # Calculate average sentiment by source
        for source in sentiment_by_source:
            count = sentiment_by_source[source]['count']
            if count > 0:
                sentiment_by_source[source]['average'] = round(
                    sentiment_by_source[source]['sentiment'] / count, 3
                )
        
        return {
            'overall_sentiment': round(overall_sentiment, 3),
            'sentiment_trend': trend,
            'news_volume': len(news_data),
            'weighted_sentiment': round(overall_sentiment, 3),
            'sentiment_by_source': sentiment_by_source
        }
    
    def analyze_social_sentiment(self, social_data: Dict) -> Dict:
        """
        Analyze sentiment from social media.
        
        Args:
            social_data: Social media data with platform-specific sentiments
        
        Returns:
            Dictionary with aggregated social sentiment
        """
        if not social_data or 'platforms' not in social_data:
            return {
                'overall_sentiment': 0.0,
                'platform_breakdown': {},
                'total_engagement': 0
            }
        
        platforms = social_data.get('platforms', {})
        platform_breakdown = {}
        total_sentiment = 0
        platform_count = 0
        total_engagement = 0
        
        for platform, data in platforms.items():
            sentiment = data.get('overall_sentiment', 0)
            
            platform_breakdown[platform] = {
                'sentiment': round(sentiment, 3),
                'data': data
            }
            
            total_sentiment += sentiment
            platform_count += 1
            
            # Track engagement
            if platform == 'reddit':
                total_engagement += data.get('total_comments', 0)
            elif platform == 'twitter':
                total_engagement += data.get('total_engagement', 0)
        
        # Calculate average sentiment
        overall_sentiment = total_sentiment / platform_count if platform_count > 0 else 0.0
        
        return {
            'overall_sentiment': round(overall_sentiment, 3),
            'platform_breakdown': platform_breakdown,
            'total_engagement': total_engagement,
            'active_platforms': platform_count
        }
    
    def calculate_fear_greed_index(self, news_sentiment: float, social_sentiment: float,
                                   price_change: float, volatility: float) -> Dict:
        """
        Calculate a simplified Fear & Greed Index.
        
        Args:
            news_sentiment: News sentiment score (-1 to 1)
            social_sentiment: Social sentiment score (-1 to 1)
            price_change: 24h price change percentage
            volatility: Volatility level (0-1)
        
        Returns:
            Dictionary with Fear & Greed metrics
        """
        # Normalize inputs to 0-100 scale
        
        # Sentiment (40% weight)
        avg_sentiment = (news_sentiment + social_sentiment) / 2
        sentiment_score = ((avg_sentiment + 1) / 2) * 100  # Convert -1,1 to 0,100
        
        # Price momentum (30% weight)
        # Normalize price change to 0-100 (assuming max change of ±20%)
        momentum_score = min(max((price_change + 20) / 40 * 100, 0), 100)
        
        # Volatility (30% weight) - inverted (high volatility = fear)
        volatility_score = (1 - volatility) * 100
        
        # Weighted average
        fear_greed_score = (
            sentiment_score * 0.4 +
            momentum_score * 0.3 +
            volatility_score * 0.3
        )
        
        # Determine category
        if fear_greed_score >= 75:
            category = 'Extreme Greed'
        elif fear_greed_score >= 60:
            category = 'Greed'
        elif fear_greed_score >= 40:
            category = 'Neutral'
        elif fear_greed_score >= 25:
            category = 'Fear'
        else:
            category = 'Extreme Fear'
        
        return {
            'score': round(fear_greed_score, 1),
            'category': category,
            'components': {
                'sentiment': round(sentiment_score, 1),
                'momentum': round(momentum_score, 1),
                'volatility': round(volatility_score, 1)
            }
        }
    
    def analyze(self, news_data: List[Dict] = None, social_data: Dict = None,
                price_data: Dict = None, volatility_data: Dict = None) -> Dict:
        """
        Perform comprehensive sentiment analysis.
        
        Args:
            news_data: News articles data
            social_data: Social media data
            price_data: Price data for momentum
            volatility_data: Volatility metrics
        
        Returns:
            Complete sentiment analysis
        """
        logger.info("Starting sentiment analysis")
        
        # News sentiment
        news_sentiment = self.analyze_news_sentiment(news_data or [])
        
        # Social sentiment
        social_sentiment = self.analyze_social_sentiment(social_data or {})
        
        # Overall sentiment (weighted average of news and social)
        news_weight = 0.6
        social_weight = 0.4
        
        overall_sentiment = (
            news_sentiment['overall_sentiment'] * news_weight +
            social_sentiment['overall_sentiment'] * social_weight
        )
        
        # Fear & Greed Index
        fear_greed = None
        if price_data and volatility_data:
            price_change = price_data.get('price_change_percent', 0)
            
            # Normalize volatility level to 0-1
            volatility_level = volatility_data.get('volatility_level', 'medium')
            volatility_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
            volatility = volatility_map.get(volatility_level, 0.5)
            
            fear_greed = self.calculate_fear_greed_index(
                news_sentiment['overall_sentiment'],
                social_sentiment['overall_sentiment'],
                price_change,
                volatility
            )
        
        result = {
            'timestamp': datetime.now(),
            'overall_sentiment': round(overall_sentiment, 3),
            'news_sentiment': news_sentiment,
            'social_sentiment': social_sentiment
        }
        
        if fear_greed:
            result['fear_greed_index'] = fear_greed
        
        return result
