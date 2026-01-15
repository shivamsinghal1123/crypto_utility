"""
News Scraper Module
Collects and aggregates cryptocurrency news from various sources
"""

import requests
import feedparser
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from textblob import TextBlob

from config.settings import API_ENDPOINTS, REQUEST_CONFIG
from utils.helpers import handle_api_errors

logger = logging.getLogger(__name__)


class NewsCollector:
    """Collects cryptocurrency news from multiple sources."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': REQUEST_CONFIG['USER_AGENT']})
        self.cryptopanic_base = API_ENDPOINTS.get('CRYPTOPANIC_BASE', '')
    
    @handle_api_errors
    def collect_rss_news(self, rss_url: str, days_back: int = 7) -> List[Dict]:
        """
        Collect news from RSS feeds.
        
        Args:
            rss_url: URL of the RSS feed
            days_back: Number of days to look back
        
        Returns:
            List of news articles
        """
        feed = feedparser.parse(rss_url)
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for entry in feed.entries:
            try:
                # Parse publication date
                pub_date = datetime(*entry.published_parsed[:6])
                
                if pub_date < cutoff_date:
                    continue
                
                article = {
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', ''),
                    'link': entry.get('link', ''),
                    'published': pub_date,
                    'source': feed.feed.get('title', 'Unknown')
                }
                
                articles.append(article)
            except Exception as e:
                logger.warning(f"Error parsing RSS entry: {e}")
                continue
        
        return articles
    
    @handle_api_errors
    def collect_coindesk_news(self, days_back: int = 7) -> List[Dict]:
        """Collect news from CoinDesk RSS feed."""
        rss_url = API_ENDPOINTS.get('COINDESK_RSS', 'https://www.coindesk.com/arc/outboundfeeds/rss/')
        articles = self.collect_rss_news(rss_url, days_back)
        
        for article in articles:
            article['source'] = 'CoinDesk'
            article['credibility_score'] = 1.0
        
        return articles
    
    @handle_api_errors
    def collect_cointelegraph_news(self, days_back: int = 7) -> List[Dict]:
        """Collect news from CoinTelegraph RSS feed."""
        rss_url = API_ENDPOINTS.get('COINTELEGRAPH_RSS', 'https://cointelegraph.com/rss')
        articles = self.collect_rss_news(rss_url, days_back)
        
        for article in articles:
            article['source'] = 'CoinTelegraph'
            article['credibility_score'] = 0.9
        
        return articles
    
    @handle_api_errors
    def collect_cryptopanic_news(self, symbol: str, api_key: str = None, days_back: int = 7) -> List[Dict]:
        """
        Collect news from CryptoPanic API.
        
        Args:
            symbol: Cryptocurrency symbol
            api_key: CryptoPanic API key
            days_back: Number of days to look back
        
        Returns:
            List of news articles
        """
        if not api_key:
            logger.warning("CryptoPanic API key not provided")
            return []
        
        url = f"{self.cryptopanic_base}posts/"
        params = {
            'auth_token': api_key,
            'currencies': symbol.replace('USDT', '').replace('USD', ''),
            'kind': 'news'
        }
        
        response = self.session.get(url, params=params, timeout=REQUEST_CONFIG['TIMEOUT'])
        response.raise_for_status()
        
        data = response.json()
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for post in data.get('results', []):
            try:
                pub_date = datetime.fromisoformat(post['published_at'].replace('Z', '+00:00'))
                
                if pub_date.replace(tzinfo=None) < cutoff_date:
                    continue
                
                article = {
                    'title': post.get('title', ''),
                    'summary': post.get('title', ''),  # CryptoPanic doesn't provide summary
                    'link': post.get('url', ''),
                    'published': pub_date.replace(tzinfo=None),
                    'source': post.get('source', {}).get('title', 'CryptoPanic'),
                    'credibility_score': 0.8,
                    'votes': post.get('votes', {})
                }
                
                articles.append(article)
            except Exception as e:
                logger.warning(f"Error parsing CryptoPanic post: {e}")
                continue
        
        return articles
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using TextBlob.
        
        Args:
            text: Text to analyze
        
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception as e:
            logger.warning(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def calculate_relevance(self, article: Dict, symbol: str) -> float:
        """
        Calculate relevance of article to specific cryptocurrency.
        
        Args:
            article: Article dictionary
            symbol: Cryptocurrency symbol
        
        Returns:
            Relevance score between 0 and 1
        """
        # Extract coin name from symbol
        coin_name = symbol.replace('USDT', '').replace('USD', '').lower()
        
        # Common cryptocurrency names mapping
        name_mapping = {
            'btc': ['bitcoin', 'btc'],
            'eth': ['ethereum', 'eth', 'ether'],
            'bnb': ['binance', 'bnb'],
            'ada': ['cardano', 'ada'],
            'sol': ['solana', 'sol'],
            'xrp': ['ripple', 'xrp'],
            'dot': ['polkadot', 'dot'],
            'doge': ['dogecoin', 'doge'],
            'avax': ['avalanche', 'avax'],
            'matic': ['polygon', 'matic']
        }
        
        search_terms = name_mapping.get(coin_name, [coin_name])
        
        text = f"{article.get('title', '')} {article.get('summary', '')}".lower()
        
        # Count mentions
        mentions = sum(text.count(term) for term in search_terms)
        
        # Calculate relevance score
        if mentions == 0:
            return 0.0
        elif mentions >= 3:
            return 1.0
        else:
            return mentions / 3.0
    
    def collect_news(self, symbol: str, days_back: int = 7, api_key: str = None) -> List[Dict]:
        """
        Collect news from all sources for a specific cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            days_back: Number of days to look back
            api_key: CryptoPanic API key (optional)
        
        Returns:
            List of news articles with sentiment and relevance scores
        """
        all_articles = []
        
        # Collect from various sources
        sources = [
            self.collect_coindesk_news(days_back),
            self.collect_cointelegraph_news(days_back)
        ]
        
        if api_key:
            sources.append(self.collect_cryptopanic_news(symbol, api_key, days_back))
        
        # Combine all articles
        for articles in sources:
            if articles:
                all_articles.extend(articles)
        
        # Process articles
        processed_articles = []
        for article in all_articles:
            # Calculate sentiment
            text = f"{article.get('title', '')} {article.get('summary', '')}"
            article['sentiment'] = self.analyze_sentiment(text)
            
            # Calculate relevance
            article['relevance'] = self.calculate_relevance(article, symbol)
            
            # Only include relevant articles
            if article['relevance'] > 0.2:
                processed_articles.append(article)
        
        # Sort by publication date (newest first)
        processed_articles.sort(key=lambda x: x['published'], reverse=True)
        
        logger.info(f"Collected {len(processed_articles)} relevant articles for {symbol}")
        return processed_articles
    
    def aggregate_news_sentiment(self, articles: List[Dict]) -> Dict:
        """
        Aggregate sentiment from multiple news articles.
        
        Args:
            articles: List of news articles
        
        Returns:
            Dictionary with aggregated sentiment metrics
        """
        if not articles:
            return {
                'overall_sentiment': 0.0,
                'sentiment_trend': 'neutral',
                'news_volume': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        sentiments = [a['sentiment'] for a in articles]
        relevance_weights = [a.get('relevance', 1.0) for a in articles]
        credibility_weights = [a.get('credibility_score', 0.5) for a in articles]
        
        # Weighted average sentiment
        weights = [r * c for r, c in zip(relevance_weights, credibility_weights)]
        weighted_sentiment = sum(s * w for s, w in zip(sentiments, weights)) / sum(weights)
        
        # Count sentiment categories
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        # Determine trend
        if weighted_sentiment > 0.2:
            trend = 'positive'
        elif weighted_sentiment < -0.2:
            trend = 'negative'
        else:
            trend = 'neutral'
        
        return {
            'overall_sentiment': round(weighted_sentiment, 3),
            'sentiment_trend': trend,
            'news_volume': len(articles),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'average_relevance': round(sum(relevance_weights) / len(relevance_weights), 3)
        }
