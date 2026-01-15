"""
Social Data Collection Module
Collects social media sentiment and engagement data
"""

import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

from config.settings import API_ENDPOINTS, REQUEST_CONFIG
from utils.helpers import handle_api_errors

logger = logging.getLogger(__name__)


class SocialDataCollector:
    """Collects social media data and sentiment from various platforms."""
    
    def __init__(self, reddit_client_id: str = None, reddit_client_secret: str = None,
                 twitter_bearer_token: str = None):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': REQUEST_CONFIG['USER_AGENT']})
        self.reddit_client_id = reddit_client_id
        self.reddit_client_secret = reddit_client_secret
        self.twitter_bearer_token = twitter_bearer_token
        self.reddit_token = None
    
    def authenticate_reddit(self) -> bool:
        """Authenticate with Reddit API."""
        if not self.reddit_client_id or not self.reddit_client_secret:
            logger.warning("Reddit credentials not provided")
            return False
        
        try:
            auth = requests.auth.HTTPBasicAuth(self.reddit_client_id, self.reddit_client_secret)
            data = {'grant_type': 'client_credentials'}
            headers = {'User-Agent': REQUEST_CONFIG['USER_AGENT']}
            
            response = requests.post(
                'https://www.reddit.com/api/v1/access_token',
                auth=auth,
                data=data,
                headers=headers,
                timeout=REQUEST_CONFIG['TIMEOUT']
            )
            response.raise_for_status()
            
            self.reddit_token = response.json()['access_token']
            return True
        except Exception as e:
            logger.error(f"Reddit authentication failed: {e}")
            return False
    
    @handle_api_errors
    def get_reddit_posts(self, subreddit: str, limit: int = 100, time_filter: str = 'week') -> Optional[List[Dict]]:
        """
        Get posts from a subreddit.
        
        Args:
            subreddit: Subreddit name (e.g., 'cryptocurrency', 'bitcoin')
            limit: Number of posts to retrieve
            time_filter: Time filter ('hour', 'day', 'week', 'month', 'year', 'all')
        
        Returns:
            List of post dictionaries or None if failed
        """
        if not self.reddit_token and not self.authenticate_reddit():
            return None
        
        headers = {
            'Authorization': f'bearer {self.reddit_token}',
            'User-Agent': REQUEST_CONFIG['USER_AGENT']
        }
        
        url = f'https://oauth.reddit.com/r/{subreddit}/top'
        params = {
            'limit': limit,
            't': time_filter
        }
        
        response = self.session.get(
            url,
            headers=headers,
            params=params,
            timeout=REQUEST_CONFIG['TIMEOUT']
        )
        response.raise_for_status()
        
        data = response.json()
        posts = []
        
        for post in data.get('data', {}).get('children', []):
            post_data = post['data']
            posts.append({
                'title': post_data.get('title'),
                'text': post_data.get('selftext', ''),
                'score': post_data.get('score', 0),
                'upvote_ratio': post_data.get('upvote_ratio', 0),
                'num_comments': post_data.get('num_comments', 0),
                'created_utc': datetime.fromtimestamp(post_data.get('created_utc', 0)),
                'author': post_data.get('author'),
                'url': post_data.get('url')
            })
        
        return posts
    
    @handle_api_errors
    def get_twitter_mentions(self, query: str, max_results: int = 100) -> Optional[List[Dict]]:
        """
        Get Twitter mentions for a cryptocurrency.
        
        Args:
            query: Search query
            max_results: Maximum number of results (10-100)
        
        Returns:
            List of tweet dictionaries or None if failed
        """
        if not self.twitter_bearer_token:
            logger.warning("Twitter bearer token not provided")
            return None
        
        headers = {
            'Authorization': f'Bearer {self.twitter_bearer_token}'
        }
        
        url = 'https://api.twitter.com/2/tweets/search/recent'
        params = {
            'query': query,
            'max_results': min(max_results, 100),
            'tweet.fields': 'created_at,public_metrics,text'
        }
        
        response = self.session.get(
            url,
            headers=headers,
            params=params,
            timeout=REQUEST_CONFIG['TIMEOUT']
        )
        response.raise_for_status()
        
        data = response.json()
        tweets = []
        
        for tweet in data.get('data', []):
            metrics = tweet.get('public_metrics', {})
            tweets.append({
                'text': tweet.get('text'),
                'created_at': datetime.fromisoformat(tweet.get('created_at', '').replace('Z', '+00:00')),
                'retweet_count': metrics.get('retweet_count', 0),
                'reply_count': metrics.get('reply_count', 0),
                'like_count': metrics.get('like_count', 0),
                'quote_count': metrics.get('quote_count', 0)
            })
        
        return tweets
    
    def calculate_reddit_sentiment(self, posts: List[Dict], symbol: str) -> Dict:
        """
        Calculate sentiment from Reddit posts.
        
        Args:
            posts: List of Reddit posts
            symbol: Cryptocurrency symbol to filter for
        
        Returns:
            Dictionary with sentiment metrics
        """
        if not posts:
            return {
                'overall_sentiment': 0.0,
                'post_count': 0,
                'average_score': 0.0,
                'engagement_rate': 0.0
            }
        
        coin_name = symbol.replace('USDT', '').replace('USD', '').lower()
        relevant_posts = []
        
        for post in posts:
            text = f"{post.get('title', '')} {post.get('text', '')}".lower()
            if coin_name in text:
                relevant_posts.append(post)
        
        if not relevant_posts:
            return {
                'overall_sentiment': 0.0,
                'post_count': 0,
                'average_score': 0.0,
                'engagement_rate': 0.0
            }
        
        # Calculate metrics
        total_score = sum(p.get('score', 0) for p in relevant_posts)
        avg_score = total_score / len(relevant_posts)
        avg_upvote_ratio = sum(p.get('upvote_ratio', 0) for p in relevant_posts) / len(relevant_posts)
        total_comments = sum(p.get('num_comments', 0) for p in relevant_posts)
        
        # Sentiment score based on upvote ratio
        sentiment = (avg_upvote_ratio - 0.5) * 2  # Scale to -1 to 1
        
        return {
            'overall_sentiment': round(sentiment, 3),
            'post_count': len(relevant_posts),
            'average_score': round(avg_score, 2),
            'total_comments': total_comments,
            'engagement_rate': round(avg_upvote_ratio, 3)
        }
    
    def calculate_twitter_sentiment(self, tweets: List[Dict]) -> Dict:
        """
        Calculate sentiment from Twitter data.
        
        Args:
            tweets: List of tweets
        
        Returns:
            Dictionary with sentiment metrics
        """
        if not tweets:
            return {
                'overall_sentiment': 0.0,
                'tweet_count': 0,
                'total_engagement': 0,
                'average_engagement': 0.0
            }
        
        total_engagement = sum(
            t.get('retweet_count', 0) + t.get('like_count', 0) + t.get('reply_count', 0)
            for t in tweets
        )
        avg_engagement = total_engagement / len(tweets)
        
        # Simple sentiment based on engagement
        # Higher engagement often correlates with positive sentiment
        normalized_engagement = min(avg_engagement / 100, 1.0)
        sentiment = (normalized_engagement - 0.5) * 2
        
        return {
            'overall_sentiment': round(sentiment, 3),
            'tweet_count': len(tweets),
            'total_engagement': total_engagement,
            'average_engagement': round(avg_engagement, 2)
        }
    
    def collect_social_data(self, symbol: str) -> Dict:
        """
        Collect comprehensive social media data.
        
        Args:
            symbol: Cryptocurrency symbol
        
        Returns:
            Dictionary with social media metrics
        """
        social_data = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'platforms': {}
        }
        
        coin_name = symbol.replace('USDT', '').replace('USD', '').lower()
        
        # Collect Reddit data
        subreddits = ['cryptocurrency', coin_name, f'{coin_name}trading']
        reddit_posts = []
        
        for subreddit in subreddits:
            posts = self.get_reddit_posts(subreddit, limit=50)
            if posts:
                reddit_posts.extend(posts)
        
        if reddit_posts:
            social_data['platforms']['reddit'] = self.calculate_reddit_sentiment(reddit_posts, symbol)
        
        # Collect Twitter data
        twitter_query = f"${coin_name.upper()} OR #{coin_name} -is:retweet lang:en"
        tweets = self.get_twitter_mentions(twitter_query, max_results=100)
        
        if tweets:
            social_data['platforms']['twitter'] = self.calculate_twitter_sentiment(tweets)
        
        # Calculate overall social sentiment
        platform_sentiments = [
            data.get('overall_sentiment', 0)
            for data in social_data['platforms'].values()
        ]
        
        if platform_sentiments:
            social_data['overall_social_sentiment'] = round(
                sum(platform_sentiments) / len(platform_sentiments), 3
            )
        else:
            social_data['overall_social_sentiment'] = 0.0
        
        logger.info(f"Collected social data from {len(social_data['platforms'])} platforms")
        return social_data
