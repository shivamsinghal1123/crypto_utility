"""
Fundamental Analysis Module
Analyzes tokenomics, project fundamentals, and valuation metrics
"""

from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FundamentalAnalyzer:
    """Performs fundamental analysis on cryptocurrencies."""
    
    def __init__(self):
        pass
    
    def analyze_tokenomics(self, symbol: str, market_data: Dict, onchain_data: Dict = None) -> Dict:
        """
        Analyze tokenomics of a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            market_data: Market data from price collector
            onchain_data: On-chain data (optional)
        
        Returns:
            Dictionary with tokenomics analysis
        """
        analysis = {
            'symbol': symbol,
            'supply_metrics': {},
            'distribution': {},
            'utility_score': 0.0,
            'scarcity_score': 0.0
        }
        
        # Extract supply metrics
        if onchain_data and 'coingecko_data' in onchain_data:
            cg_data = onchain_data['coingecko_data']
            circulating = cg_data.get('circulating_supply') or 0
            total = cg_data.get('total_supply') or 0
            max_supply = cg_data.get('max_supply') or 0
            
            analysis['supply_metrics'] = {
                'circulating_supply': circulating,
                'total_supply': total,
                'max_supply': max_supply,
                'inflation_rate': self._calculate_inflation_rate(circulating, total, max_supply)
            }
            
            # Calculate scarcity score (0-10)
            if max_supply and circulating:
                scarcity_ratio = circulating / max_supply
                # Higher scarcity when more of max supply is in circulation
                analysis['scarcity_score'] = round(scarcity_ratio * 10, 2)
            elif not max_supply:
                # Unlimited supply = low scarcity
                analysis['scarcity_score'] = 3.0
        
        # Utility score based on various factors
        # This is a simplified model - in production, would need more comprehensive data
        analysis['utility_score'] = self._calculate_utility_score(symbol, market_data, onchain_data)
        
        return analysis
    
    def _calculate_inflation_rate(self, circulating: float, total: float, max_supply: float) -> float:
        """Calculate estimated inflation rate."""
        if not circulating or not total:
            return 0.0
        
        if max_supply:
            # Remaining supply to be issued
            remaining = max_supply - circulating
            # Approximate annual inflation (simplified)
            inflation_rate = (remaining / circulating) * 100 / 10  # Divided by estimated years
            return round(min(inflation_rate, 100), 2)
        else:
            # Unlimited supply - use difference between total and circulating
            if total > circulating:
                inflation_rate = ((total - circulating) / circulating) * 100
                return round(min(inflation_rate, 100), 2)
        
        return 0.0
    
    def _calculate_utility_score(self, symbol: str, market_data: Dict, onchain_data: Dict = None) -> float:
        """
        Calculate utility score based on various factors.
        
        Returns score from 0-10
        """
        score = 5.0  # Start with neutral score
        
        # Adjust based on market cap (larger cap often indicates more utility/adoption)
        market_cap = market_data.get('market_cap') or 0
        if market_cap > 10_000_000_000:  # >$10B
            score += 2.0
        elif market_cap > 1_000_000_000:  # >$1B
            score += 1.0
        elif market_cap < 100_000_000:  # <$100M
            score -= 1.0
        
        # Adjust based on on-chain activity
        if onchain_data:
            github_score = onchain_data.get('github_activity_score') or 0
            community_score = onchain_data.get('community_engagement_score') or 0
            
            # GitHub activity contributes to utility
            if github_score > 7:
                score += 1.5
            elif github_score > 5:
                score += 1.0
            
            # Community engagement contributes to utility
            if community_score > 7:
                score += 1.5
            elif community_score > 5:
                score += 1.0
        
        return round(min(max(score, 0), 10), 2)
    
    def analyze_project_fundamentals(self, symbol: str, onchain_data: Dict = None) -> Dict:
        """
        Analyze project fundamentals including team, technology, and partnerships.
        
        Args:
            symbol: Cryptocurrency symbol
            onchain_data: On-chain data with GitHub and community metrics
        
        Returns:
            Dictionary with fundamental scores
        """
        fundamentals = {
            'symbol': symbol,
            'team_score': 5.0,  # Default neutral score
            'technology_score': 5.0,
            'community_score': 5.0,
            'overall_fundamental_score': 5.0
        }
        
        if not onchain_data:
            return fundamentals
        
        # Technology score based on GitHub activity
        github_stats = onchain_data.get('coingecko_data', {}).get('github_stats', {})
        if github_stats:
            commits = github_stats.get('commit_count_4_weeks') or 0
            stars = github_stats.get('stars') or 0
            forks = github_stats.get('forks') or 0
            
            tech_score = 0
            # Active development
            if commits > 100:
                tech_score += 3
            elif commits > 50:
                tech_score += 2
            elif commits > 10:
                tech_score += 1
            
            # Community interest (stars)
            if stars > 5000:
                tech_score += 2
            elif stars > 1000:
                tech_score += 1
            
            # Developer adoption (forks)
            if forks > 1000:
                tech_score += 2
            elif forks > 500:
                tech_score += 1
            
            fundamentals['technology_score'] = round(min(tech_score, 10), 2)
        
        # Community score
        community_stats = onchain_data.get('coingecko_data', {}).get('community_stats', {})
        if community_stats:
            twitter = community_stats.get('twitter_followers') or 0
            reddit = community_stats.get('reddit_subscribers') or 0
            
            comm_score = 0
            if twitter > 500000:
                comm_score += 3
            elif twitter > 100000:
                comm_score += 2
            elif twitter > 10000:
                comm_score += 1
            
            if reddit > 100000:
                comm_score += 3
            elif reddit > 50000:
                comm_score += 2
            elif reddit > 10000:
                comm_score += 1
            
            fundamentals['community_score'] = round(min(comm_score + 2, 10), 2)
        
        # Team score (simplified - would need more data for accurate assessment)
        # Base it on technology and community scores
        fundamentals['team_score'] = round(
            (fundamentals['technology_score'] + fundamentals['community_score']) / 2, 2
        )
        
        # Overall score
        fundamentals['overall_fundamental_score'] = round(
            (fundamentals['team_score'] + fundamentals['technology_score'] + 
             fundamentals['community_score']) / 3, 2
        )
        
        return fundamentals
    
    def calculate_valuation_metrics(self, symbol: str, price_data: Dict, 
                                   fundamental_data: Dict, onchain_data: Dict = None) -> Dict:
        """
        Calculate valuation metrics.
        
        Args:
            symbol: Cryptocurrency symbol
            price_data: Current price and market data
            fundamental_data: Fundamental analysis data
            onchain_data: On-chain data (optional)
        
        Returns:
            Dictionary with valuation metrics
        """
        valuation = {
            'symbol': symbol,
            'timestamp': datetime.now()
        }
        
        # Market Cap
        current_price = price_data.get('price') or 0
        circulating_supply = 0
        
        if onchain_data and 'coingecko_data' in onchain_data:
            circulating_supply = onchain_data['coingecko_data'].get('circulating_supply') or 0
        
        if current_price and circulating_supply:
            valuation['market_cap'] = current_price * circulating_supply
        
        # Fully Diluted Valuation
        max_supply = 0
        if onchain_data and 'coingecko_data' in onchain_data:
            max_supply = onchain_data['coingecko_data'].get('max_supply') or 0
        
        if current_price and max_supply:
            valuation['fully_diluted_valuation'] = current_price * max_supply
        
        # Network Value to Transactions (NVT) Ratio
        # Simplified calculation: market_cap / 24h volume
        volume_24h = price_data.get('volume_24h') or 0
        if valuation.get('market_cap') and volume_24h:
            valuation['nvt_ratio'] = round(valuation['market_cap'] / (volume_24h * current_price), 2)
        
        # Network growth rate (based on GitHub activity)
        if onchain_data:
            github_stats = onchain_data.get('coingecko_data', {}).get('github_stats', {})
            commits = github_stats.get('commit_count_4_weeks') or 0
            valuation['network_growth_rate'] = round(min(commits / 10, 10), 2)
        
        # Developer activity score
        if onchain_data:
            valuation['developer_activity_score'] = onchain_data.get('github_activity_score') or 0
        
        return valuation
    
    def analyze(self, symbol: str, price_data: Dict, onchain_data: Dict = None) -> Dict:
        """
        Perform comprehensive fundamental analysis.
        
        Args:
            symbol: Cryptocurrency symbol
            price_data: Price and market data
            onchain_data: On-chain data (optional)
        
        Returns:
            Complete fundamental analysis
        """
        logger.info(f"Starting fundamental analysis for {symbol}")
        
        # Tokenomics analysis
        tokenomics = self.analyze_tokenomics(symbol, price_data, onchain_data)
        
        # Project fundamentals
        fundamentals = self.analyze_project_fundamentals(symbol, onchain_data)
        
        # Valuation metrics
        valuation = self.calculate_valuation_metrics(symbol, price_data, fundamentals, onchain_data)
        
        # Calculate overall fundamental score (0-10)
        overall_score = round(
            (tokenomics.get('utility_score', 0) * 0.3 +
             tokenomics.get('scarcity_score', 0) * 0.2 +
             fundamentals.get('overall_fundamental_score', 0) * 0.5), 2
        )
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'tokenomics': tokenomics,
            'fundamentals': fundamentals,
            'valuation': valuation,
            'overall_score': overall_score
        }
