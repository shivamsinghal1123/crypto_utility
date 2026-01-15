"""
On-Chain Data Collection Module
Collects blockchain metrics and on-chain data
"""

import requests
from typing import Dict, List, Optional
from datetime import datetime
import logging

from config.settings import API_ENDPOINTS, REQUEST_CONFIG
from utils.helpers import handle_api_errors

logger = logging.getLogger(__name__)


class OnChainDataCollector:
    """Collects on-chain data from various blockchain explorers and APIs."""
    
    def __init__(self, etherscan_api_key: str = None, bscscan_api_key: str = None):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': REQUEST_CONFIG['USER_AGENT']})
        self.etherscan_base = API_ENDPOINTS['ETHERSCAN_BASE']
        self.bscscan_base = API_ENDPOINTS['BSCSCAN_BASE']
        self.defillama_base = API_ENDPOINTS['DEFILLAMA_BASE']
        self.coingecko_base = API_ENDPOINTS['COINGECKO_BASE']
        self.etherscan_api_key = etherscan_api_key
        self.bscscan_api_key = bscscan_api_key
    
    @handle_api_errors
    def get_etherscan_stats(self, contract_address: str) -> Optional[Dict]:
        """
        Get token statistics from Etherscan.
        
        Args:
            contract_address: Token contract address
        
        Returns:
            Dictionary with token statistics or None if failed
        """
        if not self.etherscan_api_key:
            logger.warning("Etherscan API key not provided")
            return None
        
        params = {
            'module': 'stats',
            'action': 'tokensupply',
            'contractaddress': contract_address,
            'apikey': self.etherscan_api_key
        }
        
        response = self.session.get(
            self.etherscan_base,
            params=params,
            timeout=REQUEST_CONFIG['TIMEOUT']
        )
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == '1':
            return {
                'total_supply': int(data['result']),
                'source': 'etherscan'
            }
        return None
    
    @handle_api_errors
    def get_token_holders(self, contract_address: str, blockchain: str = 'ethereum') -> Optional[Dict]:
        """
        Get token holder distribution.
        
        Args:
            contract_address: Token contract address
            blockchain: Blockchain network ('ethereum' or 'bsc')
        
        Returns:
            Dictionary with holder statistics or None if failed
        """
        if blockchain == 'ethereum' and not self.etherscan_api_key:
            logger.warning("Etherscan API key not provided")
            return None
        elif blockchain == 'bsc' and not self.bscscan_api_key:
            logger.warning("BSCScan API key not provided")
            return None
        
        base_url = self.etherscan_base if blockchain == 'ethereum' else self.bscscan_base
        api_key = self.etherscan_api_key if blockchain == 'ethereum' else self.bscscan_api_key
        
        params = {
            'module': 'token',
            'action': 'tokenholderlist',
            'contractaddress': contract_address,
            'page': 1,
            'offset': 100,
            'apikey': api_key
        }
        
        response = self.session.get(
            base_url,
            params=params,
            timeout=REQUEST_CONFIG['TIMEOUT']
        )
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') == '1':
            holders = data.get('result', [])
            total_holders = len(holders)
            
            return {
                'total_holders': total_holders,
                'top_holders': holders[:10],
                'blockchain': blockchain
            }
        return None
    
    @handle_api_errors
    def get_defillama_tvl(self, protocol: str) -> Optional[Dict]:
        """
        Get Total Value Locked (TVL) data from DefiLlama.
        
        Args:
            protocol: Protocol name (e.g., 'uniswap', 'aave')
        
        Returns:
            Dictionary with TVL data or None if failed
        """
        url = f"{self.defillama_base}protocol/{protocol}"
        
        response = self.session.get(url, timeout=REQUEST_CONFIG['TIMEOUT'])
        response.raise_for_status()
        
        data = response.json()
        
        return {
            'name': data.get('name'),
            'symbol': data.get('symbol'),
            'current_tvl': data.get('tvl', [{}])[-1].get('totalLiquidityUSD', 0),
            'chain_tvls': data.get('chainTvls', {}),
            'category': data.get('category'),
            'chains': data.get('chains', [])
        }
    
    @handle_api_errors
    def get_coingecko_onchain_data(self, coin_id: str) -> Optional[Dict]:
        """
        Get on-chain data from CoinGecko.
        
        Args:
            coin_id: CoinGecko coin ID
        
        Returns:
            Dictionary with on-chain metrics or None if failed
        """
        url = f"{self.coingecko_base}coins/{coin_id}"
        params = {
            'localization': 'false',
            'tickers': 'false',
            'market_data': 'true',
            'community_data': 'true',
            'developer_data': 'true'
        }
        
        response = self.session.get(url, params=params, timeout=REQUEST_CONFIG['TIMEOUT'])
        response.raise_for_status()
        
        data = response.json()
        
        developer_data = data.get('developer_data', {})
        community_data = data.get('community_data', {})
        
        return {
            'github_stats': {
                'forks': developer_data.get('forks'),
                'stars': developer_data.get('stars'),
                'subscribers': developer_data.get('subscribers'),
                'total_issues': developer_data.get('total_issues'),
                'closed_issues': developer_data.get('closed_issues'),
                'pull_requests_merged': developer_data.get('pull_requests_merged'),
                'commit_count_4_weeks': developer_data.get('commit_count_4_weeks')
            },
            'community_stats': {
                'twitter_followers': community_data.get('twitter_followers'),
                'reddit_subscribers': community_data.get('reddit_subscribers'),
                'reddit_active_accounts_48h': community_data.get('reddit_accounts_active_48h'),
                'telegram_channel_user_count': community_data.get('telegram_channel_user_count')
            },
            'circulating_supply': data.get('market_data', {}).get('circulating_supply'),
            'total_supply': data.get('market_data', {}).get('total_supply'),
            'max_supply': data.get('market_data', {}).get('max_supply')
        }
    
    def get_onchain_metrics(self, symbol: str, coin_id: str = None, contract_address: str = None) -> Dict:
        """
        Collect comprehensive on-chain metrics.
        
        Args:
            symbol: Cryptocurrency symbol
            coin_id: CoinGecko coin ID
            contract_address: Token contract address (if applicable)
        
        Returns:
            Dictionary with on-chain metrics
        """
        metrics = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'data_sources': []
        }
        
        # Get CoinGecko data
        if coin_id:
            cg_data = self.get_coingecko_onchain_data(coin_id)
            if cg_data:
                metrics['coingecko_data'] = cg_data
                metrics['data_sources'].append('coingecko')
        
        # Get Etherscan data for ERC-20 tokens
        if contract_address and self.etherscan_api_key:
            eth_stats = self.get_etherscan_stats(contract_address)
            if eth_stats:
                metrics['etherscan_data'] = eth_stats
                metrics['data_sources'].append('etherscan')
            
            holders = self.get_token_holders(contract_address, 'ethereum')
            if holders:
                metrics['holder_data'] = holders
        
        logger.info(f"Collected on-chain data from {len(metrics['data_sources'])} sources")
        return metrics
    
    def calculate_network_metrics(self, onchain_data: Dict) -> Dict:
        """
        Calculate derived network metrics.
        
        Args:
            onchain_data: Raw on-chain data
        
        Returns:
            Dictionary with calculated metrics
        """
        metrics = {}
        
        # GitHub activity score (0-10)
        github_stats = onchain_data.get('coingecko_data', {}).get('github_stats', {})
        if github_stats:
            commits = github_stats.get('commit_count_4_weeks') or 0
            stars = github_stats.get('stars') or 0
            forks = github_stats.get('forks') or 0
            
            commit_score = min(commits / 100, 1.0) * 5
            star_score = min(stars / 1000, 1.0) * 3
            fork_score = min(forks / 500, 1.0) * 2
            
            metrics['github_activity_score'] = round(commit_score + star_score + fork_score, 2)
        
        # Community engagement score (0-10)
        community_stats = onchain_data.get('coingecko_data', {}).get('community_stats', {})
        if community_stats:
            twitter_followers = community_stats.get('twitter_followers') or 0
            reddit_subscribers = community_stats.get('reddit_subscribers') or 0
            telegram_users = community_stats.get('telegram_channel_user_count') or 0
            
            twitter_score = min(twitter_followers / 100000, 1.0) * 4
            reddit_score = min(reddit_subscribers / 50000, 1.0) * 3
            telegram_score = min(telegram_users / 10000, 1.0) * 3
            
            metrics['community_engagement_score'] = round(twitter_score + reddit_score + telegram_score, 2)
        
        # Supply metrics
        cg_data = onchain_data.get('coingecko_data', {})
        if cg_data:
            circulating = cg_data.get('circulating_supply') or 0
            total = cg_data.get('total_supply') or 0
            max_supply = cg_data.get('max_supply') or 0
            
            if total and circulating:
                metrics['circulation_ratio'] = round(circulating / total, 4)
            
            if max_supply and circulating:
                metrics['supply_inflation_potential'] = round((max_supply - circulating) / circulating, 4)
        
        return metrics
