"""
Valuation Module
Additional valuation models and metrics
"""

from typing import Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ValuationAnalyzer:
    """Performs valuation analysis on cryptocurrencies."""
    
    def __init__(self):
        pass
    
    def calculate_network_value(self, market_cap: float, active_addresses: int,
                                transaction_volume: float) -> Dict:
        """
        Calculate network value metrics.
        
        Args:
            market_cap: Market capitalization
            active_addresses: Number of active addresses
            transaction_volume: Daily transaction volume
        
        Returns:
            Dictionary with network value metrics
        """
        metrics = {}
        
        # Network Value per Active Address
        if active_addresses > 0:
            metrics['value_per_address'] = market_cap / active_addresses
        
        # Network Value to Transactions Ratio
        if transaction_volume > 0:
            metrics['nvt_ratio'] = market_cap / transaction_volume
        
        return metrics
    
    def assess_relative_valuation(self, symbol: str, market_cap: float,
                                  comparable_market_caps: Dict[str, float]) -> Dict:
        """
        Assess relative valuation compared to similar cryptocurrencies.
        
        Args:
            symbol: Symbol being analyzed
            market_cap: Market cap of the symbol
            comparable_market_caps: Dictionary of comparable symbols and their market caps
        
        Returns:
            Dictionary with relative valuation assessment
        """
        if not comparable_market_caps or market_cap == 0:
            return {
                'relative_position': 'unknown',
                'percentile': 0
            }
        
        # Calculate percentile
        all_caps = list(comparable_market_caps.values()) + [market_cap]
        all_caps.sort()
        
        position = all_caps.index(market_cap)
        percentile = (position / len(all_caps)) * 100
        
        if percentile >= 75:
            relative_position = 'top_quartile'
        elif percentile >= 50:
            relative_position = 'above_median'
        elif percentile >= 25:
            relative_position = 'below_median'
        else:
            relative_position = 'bottom_quartile'
        
        return {
            'relative_position': relative_position,
            'percentile': round(percentile, 1),
            'rank': position + 1,
            'total_compared': len(all_caps)
        }
    
    def calculate_risk_metrics(self, volatility: float, liquidity: float,
                              market_cap: float) -> Dict:
        """
        Calculate risk-adjusted valuation metrics.
        
        Args:
            volatility: Volatility measure (0-1)
            liquidity: 24h volume / market cap ratio
            market_cap: Market capitalization
        
        Returns:
            Dictionary with risk metrics
        """
        risk_score = 0
        
        # Volatility risk (higher volatility = higher risk)
        if volatility > 0.8:
            risk_score += 3
        elif volatility > 0.5:
            risk_score += 2
        elif volatility > 0.3:
            risk_score += 1
        
        # Liquidity risk (lower liquidity = higher risk)
        if liquidity < 0.01:  # Less than 1% daily volume
            risk_score += 3
        elif liquidity < 0.05:
            risk_score += 2
        elif liquidity < 0.1:
            risk_score += 1
        
        # Market cap risk (lower cap = higher risk)
        if market_cap < 100_000_000:  # <$100M
            risk_score += 3
        elif market_cap < 1_000_000_000:  # <$1B
            risk_score += 2
        elif market_cap < 10_000_000_000:  # <$10B
            risk_score += 1
        
        # Categorize risk
        if risk_score >= 7:
            risk_level = 'very_high'
        elif risk_score >= 5:
            risk_level = 'high'
        elif risk_score >= 3:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'volatility_risk': 'high' if volatility > 0.5 else 'medium' if volatility > 0.3 else 'low',
            'liquidity_risk': 'high' if liquidity < 0.05 else 'medium' if liquidity < 0.1 else 'low',
            'market_cap_risk': 'high' if market_cap < 1e9 else 'medium' if market_cap < 1e10 else 'low'
        }
