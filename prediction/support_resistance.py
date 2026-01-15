"""
Support and Resistance Calculation Module
Calculates support and resistance levels using multiple methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import logging

from config.settings import SR_SETTINGS

logger = logging.getLogger(__name__)


class SupportResistanceCalculator:
    """Calculates support and resistance levels using multiple methods."""
    
    def __init__(self):
        self.settings = SR_SETTINGS
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """
        Calculate classic pivot points.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary with pivot points
        """
        # Use last complete candle
        high = df['high'].iloc[-2]
        low = df['low'].iloc[-2]
        close = df['close'].iloc[-2]
        
        # Classic pivot point
        pivot = (high + low + close) / 3
        
        # Support levels
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        # Resistance levels
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        return {
            'pivot': round(pivot, 2),
            'resistance': [round(r1, 2), round(r2, 2), round(r3, 2)],
            'support': [round(s1, 2), round(s2, 2), round(s3, 2)],
            'method': 'pivot_points'
        }
    
    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary with Fibonacci levels
        """
        # Find recent high and low
        lookback = min(len(df), 100)
        recent_data = df.tail(lookback)
        
        high = recent_data['high'].max()
        low = recent_data['low'].min()
        diff = high - low
        
        current_price = df['close'].iloc[-1]
        
        # Fibonacci levels
        fib_levels = {}
        for level in self.settings['FIBONACCI_LEVELS']:
            fib_levels[f'fib_{level}'] = low + (diff * level)
        
        # Determine which are support and which are resistance
        support = []
        resistance = []
        
        for level_name, level_price in fib_levels.items():
            if level_price < current_price:
                support.append(round(level_price, 2))
            else:
                resistance.append(round(level_price, 2))
        
        return {
            'high': round(high, 2),
            'low': round(low, 2),
            'levels': {k: round(v, 2) for k, v in fib_levels.items()},
            'support': sorted(support, reverse=True),
            'resistance': sorted(resistance),
            'method': 'fibonacci'
        }
    
    def calculate_volume_nodes(self, df: pd.DataFrame) -> Dict:
        """
        Calculate support/resistance from volume profile.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary with volume-based levels
        """
        lookback = min(len(df), 100)
        recent_data = df.tail(lookback)
        
        # Create price bins
        price_min = recent_data['low'].min()
        price_max = recent_data['high'].max()
        num_bins = 20
        
        bins = np.linspace(price_min, price_max, num_bins + 1)
        bin_volume = np.zeros(num_bins)
        
        # Distribute volume across price bins
        for _, row in recent_data.iterrows():
            low_bin = np.digitize(row['low'], bins) - 1
            high_bin = np.digitize(row['high'], bins) - 1
            
            for i in range(max(0, low_bin), min(num_bins, high_bin + 1)):
                bin_volume[i] += row['volume'] / (high_bin - low_bin + 1)
        
        # Find high volume nodes (HVN) and low volume nodes (LVN)
        avg_volume = bin_volume.mean()
        
        hvn_indices = np.where(bin_volume > avg_volume * 1.5)[0]
        lvn_indices = np.where(bin_volume < avg_volume * 0.5)[0]
        
        hvn_prices = [(bins[i] + bins[i+1]) / 2 for i in hvn_indices]
        lvn_prices = [(bins[i] + bins[i+1]) / 2 for i in lvn_indices]
        
        current_price = df['close'].iloc[-1]
        
        # HVNs act as support/resistance
        hvn_support = [p for p in hvn_prices if p < current_price]
        hvn_resistance = [p for p in hvn_prices if p > current_price]
        
        return {
            'hvn_support': [round(p, 2) for p in sorted(hvn_support, reverse=True)[:3]],
            'hvn_resistance': [round(p, 2) for p in sorted(hvn_resistance)[:3]],
            'lvn_levels': [round(p, 2) for p in lvn_prices],
            'method': 'volume_profile'
        }
    
    def calculate_psychological_levels(self, current_price: float) -> Dict:
        """
        Calculate psychological round number levels.
        
        Args:
            current_price: Current price
        
        Returns:
            Dictionary with psychological levels
        """
        # Determine the magnitude
        magnitude = 10 ** (len(str(int(current_price))) - 1)
        
        # Round numbers
        round_numbers = []
        for i in range(-5, 6):
            level = round(current_price / magnitude) * magnitude + (i * magnitude / 10)
            if level > 0:
                round_numbers.append(level)
        
        support = [p for p in round_numbers if p < current_price]
        resistance = [p for p in round_numbers if p > current_price]
        
        return {
            'support': sorted(support, reverse=True)[:3],
            'resistance': sorted(resistance)[:3],
            'method': 'psychological'
        }
    
    def calculate_ma_levels(self, df: pd.DataFrame) -> Dict:
        """
        Calculate support/resistance from moving averages.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary with MA-based levels
        """
        close = df['close']
        current_price = close.iloc[-1]
        
        # Calculate key MAs
        sma_20 = close.rolling(window=20).mean().iloc[-1]
        sma_50 = close.rolling(window=50).mean().iloc[-1]
        sma_200 = close.rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
        ema_12 = close.ewm(span=12).mean().iloc[-1]
        ema_26 = close.ewm(span=26).mean().iloc[-1]
        
        levels = [sma_20, sma_50, ema_12, ema_26]
        if sma_200:
            levels.append(sma_200)
        
        support = [round(p, 2) for p in levels if p < current_price]
        resistance = [round(p, 2) for p in levels if p > current_price]
        
        return {
            'support': sorted(support, reverse=True),
            'resistance': sorted(resistance),
            'sma_20': round(sma_20, 2),
            'sma_50': round(sma_50, 2),
            'method': 'moving_averages'
        }
    
    def calculate_bb_levels(self, df: pd.DataFrame) -> Dict:
        """
        Calculate support/resistance from Bollinger Bands.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary with Bollinger Band levels
        """
        close = df['close']
        current_price = close.iloc[-1]
        
        # Calculate Bollinger Bands
        sma = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        
        upper_band = (sma + (std * 2)).iloc[-1]
        middle_band = sma.iloc[-1]
        lower_band = (sma - (std * 2)).iloc[-1]
        
        levels = [upper_band, middle_band, lower_band]
        
        support = [round(p, 2) for p in levels if p < current_price]
        resistance = [round(p, 2) for p in levels if p > current_price]
        
        return {
            'support': sorted(support, reverse=True),
            'resistance': sorted(resistance),
            'upper_band': round(upper_band, 2),
            'middle_band': round(middle_band, 2),
            'lower_band': round(lower_band, 2),
            'method': 'bollinger_bands'
        }
    
    def combine_levels(self, all_levels: List[Dict], current_price: float) -> Dict:
        """
        Combine levels from all methods and assign confidence scores.
        
        Args:
            all_levels: List of level dictionaries from different methods
            current_price: Current price
        
        Returns:
            Combined support and resistance levels with confidence
        """
        support_levels = []
        resistance_levels = []
        
        # Collect all levels with their methods
        for level_dict in all_levels:
            method = level_dict.get('method', 'unknown')
            
            for support in level_dict.get('support', []):
                support_levels.append({
                    'price': support,
                    'method': method,
                    'distance': abs(current_price - support) / current_price
                })
            
            for resistance in level_dict.get('resistance', []):
                resistance_levels.append({
                    'price': resistance,
                    'method': method,
                    'distance': abs(resistance - current_price) / current_price
                })
        
        # Cluster nearby levels (within 1% of each other)
        def cluster_and_score(levels, tolerance=0.01):
            if not levels:
                return []
            
            levels.sort(key=lambda x: x['price'])
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if abs(level['price'] - current_cluster[0]['price']) / current_cluster[0]['price'] < tolerance:
                    current_cluster.append(level)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [level]
            
            clusters.append(current_cluster)
            
            # Calculate average and confidence for each cluster
            result = []
            for cluster in clusters:
                avg_price = np.mean([l['price'] for l in cluster])
                strength = len(cluster) / len(set(l['method'] for l in cluster))  # More methods = higher strength
                avg_distance = np.mean([l['distance'] for l in cluster])
                
                result.append({
                    'level': round(avg_price, 2),
                    'strength': round(min(strength, 1.0), 2),
                    'methods': list(set(l['method'] for l in cluster)),
                    'distance_percent': round(avg_distance * 100, 2)
                })
            
            return sorted(result, key=lambda x: x['distance_percent'])
        
        support_clustered = cluster_and_score(support_levels)
        resistance_clustered = cluster_and_score(resistance_levels)
        
        return {
            'support': support_clustered[:5],  # Top 5 support levels
            'resistance': resistance_clustered[:5]  # Top 5 resistance levels
        }
    
    def calculate_24h_levels(self, df: pd.DataFrame) -> Dict:
        """
        Calculate support and resistance levels for next 24 hours.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary with 24h support/resistance predictions
        """
        logger.info("Calculating 24h support and resistance levels")
        
        current_price = df['close'].iloc[-1]
        
        # Calculate levels using all methods
        pivot_levels = self.calculate_pivot_points(df)
        fib_levels = self.calculate_fibonacci_levels(df)
        volume_levels = self.calculate_volume_nodes(df)
        psych_levels = self.calculate_psychological_levels(current_price)
        ma_levels = self.calculate_ma_levels(df)
        bb_levels = self.calculate_bb_levels(df)
        
        all_levels = [
            pivot_levels,
            fib_levels,
            volume_levels,
            psych_levels,
            ma_levels,
            bb_levels
        ]
        
        # Combine and score levels
        combined = self.combine_levels(all_levels, current_price)
        
        # Determine market bias
        support_strength = sum(l['strength'] for l in combined['support'][:3])
        resistance_strength = sum(l['strength'] for l in combined['resistance'][:3])
        
        if support_strength > resistance_strength * 1.2:
            market_bias = 'bullish'
        elif resistance_strength > support_strength * 1.2:
            market_bias = 'bearish'
        else:
            market_bias = 'neutral'
        
        # Calculate confidence score
        total_levels = len(combined['support']) + len(combined['resistance'])
        avg_strength = (support_strength + resistance_strength) / total_levels if total_levels > 0 else 0
        confidence_score = round(min(avg_strength * 100, 100), 1)
        
        return {
            'timestamp': datetime.now(),
            'current_price': round(current_price, 2),
            'next_24h_support': combined['support'],
            'next_24h_resistance': combined['resistance'],
            'market_bias': market_bias,
            'confidence_score': confidence_score,
            'methods_used': ['pivot_points', 'fibonacci', 'volume_profile', 
                           'psychological', 'moving_averages', 'bollinger_bands']
        }
