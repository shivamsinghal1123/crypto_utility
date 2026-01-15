"""
Technical Analysis Module
Implements technical indicators and pattern recognition
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from config.settings import TECHNICAL_SETTINGS

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Performs technical analysis on price data."""
    
    def __init__(self):
        self.settings = TECHNICAL_SETTINGS
    
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=period).mean()
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            data: Price data series
            period: RSI period (default 14)
        
        Returns:
            RSI values
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, 
                                 std_dev: int = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        sma = self.calculate_sma(data, period)
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_stochastic_rsi(self, rsi: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic RSI.
        
        Returns:
            Dictionary with K and D lines
        """
        rsi_min = rsi.rolling(window=period).min()
        rsi_max = rsi.rolling(window=period).max()
        
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
        k_line = stoch_rsi.rolling(window=3).mean()
        d_line = k_line.rolling(window=3).mean()
        
        return {
            'k': k_line,
            'd': d_line
        }
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Calculate all technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary with all calculated indicators
        """
        indicators = {
            'trend_indicators': {},
            'momentum_indicators': {},
            'volume_indicators': {},
            'volatility_indicators': {}
        }
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Trend Indicators
        for period in self.settings['SMA_PERIODS']:
            indicators['trend_indicators'][f'sma_{period}'] = self.calculate_sma(close, period)
        
        for period in self.settings['EMA_PERIODS']:
            indicators['trend_indicators'][f'ema_{period}'] = self.calculate_ema(close, period)
        
        macd = self.calculate_macd(
            close,
            self.settings['MACD_FAST'],
            self.settings['MACD_SLOW'],
            self.settings['MACD_SIGNAL']
        )
        indicators['trend_indicators']['macd'] = macd
        
        bb = self.calculate_bollinger_bands(
            close,
            self.settings['BB_PERIOD'],
            self.settings['BB_STD_DEV']
        )
        indicators['trend_indicators']['bollinger_bands'] = bb
        
        # Momentum Indicators
        rsi = self.calculate_rsi(close, self.settings['RSI_PERIOD'])
        indicators['momentum_indicators']['rsi'] = rsi
        
        stoch_rsi = self.calculate_stochastic_rsi(rsi, 14)
        indicators['momentum_indicators']['stoch_rsi'] = stoch_rsi
        
        # Volume Indicators
        indicators['volume_indicators']['volume_sma'] = self.calculate_sma(
            volume, self.settings['VOLUME_SMA_PERIOD']
        )
        indicators['volume_indicators']['obv'] = self.calculate_obv(close, volume)
        
        # Volatility Indicators
        atr = self.calculate_atr(high, low, close, self.settings['ATR_PERIOD'])
        indicators['volatility_indicators']['atr'] = atr
        
        return indicators
    
    def detect_trend(self, df: pd.DataFrame) -> Dict:
        """
        Detect current trend direction.
        
        Returns:
            Dictionary with trend information
        """
        close = df['close']
        
        # Calculate moving averages
        sma_20 = self.calculate_sma(close, 20)
        sma_50 = self.calculate_sma(close, 50)
        ema_12 = self.calculate_ema(close, 12)
        ema_26 = self.calculate_ema(close, 26)
        
        current_price = close.iloc[-1]
        current_sma_20 = sma_20.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        current_ema_12 = ema_12.iloc[-1]
        current_ema_26 = ema_26.iloc[-1]
        
        # Trend determination
        bullish_signals = 0
        bearish_signals = 0
        
        if current_price > current_sma_20:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if current_price > current_sma_50:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if current_sma_20 > current_sma_50:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if current_ema_12 > current_ema_26:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Determine overall trend
        if bullish_signals >= 3:
            trend = 'bullish'
            strength = bullish_signals / 4.0
        elif bearish_signals >= 3:
            trend = 'bearish'
            strength = bearish_signals / 4.0
        else:
            trend = 'neutral'
            strength = 0.5
        
        return {
            'direction': trend,
            'strength': round(strength, 2),
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals
        }
    
    def analyze_momentum(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """
        Analyze momentum using RSI and other indicators.
        
        Returns:
            Dictionary with momentum analysis
        """
        rsi = indicators['momentum_indicators']['rsi']
        current_rsi = rsi.iloc[-1]
        
        # RSI interpretation
        if current_rsi > self.settings['RSI_OVERBOUGHT']:
            rsi_signal = 'overbought'
            momentum_score = -0.5
        elif current_rsi < self.settings['RSI_OVERSOLD']:
            rsi_signal = 'oversold'
            momentum_score = 0.5
        else:
            rsi_signal = 'neutral'
            # Scale RSI to momentum score (-1 to 1)
            momentum_score = (current_rsi - 50) / 50
        
        # MACD analysis
        macd_data = indicators['trend_indicators']['macd']
        macd_current = macd_data['macd'].iloc[-1]
        signal_current = macd_data['signal'].iloc[-1]
        histogram_current = macd_data['histogram'].iloc[-1]
        
        if macd_current > signal_current and histogram_current > 0:
            macd_signal = 'bullish'
        elif macd_current < signal_current and histogram_current < 0:
            macd_signal = 'bearish'
        else:
            macd_signal = 'neutral'
        
        return {
            'rsi': {
                'value': round(current_rsi, 2),
                'signal': rsi_signal
            },
            'macd': {
                'value': round(macd_current, 4),
                'signal': macd_signal,
                'histogram': round(histogram_current, 4)
            },
            'momentum_score': round(momentum_score, 2)
        }
    
    def assess_volatility(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """
        Assess current volatility.
        
        Returns:
            Dictionary with volatility metrics
        """
        atr = indicators['volatility_indicators']['atr']
        current_atr = atr.iloc[-1]
        avg_atr = atr.tail(20).mean()
        
        close = df['close']
        current_price = close.iloc[-1]
        
        # ATR as percentage of price
        atr_percentage = (current_atr / current_price) * 100
        
        # Bollinger Band width
        bb = indicators['trend_indicators']['bollinger_bands']
        bb_width = ((bb['upper'].iloc[-1] - bb['lower'].iloc[-1]) / bb['middle'].iloc[-1]) * 100
        
        # Volatility assessment
        if atr_percentage > 5 or bb_width > 10:
            volatility_level = 'high'
        elif atr_percentage > 2 or bb_width > 5:
            volatility_level = 'medium'
        else:
            volatility_level = 'low'
        
        return {
            'atr': round(current_atr, 2),
            'atr_percentage': round(atr_percentage, 2),
            'bb_width': round(bb_width, 2),
            'volatility_level': volatility_level
        }
    
    def find_support_resistance(self, df: pd.DataFrame, num_levels: int = 3) -> Dict:
        """
        Find support and resistance levels using pivot points and price clustering.
        
        Returns:
            Dictionary with support and resistance levels
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Find local maxima and minima
        from scipy.signal import argrelextrema
        
        # Local maxima (resistance)
        max_idx = argrelextrema(high, np.greater, order=5)[0]
        resistance_levels = high[max_idx]
        
        # Local minima (support)
        min_idx = argrelextrema(low, np.less, order=5)[0]
        support_levels = low[min_idx]
        
        # Cluster levels
        def cluster_levels(levels, num_clusters):
            if len(levels) == 0:
                return []
            
            # Sort levels
            sorted_levels = np.sort(levels)
            
            # Simple clustering: divide into bins
            if len(sorted_levels) < num_clusters:
                return sorted_levels.tolist()
            
            clusters = []
            step = len(sorted_levels) // num_clusters
            
            for i in range(num_clusters):
                start_idx = i * step
                end_idx = (i + 1) * step if i < num_clusters - 1 else len(sorted_levels)
                cluster_mean = np.mean(sorted_levels[start_idx:end_idx])
                clusters.append(cluster_mean)
            
            return clusters
        
        support = cluster_levels(support_levels, num_levels)
        resistance = cluster_levels(resistance_levels, num_levels)
        
        return {
            'support_levels': [round(s, 2) for s in support],
            'resistance_levels': [round(r, 2) for r in resistance],
            'current_price': round(close[-1], 2)
        }
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive technical analysis.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Complete technical analysis
        """
        logger.info("Starting technical analysis")
        
        # Calculate all indicators
        indicators = self.calculate_indicators(df)
        
        # Trend analysis
        trend = self.detect_trend(df)
        
        # Momentum analysis
        momentum = self.analyze_momentum(df, indicators)
        
        # Volatility assessment
        volatility = self.assess_volatility(df, indicators)
        
        # Support/Resistance levels
        sr_levels = self.find_support_resistance(df)
        
        return {
            'timestamp': datetime.now(),
            'indicators': indicators,
            'trend': trend,
            'momentum': momentum,
            'volatility': volatility,
            'support_resistance': sr_levels
        }
