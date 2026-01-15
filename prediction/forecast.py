"""
Forecast Module
Price prediction and forecasting models
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PriceForecaster:
    """Forecasts price movements and probabilities."""
    
    def __init__(self):
        pass
    
    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        """
        Calculate historical volatility.
        
        Args:
            df: DataFrame with price data
            window: Rolling window for volatility calculation
        
        Returns:
            Annualized volatility
        """
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        
        # Annualize (assuming 365 trading days for crypto)
        annualized_volatility = volatility.iloc[-1] * np.sqrt(365)
        
        return annualized_volatility
    
    def predict_price_probability(self, df: pd.DataFrame, technical_analysis: Dict,
                                 sentiment_analysis: Dict) -> Dict:
        """
        Predict probability of price moving up or down.
        
        Args:
            df: DataFrame with OHLCV data
            technical_analysis: Technical analysis results
            sentiment_analysis: Sentiment analysis results
        
        Returns:
            Dictionary with probability predictions
        """
        # Base probability
        prob_up = 0.5
        
        # Technical indicators influence
        trend = technical_analysis.get('trend', {})
        momentum = technical_analysis.get('momentum', {})
        
        # Trend direction
        if trend.get('direction') == 'bullish':
            prob_up += 0.1 * trend.get('strength', 0)
        elif trend.get('direction') == 'bearish':
            prob_up -= 0.1 * trend.get('strength', 0)
        
        # RSI
        rsi_data = momentum.get('rsi', {})
        rsi_value = rsi_data.get('value', 50)
        
        if rsi_value > 70:
            prob_up -= 0.1  # Overbought
        elif rsi_value < 30:
            prob_up += 0.1  # Oversold
        
        # MACD
        macd_data = momentum.get('macd', {})
        if macd_data.get('signal') == 'bullish':
            prob_up += 0.05
        elif macd_data.get('signal') == 'bearish':
            prob_up -= 0.05
        
        # Sentiment influence
        overall_sentiment = sentiment_analysis.get('overall_sentiment', 0)
        prob_up += overall_sentiment * 0.1
        
        # Ensure probabilities are between 0 and 1
        prob_up = max(0.1, min(0.9, prob_up))
        prob_down = 1 - prob_up
        
        return {
            'probability_up': round(prob_up, 3),
            'probability_down': round(prob_down, 3),
            'confidence': round(abs(prob_up - 0.5) * 2, 2)  # 0 = uncertain, 1 = very confident
        }
    
    def forecast_price_range(self, df: pd.DataFrame, hours: int = 24) -> Dict:
        """
        Forecast expected price range for next N hours.
        
        Args:
            df: DataFrame with OHLCV data
            hours: Forecast horizon in hours
        
        Returns:
            Dictionary with price range forecast
        """
        current_price = df['close'].iloc[-1]
        
        # Calculate volatility
        volatility = self.calculate_volatility(df)
        
        # Expected move (1 standard deviation)
        # Adjust for time horizon
        time_factor = np.sqrt(hours / (24 * 365))
        expected_move = current_price * volatility * time_factor
        
        # Calculate ranges
        # 1 std dev (68% confidence)
        range_1std = {
            'lower': round(current_price - expected_move, 2),
            'upper': round(current_price + expected_move, 2),
            'confidence': 0.68
        }
        
        # 2 std dev (95% confidence)
        range_2std = {
            'lower': round(current_price - (2 * expected_move), 2),
            'upper': round(current_price + (2 * expected_move), 2),
            'confidence': 0.95
        }
        
        return {
            'current_price': round(current_price, 2),
            'forecast_hours': hours,
            'expected_volatility': round(volatility * 100, 2),
            'ranges': {
                '1_std_dev': range_1std,
                '2_std_dev': range_2std
            }
        }
    
    def simple_trend_forecast(self, df: pd.DataFrame, periods_ahead: int = 24) -> Dict:
        """
        Simple trend-based forecast using moving averages.
        
        Args:
            df: DataFrame with OHLCV data
            periods_ahead: Number of periods to forecast
        
        Returns:
            Dictionary with trend forecast
        """
        close = df['close']
        
        # Calculate trend using linear regression on recent data
        recent_data = close.tail(50)
        x = np.arange(len(recent_data))
        y = recent_data.values
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Forecast
        current_price = close.iloc[-1]
        forecasted_price = current_price + (slope * periods_ahead)
        
        # Determine trend
        if slope > 0:
            trend_direction = 'upward'
        elif slope < 0:
            trend_direction = 'downward'
        else:
            trend_direction = 'sideways'
        
        return {
            'current_price': round(current_price, 2),
            'forecasted_price': round(forecasted_price, 2),
            'expected_change': round(((forecasted_price - current_price) / current_price) * 100, 2),
            'trend_direction': trend_direction,
            'periods_ahead': periods_ahead
        }
    
    def generate_forecast(self, df: pd.DataFrame, technical_analysis: Dict,
                         sentiment_analysis: Dict) -> Dict:
        """
        Generate comprehensive forecast.
        
        Args:
            df: DataFrame with OHLCV data
            technical_analysis: Technical analysis results
            sentiment_analysis: Sentiment analysis results
        
        Returns:
            Complete forecast
        """
        logger.info("Generating price forecast")
        
        # Price probabilities
        probabilities = self.predict_price_probability(df, technical_analysis, sentiment_analysis)
        
        # Price range forecast
        price_range = self.forecast_price_range(df, hours=24)
        
        # Trend forecast
        trend_forecast = self.simple_trend_forecast(df, periods_ahead=24)
        
        # Expected volatility
        volatility = self.calculate_volatility(df)
        
        return {
            'timestamp': datetime.now(),
            'forecast_horizon': '24 hours',
            'probabilities': probabilities,
            'price_ranges': price_range,
            'trend_forecast': trend_forecast,
            'expected_volatility': round(volatility * 100, 2)
        }
