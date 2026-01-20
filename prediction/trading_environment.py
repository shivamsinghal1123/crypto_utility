"""
Reinforcement Learning trading environment for crypto price prediction.
Simulates market conditions and provides rewards based on prediction accuracy.
"""
import numpy as np
from typing import Dict, Tuple, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TradingEnvironment:
    """
    RL Environment for learning optimal trading predictions.
    
    State: Technical indicators, sentiment, fundamentals, market conditions
    Action: Price direction prediction (up/down/neutral)
    Reward: Based on prediction accuracy and confidence calibration
    """
    
    def __init__(self, state_dim: int = 50):
        """
        Initialize trading environment.
        
        Args:
            state_dim: Dimension of state vector
        """
        self.state_dim = state_dim
        self.action_space = ['up', 'down', 'neutral']
        self.n_actions = len(self.action_space)
        
        # Current state
        self.current_state = None
        self.last_prediction = None
        self.last_confidence = None
        
        logger.info(f"Trading environment initialized: state_dim={state_dim}, "
                   f"actions={self.action_space}")
    
    def encode_state(self, analysis_data: Dict) -> np.ndarray:
        """
        Convert analysis data into normalized state vector.
        
        Args:
            analysis_data: Complete analysis results
            
        Returns:
            Normalized state vector (values 0-1)
        """
        state_values = []
        
        # Technical indicators (15 features)
        technical = analysis_data.get('technical_analysis', {})
        
        # Extract RSI
        momentum = technical.get('momentum', {})
        rsi_data = momentum.get('rsi', {}) if isinstance(momentum, dict) else {}
        rsi = rsi_data.get('value', 50) if isinstance(rsi_data, dict) else 50
        
        # Extract MACD values - handle both numeric and dict formats
        macd_data = momentum.get('macd', {}) if isinstance(momentum, dict) else {}
        if isinstance(macd_data, dict):
            macd_value = macd_data.get('value', 0)
            # Extract numeric value if it exists
            if hasattr(macd_value, 'item'):  # numpy scalar
                macd = float(macd_value.item())
            elif isinstance(macd_value, (int, float)):
                macd = float(macd_value)
            else:
                macd = 0
            
            # Signal might be nested or a string
            macd_signal_val = macd_data.get('signal', 0)
            if isinstance(macd_signal_val, str):
                signal = 0  # Can't use string signal
            elif hasattr(macd_signal_val, 'item'):
                signal = float(macd_signal_val.item())
            elif isinstance(macd_signal_val, (int, float)):
                signal = float(macd_signal_val)
            else:
                signal = 0
        else:
            macd = 0
            signal = 0
        
        # Extract Bollinger Bands
        volatility = technical.get('volatility', {})
        bb_position = volatility.get('bb_position', 0.5) if isinstance(volatility, dict) else 0.5
        atr = volatility.get('atr', 0) if isinstance(volatility, dict) else 0
        
        # Extract Moving Averages from indicators
        indicators = technical.get('indicators', {})
        trend_indicators = indicators.get('trend_indicators', {}) if isinstance(indicators, dict) else {}
        
        # Get last valid SMA values (not NaN)
        def get_last_valid(series, default=0):
            """Get last non-NaN value from pandas Series or return default."""
            if series is None:
                return default
            try:
                # If it's a pandas Series
                if hasattr(series, 'dropna'):
                    valid = series.dropna()
                    return float(valid.iloc[-1]) if len(valid) > 0 else default
                # If it's already a number
                return float(series)
            except:
                return default
        
        sma_20 = get_last_valid(trend_indicators.get('sma_20'), 0)
        sma_50 = get_last_valid(trend_indicators.get('sma_50'), 0)
        sma_100 = get_last_valid(trend_indicators.get('sma_100'), 0)
        sma_200 = get_last_valid(trend_indicators.get('sma_200'), 0)
        
        ema_12 = get_last_valid(trend_indicators.get('ema_12'), 0)
        ema_26 = get_last_valid(trend_indicators.get('ema_26'), 0)
        ema_50 = get_last_valid(trend_indicators.get('ema_50'), 0)
        
        # Volume indicators
        volume_indicators = indicators.get('volume_indicators', {}) if isinstance(indicators, dict) else {}
        volume_sma = get_last_valid(volume_indicators.get('volume_sma'), 0)
        
        # Trend strength
        trend = technical.get('trend', {})
        trend_strength = trend.get('strength', 0.5) if isinstance(trend, dict) else 0.5
        
        state_values.extend([
            self._normalize(rsi, 0, 100),
            self._normalize(macd, -100, 100),
            self._normalize(signal, -100, 100),
            self._normalize(bb_position, 0, 1),
            self._normalize(atr, 0, 1000),
            self._normalize(sma_20, 0, 100000),
            self._normalize(sma_50, 0, 100000),
            self._normalize(sma_100, 0, 100000),
            self._normalize(sma_200, 0, 100000),
            self._normalize(ema_12, 0, 100000),
            self._normalize(ema_26, 0, 100000),
            self._normalize(ema_50, 0, 100000),
            self._normalize(volume_sma, 0, 1e9),
            self._normalize(trend_strength, 0, 1),
            self._normalize(0.5, 0, 1)  # Placeholder for momentum
        ])
        
        # Sentiment indicators (8 features) - handle nested dict structure
        sentiment = analysis_data.get('sentiment_analysis', {})
        
        # Extract news sentiment value
        news_sent_data = sentiment.get('news_sentiment', {})
        if isinstance(news_sent_data, dict):
            news_sent_val = news_sent_data.get('overall_sentiment', 0)
        else:
            news_sent_val = news_sent_data if isinstance(news_sent_data, (int, float)) else 0
        
        # Extract social sentiment value
        social_sent_data = sentiment.get('social_sentiment', {})
        if isinstance(social_sent_data, dict):
            social_sent_val = social_sent_data.get('overall_sentiment', 0)
        else:
            social_sent_val = social_sent_data if isinstance(social_sent_data, (int, float)) else 0
        
        # Extract overall sentiment
        overall_sentiment = sentiment.get('overall_sentiment', 0)
        if isinstance(overall_sentiment, (int, float)):
            overall_sent_val = overall_sentiment
        else:
            overall_sent_val = 0
        
        # Use actual values with safe defaults
        state_values.extend([
            self._normalize(overall_sent_val, -1, 1),
            self._normalize(news_sent_val, -1, 1),
            self._normalize(social_sent_val, -1, 1),
            self._normalize(0, -1, 1),  # sentiment_trend placeholder
            self._normalize(0, 0, 100),  # bullish_count placeholder
            self._normalize(0, 0, 100),  # bearish_count placeholder
            self._normalize(0, 0, 100),  # neutral_count placeholder
            self._normalize(0.5, 0, 1)  # sentiment_volatility placeholder
        ])
        
        # Fundamental indicators (12 features)
        fundamental = analysis_data.get('fundamental_analysis', {})
        state_values.extend([
            self._normalize(fundamental.get('market_cap', 0), 0, 1e12),
            self._normalize(fundamental.get('volume_24h', 0), 0, 1e11),
            self._normalize(fundamental.get('circulating_supply', 0), 0, 1e9),
            self._normalize(fundamental.get('max_supply', 0), 0, 1e9),
            self._normalize(fundamental.get('supply_ratio', 0.5), 0, 1),
            self._normalize(fundamental.get('tokenomics_score', 5), 0, 10),
            self._normalize(fundamental.get('technology_score', 5), 0, 10),
            self._normalize(fundamental.get('community_score', 5), 0, 10),
            self._normalize(fundamental.get('github_stars', 0), 0, 10000),
            self._normalize(fundamental.get('github_forks', 0), 0, 5000),
            self._normalize(fundamental.get('commits_30d', 0), 0, 500),
            self._normalize(fundamental.get('developers', 0), 0, 100)
        ])
        
        # Market conditions (10 features)
        price_data = analysis_data.get('current_price_data', {})
        
        # Extract price data safely
        current_price = price_data.get('price', 0)
        high_24h = price_data.get('high_24h', current_price)
        low_24h = price_data.get('low_24h', current_price)
        price_change_24h = price_data.get('price_change_24h', 0)
        price_change_pct_24h = price_data.get('price_change_pct_24h', 0)
        volume_24h = price_data.get('volume_24h', 0)
        
        # Calculate derived metrics
        volume_change_pct = 0  # Not always available
        volatility_val = volatility.get('atr_percentage', 0.5) if isinstance(volatility, dict) else 0.5
        range_pct = ((high_24h - low_24h) / current_price * 100) if current_price > 0 else 0
        liquidity_score = 0.5  # Default placeholder
        
        state_values.extend([
            self._normalize(current_price, 0, 100000),
            self._normalize(high_24h, 0, 100000),
            self._normalize(low_24h, 0, 100000),
            self._normalize(price_change_24h, -50, 50),
            self._normalize(price_change_pct_24h, -100, 100),
            self._normalize(volume_24h, 0, 1e11),
            self._normalize(volume_change_pct, -100, 100),
            self._normalize(volatility_val, 0, 1),
            self._normalize(range_pct, 0, 100),
            self._normalize(liquidity_score, 0, 1)
        ])
        
        # Time features (5 features)
        now = datetime.now()
        state_values.extend([
            self._normalize(now.hour, 0, 23),
            self._normalize(now.weekday(), 0, 6),
            self._normalize(now.day, 1, 31),
            self._normalize(now.month, 1, 12),
            self._normalize(now.isocalendar()[1], 1, 52)  # Week of year
        ])
        
        # Ensure we have exactly state_dim features
        if len(state_values) < self.state_dim:
            # Pad with zeros
            state_values.extend([0.0] * (self.state_dim - len(state_values)))
        elif len(state_values) > self.state_dim:
            # Truncate
            state_values = state_values[:self.state_dim]
        
        return np.array(state_values, dtype=np.float32)
    
    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize value to 0-1 range.
        
        Args:
            value: Value to normalize
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            Normalized value (0-1)
        """
        if max_val == min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def reset(self, initial_state: Dict) -> np.ndarray:
        """
        Reset environment with new market state.
        
        Args:
            initial_state: Initial analysis data
            
        Returns:
            Encoded state vector
        """
        self.current_state = self.encode_state(initial_state)
        self.last_prediction = None
        self.last_confidence = None
        return self.current_state
    
    def step(self, 
             action: int,
             confidence: float,
             actual_direction: str,
             actual_price: float,
             predicted_price: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return reward.
        
        Args:
            action: Action index (0=up, 1=down, 2=neutral)
            confidence: Prediction confidence (0-1)
            actual_direction: Actual price direction
            actual_price: Actual price after 24h
            predicted_price: Predicted price
            
        Returns:
            (next_state, reward, done, info)
        """
        predicted_direction = self.action_space[action]
        
        # Calculate reward
        reward = self._calculate_reward(
            predicted_direction,
            actual_direction,
            confidence,
            predicted_price,
            actual_price
        )
        
        # Store for analysis
        self.last_prediction = predicted_direction
        self.last_confidence = confidence
        
        # Episode is done (24h prediction cycle complete)
        done = True
        
        info = {
            'predicted_direction': predicted_direction,
            'actual_direction': actual_direction,
            'confidence': confidence,
            'reward': reward
        }
        
        return self.current_state, reward, done, info
    
    def _calculate_reward(self,
                         predicted_direction: str,
                         actual_direction: str,
                         confidence: float,
                         predicted_price: float,
                         actual_price: float) -> float:
        """
        Calculate reward for prediction.
        
        Reward components:
        1. Direction accuracy: +1 for correct, -1 for wrong
        2. Price accuracy: Bonus for accurate price prediction (0 to +0.5)
        3. Confidence calibration: Penalty for overconfidence (-0.3 to 0)
        
        Args:
            predicted_direction: Predicted direction
            actual_direction: Actual direction
            confidence: Prediction confidence
            predicted_price: Predicted price
            actual_price: Actual price
            
        Returns:
            Reward value (-1.5 to +1.5)
        """
        # Direction reward
        if predicted_direction == actual_direction:
            direction_reward = 1.0
        else:
            direction_reward = -1.0
        
        # Price accuracy bonus
        price_error = abs(predicted_price - actual_price) / actual_price
        price_bonus = max(0, 0.5 * (1.0 - price_error))
        
        # Confidence calibration penalty
        # Penalize high confidence when wrong, reward calibrated confidence
        if predicted_direction != actual_direction:
            # Wrong prediction - penalize high confidence
            confidence_penalty = -0.3 * confidence
        else:
            # Correct prediction - small bonus for confidence
            confidence_penalty = 0.1 * confidence
        
        total_reward = direction_reward + price_bonus + confidence_penalty
        
        logger.debug(f"Reward calculation: direction={direction_reward:.2f}, "
                    f"price_bonus={price_bonus:.2f}, "
                    f"confidence={confidence_penalty:.2f}, "
                    f"total={total_reward:.2f}")
        
        return total_reward
    
    def get_state_description(self, state: np.ndarray) -> Dict:
        """
        Convert state vector back to readable description.
        
        Args:
            state: State vector
            
        Returns:
            Dictionary describing state components
        """
        return {
            'technical': state[0:15].tolist(),
            'sentiment': state[15:23].tolist(),
            'fundamental': state[23:35].tolist(),
            'market': state[35:45].tolist(),
            'time': state[45:50].tolist()
        }
