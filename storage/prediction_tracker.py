"""
Prediction tracking system for monitoring and learning from prediction accuracy.
Stores predictions and verifies them after 24 hours to calculate model performance.
"""
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def serialize_for_json(obj):
    """Convert objects to JSON-serializable formats."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        # For Series, try to extract the last value or convert to list
        if len(obj) > 0:
            return float(obj.iloc[-1]) if len(obj) == 1 else obj.tolist()
        return None
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    else:
        return obj


class PredictionTracker:
    """Tracks predictions and their actual outcomes for RL learning."""
    
    def __init__(self, db_path: str = None):
        """
        Initialize prediction tracker.
        
        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "crypto_analysis.db"
        self.db_path = db_path
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Create predictions table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                predicted_price REAL,
                predicted_direction TEXT,
                confidence REAL,
                actual_price REAL,
                actual_direction TEXT,
                accuracy REAL,
                technical_state TEXT,
                sentiment_state TEXT,
                fundamental_state TEXT,
                market_conditions TEXT,
                verified INTEGER DEFAULT 0,
                verification_timestamp TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
            ON predictions(symbol, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_verified 
            ON predictions(verified)
        """)
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, 
                       symbol: str,
                       predicted_price: float,
                       predicted_direction: str,
                       confidence: float,
                       technical_state: Dict,
                       sentiment_state: Dict,
                       fundamental_state: Dict,
                       market_conditions: Dict) -> int:
        """
        Save a new prediction to track.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., BTCUSDT)
            predicted_price: Predicted price in 24h
            predicted_direction: 'up', 'down', or 'neutral'
            confidence: Confidence level (0-1)
            technical_state: Technical indicators at prediction time
            sentiment_state: Sentiment scores at prediction time
            fundamental_state: Fundamental metrics at prediction time
            market_conditions: Overall market state
            
        Returns:
            Prediction ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO predictions (
                symbol, timestamp, prediction_type, predicted_price,
                predicted_direction, confidence, technical_state,
                sentiment_state, fundamental_state, market_conditions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            timestamp,
            '24h_price',
            predicted_price,
            predicted_direction,
            confidence,
            json.dumps(serialize_for_json(technical_state)),
            json.dumps(serialize_for_json(sentiment_state)),
            json.dumps(serialize_for_json(fundamental_state)),
            json.dumps(serialize_for_json(market_conditions))
        ))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Saved prediction {prediction_id} for {symbol}: "
                   f"{predicted_direction} to ${predicted_price:.2f} "
                   f"(confidence: {confidence:.2%})")
        
        return prediction_id
    
    def get_unverified_predictions(self, hours_old: int = 24) -> List[Dict]:
        """
        Get predictions that are ready to be verified.
        
        Args:
            hours_old: Minimum age in hours for verification
            
        Returns:
            List of unverified predictions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = (datetime.now() - timedelta(hours=hours_old)).isoformat()
        
        cursor.execute("""
            SELECT id, symbol, timestamp, predicted_price, predicted_direction,
                   confidence, technical_state, sentiment_state, 
                   fundamental_state, market_conditions
            FROM predictions
            WHERE verified = 0 AND timestamp < ?
            ORDER BY timestamp ASC
        """, (cutoff_time,))
        
        predictions = []
        for row in cursor.fetchall():
            predictions.append({
                'id': row[0],
                'symbol': row[1],
                'timestamp': row[2],
                'predicted_price': row[3],
                'predicted_direction': row[4],
                'confidence': row[5],
                'technical_state': json.loads(row[6]),
                'sentiment_state': json.loads(row[7]),
                'fundamental_state': json.loads(row[8]),
                'market_conditions': json.loads(row[9])
            })
        
        conn.close()
        return predictions
    
    def verify_prediction(self, 
                         prediction_id: int,
                         actual_price: float,
                         actual_direction: str) -> float:
        """
        Verify a prediction against actual outcome.
        
        Args:
            prediction_id: ID of prediction to verify
            actual_price: Actual price after 24h
            actual_direction: Actual direction ('up', 'down', 'neutral')
            
        Returns:
            Accuracy score (0-1)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get prediction details
        cursor.execute("""
            SELECT predicted_price, predicted_direction, confidence
            FROM predictions WHERE id = ?
        """, (prediction_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return 0.0
        
        predicted_price, predicted_direction, confidence = row
        
        # Calculate accuracy
        # Direction accuracy: 1 if correct, 0 if wrong
        direction_correct = 1.0 if predicted_direction == actual_direction else 0.0
        
        # Price accuracy: inverse of percentage error (max 1.0)
        price_error = abs(predicted_price - actual_price) / actual_price
        price_accuracy = max(0, 1.0 - price_error)
        
        # Combined accuracy (weighted: 60% direction, 40% price)
        accuracy = 0.6 * direction_correct + 0.4 * price_accuracy
        
        # Update prediction record
        cursor.execute("""
            UPDATE predictions
            SET actual_price = ?,
                actual_direction = ?,
                accuracy = ?,
                verified = 1,
                verification_timestamp = ?
            WHERE id = ?
        """, (
            actual_price,
            actual_direction,
            accuracy,
            datetime.now().isoformat(),
            prediction_id
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Verified prediction {prediction_id}: "
                   f"accuracy={accuracy:.2%}, "
                   f"predicted=${predicted_price:.2f} {predicted_direction}, "
                   f"actual=${actual_price:.2f} {actual_direction}")
        
        return accuracy
    
    def get_performance_stats(self, 
                             symbol: Optional[str] = None,
                             days: int = 30) -> Dict:
        """
        Get performance statistics for predictions.
        
        Args:
            symbol: Filter by symbol (None for all)
            days: Number of days to analyze
            
        Returns:
            Performance statistics dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
        
        if symbol:
            cursor.execute("""
                SELECT AVG(accuracy), COUNT(*), 
                       AVG(CASE WHEN predicted_direction = actual_direction THEN 1 ELSE 0 END)
                FROM predictions
                WHERE verified = 1 AND timestamp > ? AND symbol = ?
            """, (cutoff_time, symbol))
        else:
            cursor.execute("""
                SELECT AVG(accuracy), COUNT(*),
                       AVG(CASE WHEN predicted_direction = actual_direction THEN 1 ELSE 0 END)
                FROM predictions
                WHERE verified = 1 AND timestamp > ?
            """, (cutoff_time,))
        
        row = cursor.fetchone()
        avg_accuracy = row[0] or 0.0
        total_predictions = row[1] or 0
        direction_accuracy = row[2] or 0.0
        
        conn.close()
        
        return {
            'average_accuracy': avg_accuracy,
            'total_predictions': total_predictions,
            'direction_accuracy': direction_accuracy,
            'symbol': symbol or 'all',
            'period_days': days
        }
    
    def get_training_data(self, 
                         symbol: Optional[str] = None,
                         limit: int = 1000) -> List[Tuple]:
        """
        Get verified predictions as training data for RL.
        
        Args:
            symbol: Filter by symbol (None for all)
            limit: Maximum number of records
            
        Returns:
            List of (state, action, reward, next_state) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if symbol:
            cursor.execute("""
                SELECT technical_state, sentiment_state, fundamental_state,
                       market_conditions, predicted_direction, accuracy
                FROM predictions
                WHERE verified = 1 AND symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (symbol, limit))
        else:
            cursor.execute("""
                SELECT technical_state, sentiment_state, fundamental_state,
                       market_conditions, predicted_direction, accuracy
                FROM predictions
                WHERE verified = 1
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
        
        training_data = []
        for row in cursor.fetchall():
            technical = json.loads(row[0])
            sentiment = json.loads(row[1])
            fundamental = json.loads(row[2])
            market = json.loads(row[3])
            action = row[4]
            reward = row[5]
            
            # Combine into state vector
            state = {
                'technical': technical,
                'sentiment': sentiment,
                'fundamental': fundamental,
                'market': market
            }
            
            training_data.append((state, action, reward))
        
        conn.close()
        return training_data
