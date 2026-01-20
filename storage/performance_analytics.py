"""
Performance analytics for tracking and analyzing prediction accuracy over time.
Provides insights into model performance, learning curves, and improvement trends.
"""
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PerformanceAnalytics:
    """Analyzes prediction performance and learning progress."""
    
    def __init__(self, db_path: str = None):
        """
        Initialize performance analytics.
        
        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "crypto_analysis.db"
        self.db_path = db_path
    
    def get_accuracy_trend(self, 
                          symbol: str,
                          days: int = 30,
                          interval: str = 'daily') -> List[Dict]:
        """
        Get accuracy trend over time.
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days to analyze
            interval: 'daily' or 'weekly'
            
        Returns:
            List of {date, accuracy, count} dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
        
        if interval == 'daily':
            group_format = '%Y-%m-%d'
        else:
            group_format = '%Y-W%W'
        
        cursor.execute(f"""
            SELECT strftime('{group_format}', timestamp) as period,
                   AVG(accuracy) as avg_accuracy,
                   COUNT(*) as prediction_count
            FROM predictions
            WHERE verified = 1 
              AND symbol = ?
              AND timestamp > ?
            GROUP BY period
            ORDER BY period ASC
        """, (symbol, cutoff_time))
        
        trend = []
        for row in cursor.fetchall():
            trend.append({
                'period': row[0],
                'accuracy': row[1],
                'count': row[2]
            })
        
        conn.close()
        return trend
    
    def get_confidence_calibration(self, symbol: str) -> Dict:
        """
        Analyze how well confidence scores match actual accuracy.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Calibration analysis dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Group predictions by confidence buckets
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN confidence < 0.5 THEN '0-50%'
                    WHEN confidence < 0.7 THEN '50-70%'
                    WHEN confidence < 0.85 THEN '70-85%'
                    ELSE '85-100%'
                END as confidence_bucket,
                AVG(accuracy) as avg_accuracy,
                AVG(confidence) as avg_confidence,
                COUNT(*) as count
            FROM predictions
            WHERE verified = 1 AND symbol = ?
            GROUP BY confidence_bucket
            ORDER BY avg_confidence ASC
        """, (symbol,))
        
        calibration = {
            'buckets': [],
            'is_well_calibrated': True,
            'calibration_error': 0.0
        }
        
        total_error = 0.0
        total_count = 0
        
        for row in cursor.fetchall():
            bucket_data = {
                'range': row[0],
                'avg_accuracy': row[1],
                'avg_confidence': row[2],
                'count': row[3],
                'gap': abs(row[1] - row[2])
            }
            calibration['buckets'].append(bucket_data)
            
            # Calculate calibration error
            total_error += bucket_data['gap'] * row[3]
            total_count += row[3]
        
        if total_count > 0:
            calibration['calibration_error'] = total_error / total_count
            calibration['is_well_calibrated'] = calibration['calibration_error'] < 0.1
        
        conn.close()
        return calibration
    
    def get_best_performing_conditions(self, 
                                      symbol: str,
                                      top_n: int = 5) -> List[Dict]:
        """
        Find market conditions where predictions perform best.
        
        Args:
            symbol: Cryptocurrency symbol
            top_n: Number of top conditions to return
            
        Returns:
            List of best performing condition patterns
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT market_conditions, technical_state,
                   AVG(accuracy) as avg_accuracy,
                   COUNT(*) as count
            FROM predictions
            WHERE verified = 1 AND symbol = ?
            GROUP BY market_conditions
            HAVING count >= 3
            ORDER BY avg_accuracy DESC
            LIMIT ?
        """, (symbol, top_n))
        
        conditions = []
        for row in cursor.fetchall():
            market = json.loads(row[0])
            technical = json.loads(row[1])
            
            conditions.append({
                'market_conditions': market,
                'technical_pattern': technical,
                'avg_accuracy': row[2],
                'sample_size': row[3]
            })
        
        conn.close()
        return conditions
    
    def get_worst_performing_conditions(self,
                                       symbol: str,
                                       bottom_n: int = 5) -> List[Dict]:
        """
        Find market conditions where predictions perform worst.
        
        Args:
            symbol: Cryptocurrency symbol
            bottom_n: Number of worst conditions to return
            
        Returns:
            List of worst performing condition patterns
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT market_conditions, technical_state,
                   AVG(accuracy) as avg_accuracy,
                   COUNT(*) as count
            FROM predictions
            WHERE verified = 1 AND symbol = ?
            GROUP BY market_conditions
            HAVING count >= 3
            ORDER BY avg_accuracy ASC
            LIMIT ?
        """, (symbol, bottom_n))
        
        conditions = []
        for row in cursor.fetchall():
            market = json.loads(row[0])
            technical = json.loads(row[1])
            
            conditions.append({
                'market_conditions': market,
                'technical_pattern': technical,
                'avg_accuracy': row[2],
                'sample_size': row[3]
            })
        
        conn.close()
        return conditions
    
    def get_learning_velocity(self, symbol: str) -> Dict:
        """
        Calculate how fast the model is improving.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Learning velocity metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get first 50 predictions accuracy
        cursor.execute("""
            SELECT AVG(accuracy)
            FROM (
                SELECT accuracy
                FROM predictions
                WHERE verified = 1 AND symbol = ?
                ORDER BY timestamp ASC
                LIMIT 50
            )
        """, (symbol,))
        early_accuracy = cursor.fetchone()[0] or 0.0
        
        # Get last 50 predictions accuracy
        cursor.execute("""
            SELECT AVG(accuracy)
            FROM (
                SELECT accuracy
                FROM predictions
                WHERE verified = 1 AND symbol = ?
                ORDER BY timestamp DESC
                LIMIT 50
            )
        """, (symbol,))
        recent_accuracy = cursor.fetchone()[0] or 0.0
        
        # Get total verified predictions
        cursor.execute("""
            SELECT COUNT(*)
            FROM predictions
            WHERE verified = 1 AND symbol = ?
        """, (symbol,))
        total_predictions = cursor.fetchone()[0]
        
        conn.close()
        
        improvement = recent_accuracy - early_accuracy
        improvement_rate = (improvement / early_accuracy * 100) if early_accuracy > 0 else 0
        
        return {
            'early_accuracy': early_accuracy,
            'recent_accuracy': recent_accuracy,
            'improvement': improvement,
            'improvement_rate': improvement_rate,
            'total_predictions': total_predictions,
            'is_improving': improvement > 0
        }
    
    def generate_performance_report(self, symbol: str) -> Dict:
        """
        Generate comprehensive performance report.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Complete performance analysis
        """
        report = {
            'symbol': symbol,
            'generated_at': datetime.now().isoformat(),
            'accuracy_trend_30d': self.get_accuracy_trend(symbol, days=30),
            'confidence_calibration': self.get_confidence_calibration(symbol),
            'best_conditions': self.get_best_performing_conditions(symbol),
            'worst_conditions': self.get_worst_performing_conditions(symbol),
            'learning_velocity': self.get_learning_velocity(symbol)
        }
        
        return report
    
    def get_feature_importance(self, symbol: str) -> Dict:
        """
        Analyze which features correlate most with accuracy.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Feature importance scores
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT technical_state, sentiment_state, fundamental_state, accuracy
            FROM predictions
            WHERE verified = 1 AND symbol = ?
        """, (symbol,))
        
        # Collect feature correlations
        feature_scores = {
            'technical_features': {},
            'sentiment_features': {},
            'fundamental_features': {}
        }
        
        predictions = cursor.fetchall()
        conn.close()
        
        if not predictions:
            return feature_scores
        
        # Simple correlation analysis
        for row in predictions:
            technical = json.loads(row[0])
            sentiment = json.loads(row[1])
            fundamental = json.loads(row[2])
            accuracy = row[3]
            
            # Track technical features
            for key, value in technical.items():
                if key not in feature_scores['technical_features']:
                    feature_scores['technical_features'][key] = []
                if isinstance(value, (int, float)):
                    feature_scores['technical_features'][key].append((value, accuracy))
            
            # Track sentiment features
            for key, value in sentiment.items():
                if key not in feature_scores['sentiment_features']:
                    feature_scores['sentiment_features'][key] = []
                if isinstance(value, (int, float)):
                    feature_scores['sentiment_features'][key].append((value, accuracy))
            
            # Track fundamental features
            for key, value in fundamental.items():
                if key not in feature_scores['fundamental_features']:
                    feature_scores['fundamental_features'][key] = []
                if isinstance(value, (int, float)):
                    feature_scores['fundamental_features'][key].append((value, accuracy))
        
        return feature_scores
