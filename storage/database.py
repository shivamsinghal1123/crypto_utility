"""
Database Module
Manages local SQLite database for storing analysis data
"""

import sqlite3
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from config.settings import DATA_STORAGE

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages local SQLite database operations."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DATA_STORAGE['DATABASE_PATH']
        self.conn = None
        self.setup_database()
    
    def connect(self):
        """Establish database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            return self.conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            return None
    
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def setup_database(self):
        """Create database tables if they don't exist."""
        conn = self.connect()
        if not conn:
            return
        
        cursor = conn.cursor()
        
        # Price data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                interval TEXT,
                UNIQUE(symbol, timestamp, interval)
            )
        ''')
        
        # News data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                title TEXT,
                summary TEXT,
                link TEXT UNIQUE,
                source TEXT,
                published DATETIME,
                sentiment REAL,
                relevance REAL,
                credibility_score REAL
            )
        ''')
        
        # Fundamental data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fundamental_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                data JSON,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        # Analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                analysis_type TEXT,
                results JSON,
                UNIQUE(symbol, timestamp, analysis_type)
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                prediction_time DATETIME NOT NULL,
                target_time DATETIME,
                predicted_price REAL,
                actual_price REAL,
                prediction_data JSON
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                metric_name TEXT,
                metric_value REAL,
                metadata JSON
            )
        ''')
        
        conn.commit()
        self.disconnect()
        logger.info("Database setup completed")
    
    def save_price_data(self, symbol: str, df) -> bool:
        """
        Save OHLCV data to database.
        
        Args:
            symbol: Trading pair symbol
            df: DataFrame with OHLCV data
        
        Returns:
            True if successful, False otherwise
        """
        conn = self.connect()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO price_data 
                    (symbol, timestamp, open, high, low, close, volume, interval)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    str(row['timestamp']),  # Convert Timestamp to string
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume']),
                    '1h'  # Default interval
                ))
            
            conn.commit()
            logger.info(f"Saved {len(df)} price records for {symbol}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error saving price data: {e}")
            return False
        finally:
            self.disconnect()
    
    def save_news_data(self, symbol: str, news_list: List[Dict]) -> bool:
        """Save news data to database."""
        conn = self.connect()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            for news in news_list:
                cursor.execute('''
                    INSERT OR IGNORE INTO news_data 
                    (symbol, title, summary, link, source, published, sentiment, relevance, credibility_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    news.get('title'),
                    news.get('summary'),
                    news.get('link'),
                    news.get('source'),
                    news.get('published'),
                    news.get('sentiment'),
                    news.get('relevance'),
                    news.get('credibility_score')
                ))
            
            conn.commit()
            logger.info(f"Saved {len(news_list)} news records for {symbol}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error saving news data: {e}")
            return False
        finally:
            self.disconnect()
    
    def save_analysis_results(self, symbol: str, analysis_type: str, results: Dict) -> bool:
        """Save analysis results to database."""
        conn = self.connect()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO analysis_results 
                (symbol, timestamp, analysis_type, results)
                VALUES (?, ?, ?, ?)
            ''', (
                symbol,
                datetime.now(),
                analysis_type,
                json.dumps(results, default=str)
            ))
            
            conn.commit()
            logger.info(f"Saved {analysis_type} analysis for {symbol}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error saving analysis results: {e}")
            return False
        finally:
            self.disconnect()
    
    def get_latest_analysis(self, symbol: str, analysis_type: str) -> Optional[Dict]:
        """Retrieve latest analysis results."""
        conn = self.connect()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT results FROM analysis_results 
                WHERE symbol = ? AND analysis_type = ?
                ORDER BY timestamp DESC LIMIT 1
            ''', (symbol, analysis_type))
            
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving analysis: {e}")
            return None
        finally:
            self.disconnect()
    
    def save_prediction(self, symbol: str, prediction_data: Dict) -> bool:
        """Save prediction data for backtesting."""
        conn = self.connect()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions 
                (symbol, prediction_time, target_time, predicted_price, prediction_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                symbol,
                datetime.now(),
                prediction_data.get('target_time'),
                prediction_data.get('predicted_price'),
                json.dumps(prediction_data, default=str)
            ))
            
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error saving prediction: {e}")
            return False
        finally:
            self.disconnect()
