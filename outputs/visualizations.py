"""
Visualizations Module
Creates charts and graphs for analysis results
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import logging

from config.settings import OUTPUT_SETTINGS

logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use(OUTPUT_SETTINGS.get('CHART_STYLE', 'seaborn'))


class ChartGenerator:
    """Generates visualizations for cryptocurrency analysis."""
    
    def __init__(self):
        self.settings = OUTPUT_SETTINGS
        self.reports_dir = Path(self.settings['REPORTS_DIR'])
        self.session_folder = None  # Will be set by ReportGenerator
    
    def set_session_folder(self, session_folder: Path):
        """Set the session folder for saving charts."""
        self.session_folder = session_folder
    
    def create_price_chart(self, df: pd.DataFrame, symbol: str, 
                          sr_levels: Dict = None, save: bool = True) -> Optional[str]:
        """
        Create price chart with support/resistance levels.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Cryptocurrency symbol
            sr_levels: Support and resistance levels
            save: Whether to save the chart
        
        Returns:
            Path to saved chart or None
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                           gridspec_kw={'height_ratios': [3, 1]})
            
            # Price chart
            ax1.plot(df['timestamp'], df['close'], label='Close Price', linewidth=2)
            
            # Add moving averages
            if len(df) >= 20:
                sma_20 = df['close'].rolling(window=20).mean()
                ax1.plot(df['timestamp'], sma_20, label='SMA 20', alpha=0.7)
            
            if len(df) >= 50:
                sma_50 = df['close'].rolling(window=50).mean()
                ax1.plot(df['timestamp'], sma_50, label='SMA 50', alpha=0.7)
            
            # Add support and resistance levels
            if sr_levels:
                current_price = df['close'].iloc[-1]
                
                # Support levels
                for level in sr_levels.get('next_24h_support', [])[:3]:
                    ax1.axhline(y=level['level'], color='green', linestyle='--', 
                               alpha=0.5, label=f"Support: {level['level']}")
                
                # Resistance levels
                for level in sr_levels.get('next_24h_resistance', [])[:3]:
                    ax1.axhline(y=level['level'], color='red', linestyle='--', 
                               alpha=0.5, label=f"Resistance: {level['level']}")
            
            ax1.set_title(f'{symbol} Price Chart', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Price (USDT)', fontsize=12)
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Volume chart
            colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] 
                     else 'red' for i in range(len(df))]
            ax2.bar(df['timestamp'], df['volume'], color=colors, alpha=0.6)
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.set_xlabel('Time', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            if save:
                filename = "price_chart.png"
                filepath = self.session_folder / filename if self.session_folder else self.reports_dir / filename
                plt.savefig(filepath, dpi=self.settings['CHART_DPI'], bbox_inches='tight')
                logger.info(f"Price chart saved to {filepath}")
                plt.close()
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error creating price chart: {e}")
            return None
    
    def create_technical_indicators_chart(self, df: pd.DataFrame, indicators: Dict, 
                                         symbol: str, save: bool = True) -> Optional[str]:
        """
        Create chart with technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            indicators: Dictionary with technical indicators
            symbol: Cryptocurrency symbol
            save: Whether to save the chart
        
        Returns:
            Path to saved chart or None
        """
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
            
            # Price with Bollinger Bands
            ax1.plot(df['timestamp'], df['close'], label='Close Price', linewidth=2)
            
            if 'trend_indicators' in indicators and 'bollinger_bands' in indicators['trend_indicators']:
                bb = indicators['trend_indicators']['bollinger_bands']
                ax1.plot(df['timestamp'], bb['upper'], 'r--', alpha=0.5, label='BB Upper')
                ax1.plot(df['timestamp'], bb['middle'], 'b--', alpha=0.5, label='BB Middle')
                ax1.plot(df['timestamp'], bb['lower'], 'g--', alpha=0.5, label='BB Lower')
                ax1.fill_between(df['timestamp'], bb['upper'], bb['lower'], alpha=0.1)
            
            ax1.set_title(f'{symbol} - Technical Indicators', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Price (USDT)', fontsize=12)
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # RSI
            if 'momentum_indicators' in indicators and 'rsi' in indicators['momentum_indicators']:
                rsi = indicators['momentum_indicators']['rsi']
                ax2.plot(df['timestamp'], rsi, label='RSI', color='purple', linewidth=2)
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
                ax2.fill_between(df['timestamp'], 30, 70, alpha=0.1)
                ax2.set_ylabel('RSI', fontsize=12)
                ax2.set_ylim(0, 100)
                ax2.legend(loc='best')
                ax2.grid(True, alpha=0.3)
            
            # MACD
            if 'trend_indicators' in indicators and 'macd' in indicators['trend_indicators']:
                macd = indicators['trend_indicators']['macd']
                ax3.plot(df['timestamp'], macd['macd'], label='MACD', linewidth=2)
                ax3.plot(df['timestamp'], macd['signal'], label='Signal', linewidth=2)
                ax3.bar(df['timestamp'], macd['histogram'], label='Histogram', alpha=0.3)
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax3.set_ylabel('MACD', fontsize=12)
                ax3.set_xlabel('Time', fontsize=12)
                ax3.legend(loc='best')
                ax3.grid(True, alpha=0.3)
            
            # Format x-axis
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            if save:
                filename = "indicators_chart.png"
                filepath = self.session_folder / filename if self.session_folder else self.reports_dir / filename
                plt.savefig(filepath, dpi=self.settings['CHART_DPI'], bbox_inches='tight')
                logger.info(f"Indicators chart saved to {filepath}")
                plt.close()
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error creating indicators chart: {e}")
            return None
    
    def create_sentiment_chart(self, sentiment_data: Dict, symbol: str, 
                              save: bool = True) -> Optional[str]:
        """
        Create sentiment visualization.
        
        Args:
            sentiment_data: Sentiment analysis data
            symbol: Cryptocurrency symbol
            save: Whether to save the chart
        
        Returns:
            Path to saved chart or None
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Overall sentiment gauge
            overall_sentiment = sentiment_data.get('overall_sentiment', 0)
            
            # Create gauge
            categories = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
            colors = ['darkred', 'red', 'gray', 'green', 'darkgreen']
            
            # Map sentiment to category
            if overall_sentiment < -0.6:
                cat_idx = 0
            elif overall_sentiment < -0.2:
                cat_idx = 1
            elif overall_sentiment < 0.2:
                cat_idx = 2
            elif overall_sentiment < 0.6:
                cat_idx = 3
            else:
                cat_idx = 4
            
            ax1.barh(categories, [1]*5, color=['lightgray']*5)
            ax1.barh(categories[cat_idx], 1, color=colors[cat_idx])
            ax1.set_xlabel('Sentiment Level', fontsize=12)
            ax1.set_title(f'{symbol} Overall Sentiment\nScore: {overall_sentiment:.2f}', 
                         fontsize=14, fontweight='bold')
            
            # Sentiment breakdown
            news_sent = sentiment_data.get('news_sentiment', {}).get('overall_sentiment', 0)
            social_sent = sentiment_data.get('social_sentiment', {}).get('overall_sentiment', 0)
            
            sources = ['News', 'Social Media']
            sentiments = [news_sent, social_sent]
            bar_colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in sentiments]
            
            ax2.bar(sources, sentiments, color=bar_colors, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_ylabel('Sentiment Score', fontsize=12)
            ax2.set_ylim(-1, 1)
            ax2.set_title('Sentiment by Source', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            if save:
                filename = "sentiment_chart.png"
                filepath = self.session_folder / filename if self.session_folder else self.reports_dir / filename
                plt.savefig(filepath, dpi=self.settings['CHART_DPI'], bbox_inches='tight')
                logger.info(f"Sentiment chart saved to {filepath}")
                plt.close()
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error creating sentiment chart: {e}")
            return None
    
    def create_fundamental_radar(self, fundamental_data: Dict, symbol: str, 
                                save: bool = True) -> Optional[str]:
        """
        Create radar chart for fundamental scores.
        
        Args:
            fundamental_data: Fundamental analysis data
            symbol: Cryptocurrency symbol
            save: Whether to save the chart
        
        Returns:
            Path to saved chart or None
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Extract scores
            tokenomics = fundamental_data.get('tokenomics', {})
            fundamentals = fundamental_data.get('fundamentals', {})
            
            categories = [
                'Utility',
                'Scarcity',
                'Technology',
                'Team',
                'Community'
            ]
            
            values = [
                tokenomics.get('utility_score', 0),
                tokenomics.get('scarcity_score', 0),
                fundamentals.get('technology_score', 0),
                fundamentals.get('team_score', 0),
                fundamentals.get('community_score', 0)
            ]
            
            # Close the plot
            values += values[:1]
            
            # Angles for each category
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label=symbol)
            ax.fill(angles, values, alpha=0.25)
            
            # Fix axis
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 10)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.set_yticklabels(['2', '4', '6', '8', '10'])
            ax.grid(True)
            
            plt.title(f'{symbol} Fundamental Analysis Scores', 
                     fontsize=16, fontweight='bold', pad=20)
            
            if save:
                filename = "fundamental_radar.png"
                filepath = self.session_folder / filename if self.session_folder else self.reports_dir / filename
                plt.savefig(filepath, dpi=self.settings['CHART_DPI'], bbox_inches='tight')
                logger.info(f"Fundamental radar chart saved to {filepath}")
                plt.close()
                return str(filepath)
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error creating fundamental radar chart: {e}")
            return None
