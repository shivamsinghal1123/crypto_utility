"""
Pipeline Orchestration Module
Unified entry point for crypto analysis workflow
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """
    Orchestrates the full crypto analysis workflow.
    This wraps the existing CryptoAnalyzer logic for scheduled execution.
    """
    
    def __init__(self, analyzer):
        """
        Initialize the pipeline with an existing CryptoAnalyzer instance.
        
        Args:
            analyzer: CryptoAnalyzer instance with all components initialized
        """
        self.analyzer = analyzer
        logger.info("Analysis Pipeline initialized")
    
    def run_pipeline(self, symbol: str, interval: str = "1h", generate_charts: bool = True) -> Optional[dict]:
        """
        Execute the complete analysis pipeline for a single symbol.
        
        This function is idempotent and safe to call repeatedly on a schedule.
        
        Args:
            symbol: Binance trading pair symbol (e.g., 'BTCUSDT')
            interval: Binance kline interval for price data (e.g., '1m', '5m', '15m', '1h', '4h', '1d')
            generate_charts: Whether to generate visualization charts
        
        Returns:
            Dictionary with analysis results or None if failed
        """
        start_ts = datetime.utcnow()
        symbol = symbol.upper()
        
        logger.info(f"[{symbol}] Pipeline start at {start_ts.isoformat()} interval={interval}")
        
        try:
            # Run the full analysis using existing CryptoAnalyzer logic
            results = self.analyzer.analyze_cryptocurrency(
                symbol=symbol,
                generate_charts=generate_charts
            )
            
            if results:
                end_ts = datetime.utcnow()
                duration = (end_ts - start_ts).total_seconds()
                logger.info(f"[{symbol}] Pipeline completed successfully in {duration:.2f}s")
                print(f"\n✅ [{symbol}] Analysis completed - Next run in scheduled interval")
                return results
            else:
                logger.warning(f"[{symbol}] Pipeline failed - no results returned (will retry on next schedule)")
                print(f"\n⚠️  [{symbol}] Analysis failed - Will retry on next scheduled run")
                return None
                
        except Exception as e:
            logger.error(f"[{symbol}] Pipeline error: {str(e)}", exc_info=True)
            print(f"\n❌ [{symbol}] Error: {str(e)} - Will retry on next scheduled run")
            return None
    
    def run_pipeline_quick(self, symbol: str) -> Optional[dict]:
        """
        Run a quick analysis pipeline without heavy processing.
        
        Args:
            symbol: Binance trading pair symbol
            
        Returns:
            Dictionary with analysis results or None if failed
        """
        start_ts = datetime.utcnow()
        symbol = symbol.upper()
        
        logger.info(f"[{symbol}] Quick pipeline start at {start_ts.isoformat()}")
        
        try:
            # Collect data
            collected_data = self.analyzer.collect_data(symbol)
            
            if not collected_data:
                logger.error(f"[{symbol}] Quick pipeline failed - no data collected")
                return None
            
            # Perform lightweight analysis
            results = {
                'symbol': symbol,
                'timestamp': start_ts,
                'price_data': collected_data.get('current_price_data'),
                'ohlcv_data': collected_data.get('price_data_df'),
            }
            
            end_ts = datetime.utcnow()
            duration = (end_ts - start_ts).total_seconds()
            logger.info(f"[{symbol}] Quick pipeline completed in {duration:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"[{symbol}] Quick pipeline error: {str(e)}", exc_info=True)
            return None
