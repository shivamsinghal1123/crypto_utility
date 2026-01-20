"""
WebSocket client for real-time cryptocurrency data streaming.
Enables continuous price monitoring and automatic RL training triggers.
"""
import json
import asyncio
import websockets
from typing import Dict, Callable, Optional, List
from datetime import datetime
import logging
import threading

logger = logging.getLogger(__name__)


class BinanceWebSocketClient:
    """
    WebSocket client for real-time Binance price feeds.
    Supports multiple symbols and automatic reconnection.
    """
    
    def __init__(self, symbols: List[str] = None):
        """
        Initialize WebSocket client.
        
        Args:
            symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        """
        self.symbols = symbols or []
        self.ws_url = "wss://stream.binance.com:9443/ws"
        self.callbacks = {}
        self.running = False
        self.websocket = None
        self.thread = None
        
        logger.info(f"WebSocket client initialized for symbols: {symbols}")
    
    def subscribe_ticker(self, symbol: str, callback: Callable):
        """
        Subscribe to ticker updates for a symbol.
        
        Args:
            symbol: Trading pair symbol
            callback: Function to call with ticker updates
        """
        symbol_lower = symbol.lower()
        stream_name = f"{symbol_lower}@ticker"
        self.callbacks[stream_name] = callback
        
        if symbol not in self.symbols:
            self.symbols.append(symbol)
        
        logger.info(f"Subscribed to {symbol} ticker updates")
    
    def subscribe_kline(self, symbol: str, interval: str, callback: Callable):
        """
        Subscribe to candlestick updates for a symbol.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
            callback: Function to call with kline updates
        """
        symbol_lower = symbol.lower()
        stream_name = f"{symbol_lower}@kline_{interval}"
        self.callbacks[stream_name] = callback
        
        if symbol not in self.symbols:
            self.symbols.append(symbol)
        
        logger.info(f"Subscribed to {symbol} {interval} kline updates")
    
    def subscribe_depth(self, symbol: str, callback: Callable):
        """
        Subscribe to order book depth updates.
        
        Args:
            symbol: Trading pair symbol
            callback: Function to call with depth updates
        """
        symbol_lower = symbol.lower()
        stream_name = f"{symbol_lower}@depth"
        self.callbacks[stream_name] = callback
        
        if symbol not in self.symbols:
            self.symbols.append(symbol)
        
        logger.info(f"Subscribed to {symbol} depth updates")
    
    async def _connect(self):
        """Establish WebSocket connection and listen for messages."""
        # Build stream URL
        if not self.callbacks:
            logger.warning("No subscriptions configured")
            return
        
        streams = list(self.callbacks.keys())
        stream_url = f"{self.ws_url}/{'/'.join(streams)}"
        
        logger.info(f"Connecting to {stream_url}")
        
        try:
            async with websockets.connect(stream_url) as websocket:
                self.websocket = websocket
                logger.info("WebSocket connected")
                
                while self.running:
                    try:
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=30.0
                        )
                        
                        data = json.loads(message)
                        await self._handle_message(data)
                        
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.ping()
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                    except Exception as e:
                        logger.error(f"Message handling error: {e}")
                        
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
            if self.running:
                logger.info("Attempting to reconnect in 5 seconds...")
                await asyncio.sleep(5)
                await self._connect()
        except Exception as e:
            logger.error(f"Connection error: {e}")
    
    async def _handle_message(self, data: Dict):
        """
        Handle incoming WebSocket message.
        
        Args:
            data: Parsed JSON message
        """
        if 'stream' in data:
            stream = data['stream']
            payload = data['data']
        else:
            # Single stream format
            stream = data.get('e', '')
            payload = data
        
        # Find matching callback
        for stream_name, callback in self.callbacks.items():
            if stream_name in stream or stream in stream_name:
                try:
                    # Run callback in thread to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, callback, payload)
                except Exception as e:
                    logger.error(f"Callback error for {stream_name}: {e}")
    
    def start(self):
        """Start WebSocket client in background thread."""
        if self.running:
            logger.warning("WebSocket client already running")
            return
        
        self.running = True
        
        def run_event_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._connect())
        
        self.thread = threading.Thread(target=run_event_loop, daemon=True)
        self.thread.start()
        
        logger.info("WebSocket client started")
    
    def stop(self):
        """Stop WebSocket client."""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("WebSocket client stopped")


class PriceMonitor:
    """
    Monitors real-time price changes and triggers analysis.
    Integrates with RL system for continuous learning.
    """
    
    def __init__(self, 
                 analysis_trigger: Callable,
                 price_change_threshold: float = 2.0,
                 time_interval: int = 3600):
        """
        Initialize price monitor.
        
        Args:
            analysis_trigger: Function to call when analysis should run
            price_change_threshold: Price change % to trigger analysis
            time_interval: Minimum seconds between analyses
        """
        self.analysis_trigger = analysis_trigger
        self.price_change_threshold = price_change_threshold
        self.time_interval = time_interval
        
        # Track last analysis time and price per symbol
        self.last_analysis = {}
        self.last_price = {}
        
        logger.info(f"Price monitor initialized: threshold={price_change_threshold}%, "
                   f"interval={time_interval}s")
    
    def on_ticker_update(self, ticker_data: Dict):
        """
        Handle ticker update from WebSocket.
        
        Args:
            ticker_data: Binance ticker data
        """
        try:
            symbol = ticker_data.get('s')
            current_price = float(ticker_data.get('c', 0))
            price_change_pct = float(ticker_data.get('P', 0))
            
            if not symbol or not current_price:
                return
            
            # Initialize tracking for new symbol
            if symbol not in self.last_price:
                self.last_price[symbol] = current_price
                self.last_analysis[symbol] = datetime.now()
                logger.info(f"Started monitoring {symbol} at ${current_price:.2f}")
                return
            
            # Check if enough time has passed
            time_since_last = (datetime.now() - self.last_analysis[symbol]).total_seconds()
            if time_since_last < self.time_interval:
                return
            
            # Check if price change is significant
            price_change = abs(
                (current_price - self.last_price[symbol]) / self.last_price[symbol] * 100
            )
            
            if price_change >= self.price_change_threshold:
                logger.info(f"{symbol} price changed {price_change:.2f}% "
                           f"(${self.last_price[symbol]:.2f} -> ${current_price:.2f})")
                
                # Trigger analysis
                self.analysis_trigger(symbol, current_price)
                
                # Update tracking
                self.last_price[symbol] = current_price
                self.last_analysis[symbol] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing ticker update: {e}")
    
    def on_kline_update(self, kline_data: Dict):
        """
        Handle kline (candlestick) update from WebSocket.
        
        Args:
            kline_data: Binance kline data
        """
        try:
            if not kline_data.get('k', {}).get('x', False):
                # Kline not closed yet
                return
            
            symbol = kline_data.get('s')
            kline = kline_data.get('k', {})
            close_price = float(kline.get('c', 0))
            
            logger.debug(f"{symbol} kline closed at ${close_price:.2f}")
            
            # Could trigger analysis based on kline patterns here
            
        except Exception as e:
            logger.error(f"Error processing kline update: {e}")


class RealTimeAnalyzer:
    """
    Coordinates real-time data streaming with RL-based analysis.
    """
    
    def __init__(self, crypto_analyzer, rl_enabled: bool = True):
        """
        Initialize real-time analyzer.
        
        Args:
            crypto_analyzer: Main CryptoAnalyzer instance
            rl_enabled: Whether to use RL predictions
        """
        self.analyzer = crypto_analyzer
        self.rl_enabled = rl_enabled
        self.ws_client = BinanceWebSocketClient()
        self.price_monitor = PriceMonitor(self.trigger_analysis)
        
        logger.info(f"Real-time analyzer initialized (RL: {rl_enabled})")
    
    def trigger_analysis(self, symbol: str, current_price: float):
        """
        Triggered when price change threshold is met.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current price
        """
        logger.info(f"Analysis triggered for {symbol} at ${current_price:.2f}")
        
        try:
            # Run full analysis
            self.analyzer.analyze_symbol(symbol)
            logger.info(f"Analysis completed for {symbol}")
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
    
    def start_monitoring(self, symbols: List[str]):
        """
        Start monitoring symbols in real-time.
        
        Args:
            symbols: List of symbols to monitor
        """
        logger.info(f"Starting real-time monitoring for {symbols}")
        
        # Subscribe to ticker updates
        for symbol in symbols:
            self.ws_client.subscribe_ticker(
                symbol,
                self.price_monitor.on_ticker_update
            )
        
        # Start WebSocket client
        self.ws_client.start()
        
        logger.info("Real-time monitoring active")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.ws_client.stop()
        logger.info("Real-time monitoring stopped")
