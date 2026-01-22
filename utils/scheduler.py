"""
Scheduler Module
Manages automatic execution of crypto analysis at fixed intervals
"""

import logging
import signal
import sys
from typing import List, Dict
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.memory import MemoryJobStore

logger = logging.getLogger(__name__)


# Valid interval values mapping to minutes
VALID_INTERVALS = {
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "20m": 20,
    "30m": 30,
    "60m": 60,
    "1h": 60
}


def parse_interval(interval_str: str) -> int:
    """
    Parse interval string to minutes.
    
    Args:
        interval_str: Interval string (e.g., '5m', '15m', '1h')
    
    Returns:
        Number of minutes
        
    Raises:
        ValueError: If interval is invalid
    """
    key = interval_str.lower().strip()
    if key not in VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval: {interval_str}. "
            f"Valid values: {list(VALID_INTERVALS.keys())}"
        )
    return VALID_INTERVALS[key]


class AnalysisScheduler:
    """
    Manages scheduled execution of crypto analysis tasks.
    Prevents overlapping runs and handles graceful shutdown.
    """
    
    def __init__(self, pipeline, kline_interval: str = "1h", generate_charts: bool = False):
        """
        Initialize the scheduler.
        
        Args:
            pipeline: AnalysisPipeline instance
            kline_interval: Binance kline interval for data collection
            generate_charts: Whether to generate charts during scheduled runs
        """
        self.pipeline = pipeline
        self.kline_interval = kline_interval
        self.generate_charts = generate_charts
        
        # Configure APScheduler
        jobstores = {"default": MemoryJobStore()}
        executors = {"default": ThreadPoolExecutor(max_workers=8)}
        job_defaults = {
            "coalesce": True,  # Combine missed runs into one
            "max_instances": 1,  # Only one instance per job
            "misfire_grace_time": 60  # Allow 60s grace time
        }
        
        self.scheduler = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone="UTC"
        )
        
        logger.info("AnalysisScheduler initialized")
    
    def _job_wrapper(self, symbol: str):
        """
        Wrapper function for scheduled jobs.
        
        Args:
            symbol: Crypto symbol to analyze
        """
        try:
            logger.info(f"[Scheduler] Starting scheduled job for {symbol}")
            self.pipeline.run_pipeline(
                symbol=symbol,
                interval=self.kline_interval,
                generate_charts=self.generate_charts
            )
        except Exception as e:
            logger.error(f"[Scheduler] Job failed for {symbol}: {str(e)}", exc_info=True)
    
    def add_schedule(self, symbol: str, interval_str: str):
        """
        Add a scheduled job for a symbol.
        
        Args:
            symbol: Crypto symbol to analyze
            interval_str: Interval string (e.g., '5m', '15m', '1h')
        """
        symbol = symbol.upper()
        minutes = parse_interval(interval_str)
        
        job_id = f"{symbol}_{interval_str}"
        trigger = IntervalTrigger(minutes=minutes)
        
        self.scheduler.add_job(
            self._job_wrapper,
            trigger,
            id=job_id,
            replace_existing=True,
            kwargs={"symbol": symbol},
            name=f"{symbol} every {interval_str}"
        )
        
        logger.info(f"Added schedule: {symbol} every {minutes} minutes (job_id: {job_id})")
    
    def start(self, schedules: List[Dict]):
        """
        Start the scheduler with the given schedules.
        
        Args:
            schedules: List of schedule dictionaries with 'symbol' and 'every' keys
        """
        if not schedules:
            logger.warning("No schedules provided")
            return
        
        # Add all schedules
        added = []
        for sch in schedules:
            symbol = sch["symbol"].upper()
            interval = sch["every"]
            
            try:
                self.add_schedule(symbol, interval)
                added.append(f"{symbol}@{interval}")
            except Exception as e:
                logger.error(f"Failed to add schedule for {symbol}: {str(e)}")
        
        if not added:
            logger.error("No valid schedules were added")
            return
        
        # Start the scheduler
        self.scheduler.start()
        logger.info(f"Scheduler started for: {', '.join(added)}")
        
        # Setup graceful shutdown
        self._setup_signal_handlers()
        
        # Keep process alive
        self._keep_alive()
    
    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown."""
        def shutdown_handler(signum, frame):
            logger.info("Received shutdown signal, stopping scheduler...")
            self.shutdown()
            sys.exit(0)
        
        # Handle SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
    
    def _keep_alive(self):
        """Keep the main process alive while scheduler runs."""
        import time
        logger.info("Scheduler running (Press Ctrl+C to stop)")
        print("\n✅ Scheduler is active. Jobs will run at their scheduled times.")
        print("Press Ctrl+C to stop.\n")
        
        try:
            while True:
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler interrupted by user")
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the scheduler gracefully."""
        logger.info("Shutting down scheduler...")
        self.scheduler.shutdown(wait=True)
        logger.info("Scheduler stopped")
    
    def list_jobs(self):
        """List all scheduled jobs."""
        jobs = self.scheduler.get_jobs()
        if not jobs:
            logger.info("No scheduled jobs")
            return
        
        logger.info("Scheduled jobs:")
        for job in jobs:
            logger.info(f"  - {job.name} (id: {job.id}, next run: {job.next_run_time})")


def start_scheduler(schedules: List[Dict], pipeline, kline_interval: str = "1h", generate_charts: bool = False):
    """
    Convenience function to create and start a scheduler.
    
    Args:
        schedules: List of schedule dictionaries
        pipeline: AnalysisPipeline instance
        kline_interval: Binance kline interval for data collection
        generate_charts: Whether to generate charts during scheduled runs
    """
    scheduler = AnalysisScheduler(
        pipeline=pipeline,
        kline_interval=kline_interval,
        generate_charts=generate_charts
    )
    scheduler.start(schedules)
