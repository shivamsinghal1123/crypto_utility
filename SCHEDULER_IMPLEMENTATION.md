# Crypto Analyzer - Scheduler Implementation Summary

## Overview

Successfully implemented automatic scheduling functionality for the Crypto Analyzer application as per the requirements in `prompt_runregularly.md`. The implementation wraps around the existing framework without breaking any current features.

## Changes Made

### 1. New Dependencies (`requirements.txt`)
Added scheduler and configuration support:
- `apscheduler>=3.10.0` - Background task scheduling
- `tenacity>=8.2.0` - Automatic retry with exponential backoff
- `pyyaml>=6.0` - YAML configuration file support
- `python-dotenv>=1.0.0` - Environment variable management

### 2. Configuration Files

#### `config.yaml` (New)
- YAML configuration for multiple symbol schedules
- Example configuration with BTCUSDT, ETHUSDT, and SOLUSDT
- Supports intervals: 5m, 10m, 15m, 20m, 30m, 60m/1h

#### `.env.example` (Updated)
- Added Binance API key placeholders
- Added LOG_LEVEL configuration
- Maintains all existing API key examples

### 3. New Modules

#### `utils/pipeline.py` (New)
**Purpose**: Orchestration wrapper for scheduled execution

**Key Features**:
- `AnalysisPipeline` class wraps existing `CryptoAnalyzer`
- `run_pipeline()` method for full analysis runs
- `run_pipeline_quick()` method for lightweight runs
- Idempotent and safe for repeated execution
- Comprehensive logging of execution time and status

**Integration**: Uses existing `CryptoAnalyzer.analyze_cryptocurrency()` method

#### `utils/scheduler.py` (New)
**Purpose**: APScheduler management and job control

**Key Features**:
- `AnalysisScheduler` class manages job lifecycle
- Prevents overlapping runs per symbol (max_instances=1)
- Coalesces missed runs into single execution
- Graceful shutdown on SIGINT/SIGTERM signals
- ThreadPoolExecutor with 8 worker threads
- 60-second misfire grace time

**Job Management**:
- `add_schedule()` - Add jobs dynamically
- `start()` - Start scheduler with schedule list
- `shutdown()` - Graceful shutdown
- `list_jobs()` - List all scheduled jobs

### 4. Updated Modules

#### `main.py` (Updated)
**New Imports**:
```python
import os
import yaml
from dotenv import load_dotenv
from utils.pipeline import AnalysisPipeline
from utils.scheduler import start_scheduler, parse_interval
```

**New CLI Arguments**:
- `--run-once` - Run analysis once and exit (explicit)
- `--every INTERVAL` - Schedule interval (5m, 10m, 15m, 20m, 30m, 60m)
- `--config CONFIG` - Path to config.yaml
- `--kline-interval INTERVAL` - Binance kline interval (default: 1h)

**New Execution Modes**:
1. **Run-once mode**: `--symbol BTCUSDT --run-once`
2. **Single symbol scheduler**: `--symbol BTCUSDT --every 15m`
3. **Multi-symbol scheduler**: `--config config.yaml`
4. **Backwards compatible**: `--symbol BTCUSDT` (same as --run-once)

**Preserved Modes**:
- `--verify-predictions` mode unchanged
- `--monitor` WebSocket mode unchanged
- All original analysis options work

#### `data_collection/price_data.py` (Updated)
**New Import**:
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
```

**Enhanced Methods with Retry**:
- `get_ohlcv_data()` - 5 retry attempts, exponential backoff
- `get_current_price()` - 5 retry attempts, exponential backoff

**Retry Configuration**:
- Stop after: 5 attempts
- Wait strategy: Exponential (1s min, 30s max)
- Retry on: RequestException, Timeout

#### `data_collection/news_scraper.py` (Updated)
**New Import**:
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
```

**Enhanced Methods with Retry**:
- `collect_rss_news()` - 3 retry attempts, exponential backoff

**Retry Configuration**:
- Stop after: 3 attempts
- Wait strategy: Exponential (2s min, 20s max)
- Retry on: RequestException, Timeout

### 5. Documentation

#### `SCHEDULER_README.md` (New)
Comprehensive documentation covering:
- Quick start guide
- Usage examples for all modes
- Feature descriptions (no overlaps, retries, error handling)
- Architecture overview
- Advanced usage (custom intervals, systemd, Docker)
- Configuration reference
- Performance tips
- Troubleshooting guide
- Migration notes

#### `SCHEDULER_QUICKSTART.md` (New)
Quick reference guide with:
- Installation steps
- Common usage examples
- Valid intervals
- Config file examples
- CLI options reference
- Background execution options

## Architecture Changes

### Execution Flow

#### Original Flow
```
main.py → CryptoAnalyzer → analyze_cryptocurrency()
```

#### New Scheduler Flow
```
main.py → AnalysisPipeline → run_pipeline() → CryptoAnalyzer.analyze_cryptocurrency()
           ↓
      AnalysisScheduler → APScheduler → IntervalTrigger → run_pipeline()
```

### Key Design Decisions

1. **Wrapper Pattern**: Pipeline wraps existing analyzer instead of modifying it
2. **Backwards Compatibility**: Original CLI usage still works without changes
3. **No Overlaps**: APScheduler job_defaults prevent concurrent runs per symbol
4. **Error Isolation**: Individual job failures don't crash scheduler
5. **Graceful Shutdown**: Signal handlers ensure clean shutdown

## Testing Recommendations

### 1. Run-Once Mode
```bash
python main.py --symbol BTCUSDT --run-once
python main.py --symbol ETHUSDT --run-once --enable-rl
```

### 2. Single Symbol Scheduler
```bash
# Short interval for testing
python main.py --symbol BTCUSDT --every 5m --no-charts
```

### 3. Multi-Symbol Scheduler
```bash
python main.py --config config.yaml --no-charts
```

### 4. Verify Backwards Compatibility
```bash
# Original usage (should work unchanged)
python main.py --symbol BTCUSDT
python main.py --symbol SOLUSDT --enable-rl
```

## Operational Considerations

### Resource Usage
- Each scheduled job runs in a thread
- Max 8 concurrent threads (ThreadPoolExecutor)
- Consider memory for long-running schedulers
- Chart generation increases resource usage

### Logging
- All operations logged with timestamps
- Failed jobs logged but don't crash scheduler
- Set `LOG_LEVEL=DEBUG` for detailed logs

### Data Storage
- Reports saved in `data/reports/SYMBOL_TIMESTAMP/`
- Database grows over time - plan for cleanup
- Each run creates new report directory

### Production Deployment
1. Use systemd service for Linux
2. Use Docker for containerized deployment
3. Use screen/tmux for simple background execution
4. Monitor logs and system resources
5. Configure appropriate intervals to avoid API rate limits

## Compatibility

### Preserved Features
✅ All original CLI arguments work  
✅ RL functionality unchanged  
✅ WebSocket monitoring unchanged  
✅ Prediction verification unchanged  
✅ Chart generation unchanged  
✅ Report generation unchanged  
✅ Database operations unchanged  

### New Features
✅ Automatic scheduling at fixed intervals  
✅ Multiple symbol support via config  
✅ No overlapping runs per symbol  
✅ Automatic retry on network errors  
✅ Graceful shutdown handling  
✅ Config file support  
✅ Enhanced error resilience  

## Future Enhancements

Potential improvements not yet implemented:
- Persistent job store (currently in-memory)
- Dynamic schedule modification without restart
- Web UI for schedule management
- Email/Slack notifications on failures
- Performance metrics and monitoring
- Distributed scheduling across multiple machines

## Dependencies

### Minimum Python Version
Python 3.7+ (existing requirement)

### New Package Versions
- apscheduler>=3.10.0
- tenacity>=8.2.0
- pyyaml>=6.0
- python-dotenv>=1.0.0

All other dependencies remain unchanged.

## Summary

The scheduler implementation successfully adds automatic execution capabilities while:
- Maintaining full backwards compatibility
- Preserving all existing features
- Following the existing architecture patterns
- Adding robust error handling
- Providing comprehensive documentation
- Requiring minimal code changes to existing modules

The implementation is production-ready and can be deployed immediately using the provided documentation.
