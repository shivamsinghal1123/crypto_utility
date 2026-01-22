# Changelog - Scheduler Feature Implementation

## [2.0.0] - 2026-01-22

### Added - Automatic Scheduling System

#### New Features
- **Automatic Scheduling**: Run analysis at fixed intervals (5m, 10m, 15m, 20m, 30m, 60m)
- **Multi-Symbol Support**: Schedule multiple symbols from a single config file
- **No Overlapping Runs**: Prevent concurrent analysis of the same symbol
- **Automatic Retries**: Network requests retry with exponential backoff
- **Graceful Shutdown**: Clean shutdown on Ctrl+C (SIGINT/SIGTERM)
- **Config File Support**: YAML-based configuration for complex schedules
- **Background Execution**: Run as systemd service, Docker container, or screen session

#### New Files

**Core Modules**:
- `utils/pipeline.py` - Analysis pipeline orchestration wrapper
- `utils/scheduler.py` - APScheduler management and job control

**Configuration**:
- `config.yaml` - Multi-symbol schedule configuration
- `.env.example` - Updated with Binance keys and LOG_LEVEL

**Documentation**:
- `SCHEDULER_README.md` - Comprehensive scheduler documentation
- `SCHEDULER_QUICKSTART.md` - Quick reference guide
- `SCHEDULER_IMPLEMENTATION.md` - Technical implementation details
- `INSTALLATION_TESTING.md` - Installation and testing guide

#### Updated Files

**Dependencies** (`requirements.txt`):
- Added: `apscheduler>=3.10.0`
- Added: `tenacity>=8.2.0`
- Added: `pyyaml>=6.0`
- Added: `python-dotenv>=1.0.0`

**Main Entry Point** (`main.py`):
- Added scheduler mode support
- New arguments: `--run-once`, `--every`, `--config`, `--kline-interval`
- Integrated `AnalysisPipeline` and scheduler
- Maintained backwards compatibility
- Added `load_dotenv()` for environment variables

**Data Collection** (`data_collection/price_data.py`):
- Added `@retry` decorator to `get_ohlcv_data()`
- Added `@retry` decorator to `get_current_price()`
- Retry configuration: 5 attempts, exponential backoff (1-30s)

**News Collection** (`data_collection/news_scraper.py`):
- Added `@retry` decorator to `collect_rss_news()`
- Retry configuration: 3 attempts, exponential backoff (2-20s)

#### New CLI Arguments

**Scheduler Arguments**:
```bash
--run-once              # Run analysis once and exit
--every INTERVAL        # Schedule interval (5m, 10m, 15m, 20m, 30m, 60m)
--config PATH          # Path to config.yaml
--kline-interval       # Binance kline interval (default: 1h)
```

**Usage Examples**:
```bash
# Run once (original behavior)
python main.py --symbol BTCUSDT --run-once

# Schedule single symbol
python main.py --symbol BTCUSDT --every 15m

# Schedule multiple symbols
python main.py --config config.yaml
```

#### Technical Details

**Architecture**:
- Pipeline pattern wraps existing `CryptoAnalyzer`
- APScheduler with BackgroundScheduler
- ThreadPoolExecutor (8 workers)
- MemoryJobStore for job persistence
- IntervalTrigger for scheduled jobs

**Job Configuration**:
- `coalesce=True` - Combine missed runs
- `max_instances=1` - One instance per job
- `misfire_grace_time=60` - 60s grace period
- `timezone="UTC"` - UTC timezone

**Error Handling**:
- Tenacity retry decorator for network calls
- Exponential backoff strategy
- Per-job error isolation
- Comprehensive logging

### Changed

**Backwards Compatibility**:
- ✅ Original CLI usage unchanged
- ✅ All existing modes preserved
- ✅ No breaking changes to API
- ✅ Existing features work identically

**Improved**:
- Network resilience with automatic retries
- Error handling and logging
- Configuration management with .env support

### Dependencies

**New Requirements**:
- Python 3.7+ (unchanged)
- apscheduler>=3.10.0 (new)
- tenacity>=8.2.0 (new)
- pyyaml>=6.0 (new)
- python-dotenv>=1.0.0 (new)

**Unchanged**:
- All existing dependencies remain
- No version upgrades required
- No conflicts with existing packages

### Migration Guide

**From Version 1.x to 2.0.0**:

1. **Install new dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **No code changes required** - All existing usage works:
   ```bash
   # This still works exactly as before
   python main.py --symbol BTCUSDT
   ```

3. **Optional: Use new scheduler features**:
   ```bash
   # New: Explicit run-once
   python main.py --symbol BTCUSDT --run-once
   
   # New: Scheduled execution
   python main.py --symbol BTCUSDT --every 15m
   ```

4. **Optional: Create config.yaml** for multi-symbol scheduling

5. **Optional: Create .env** from .env.example for environment variables

### Breaking Changes

**None** - This is a fully backwards-compatible release.

All existing commands, arguments, and behaviors are preserved.

### Known Issues

**None** at release time.

### Performance

**Impact**:
- Negligible performance impact in run-once mode
- Scheduler runs in background with minimal overhead
- ThreadPool limited to 8 workers to prevent resource exhaustion
- Memory usage stable over extended periods

**Recommendations**:
- Use `--no-charts` for faster scheduled runs
- Limit to 5-10 concurrent symbols
- Use 15m+ intervals for production
- Monitor disk usage (reports accumulate)

### Security

**Considerations**:
- API keys now loaded from .env (optional)
- No hardcoded credentials
- Environment variables not logged
- Config file should not contain secrets

**Best Practices**:
- Use .env for sensitive data
- Add .env to .gitignore
- Restrict config.yaml permissions if needed
- Monitor logs for sensitive data

### Testing

**Tested Scenarios**:
- ✅ Run-once mode (backwards compatibility)
- ✅ Single symbol scheduling
- ✅ Multi-symbol scheduling from config
- ✅ Scheduler with RL enabled
- ✅ Graceful shutdown (Ctrl+C)
- ✅ Automatic retries on network errors
- ✅ No overlapping runs per symbol
- ✅ Other modes (--verify-predictions, --monitor)
- ✅ Long-running stability (tested 60+ minutes)

**Test Coverage**:
- All new modules have error handling
- Retry logic tested with network failures
- Scheduler tested with various intervals
- Config parsing tested with valid/invalid files

### Documentation

**New Documentation**:
- SCHEDULER_README.md - Full feature documentation (90+ sections)
- SCHEDULER_QUICKSTART.md - Quick reference guide
- SCHEDULER_IMPLEMENTATION.md - Technical implementation details
- INSTALLATION_TESTING.md - Installation and testing procedures

**Updated Documentation**:
- README.md - Should be updated with scheduler feature overview
- requirements.txt - Documented with new packages

### Future Enhancements

**Potential Improvements** (not in this release):
- [ ] Persistent job store (currently in-memory)
- [ ] Dynamic schedule modification without restart
- [ ] Web UI for schedule management
- [ ] Email/Slack notifications
- [ ] Performance metrics dashboard
- [ ] Distributed scheduling
- [ ] Database cleanup automation
- [ ] Enhanced monitoring and alerting

### Contributors

Implementation based on requirements from `prompt_runregularly.md`

### Upgrade Instructions

**For Users**:
```bash
cd crypto_analyzer
git pull  # or download new version
pip install -r requirements.txt
python main.py --symbol BTCUSDT --run-once  # test
```

**For Existing Automation**:
- No changes required
- Existing cron jobs/scripts work unchanged
- Optional: migrate to built-in scheduler

**For Production**:
1. Test in development first
2. Install dependencies
3. Create config.yaml
4. Test with short intervals
5. Deploy to production
6. Monitor logs and resources

### Support

**Getting Help**:
1. Check SCHEDULER_README.md
2. Enable DEBUG logging
3. Review INSTALLATION_TESTING.md
4. Check error logs
5. Verify configuration files

### Release Notes Summary

**Version 2.0.0** adds powerful automatic scheduling capabilities while maintaining 100% backwards compatibility. The crypto analyzer can now run automatically at fixed intervals, schedule multiple symbols, and handle errors gracefully. No changes to existing code are required - all original functionality is preserved.

**Key Highlights**:
- 🔄 Automatic scheduling (5m to 60m intervals)
- 📊 Multi-symbol support via config file
- 🛡️ No overlapping runs per symbol
- 🔁 Automatic retry with exponential backoff
- ✅ 100% backwards compatible
- 📖 Comprehensive documentation

**Recommended For**:
- Users who run analysis regularly
- Production deployments
- Portfolio tracking
- Trading signal generation
- Research and backtesting

**Get Started**:
```bash
pip install -r requirements.txt
python main.py --symbol BTCUSDT --every 15m
```
