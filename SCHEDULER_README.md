# Crypto Analyzer - Scheduler Feature

This document describes the automatic scheduling feature that allows the crypto analyzer to run at fixed intervals without manual intervention.

## Overview

The scheduler enables you to:
- Run analysis automatically at intervals: 5, 10, 15, 20, 30, or 60 minutes
- Schedule multiple symbols simultaneously
- Prevent overlapping runs per symbol
- Handle errors gracefully with automatic retries
- Run in background as a service

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required new packages:
- `apscheduler` - Task scheduling
- `tenacity` - Retry logic
- `pyyaml` - Config file support
- `python-dotenv` - Environment variables

### 2. Configure API Keys (Optional)

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Most features work without API keys using public Binance endpoints.

## Usage Examples

### Run Once (Original Behavior)

```bash
# Run analysis once for a symbol
python main.py --symbol BTCUSDT --run-once

# Run with RL enabled
python main.py --symbol SOLUSDT --run-once --enable-rl

# Run without charts
python main.py --symbol ETHUSDT --run-once --no-charts
```

### Schedule Single Symbol

```bash
# Analyze BTCUSDT every 15 minutes
python main.py --symbol BTCUSDT --every 15m

# Analyze ETHUSDT every 30 minutes with RL enabled
python main.py --symbol ETHUSDT --every 30m --enable-rl

# Analyze SOLUSDT every hour without charts
python main.py --symbol SOLUSDT --every 1h --no-charts
```

Valid intervals: `5m`, `10m`, `15m`, `20m`, `30m`, `60m` (or `1h`)

### Schedule Multiple Symbols from Config

1. Edit `config.yaml`:

```yaml
schedules:
  - symbol: BTCUSDT
    every: 15m
  
  - symbol: ETHUSDT
    every: 30m
  
  - symbol: SOLUSDT
    every: 30m
```

2. Run scheduler:

```bash
# Start scheduler with config
python main.py --config config.yaml

# With RL enabled
python main.py --config config.yaml --enable-rl

# Without charts
python main.py --config config.yaml --no-charts
```

The scheduler will run continuously. Press `Ctrl+C` to stop.

## Features

### No Overlapping Runs

The scheduler ensures that only one analysis runs per symbol at a time:
- If a run takes longer than the interval, the next one is skipped
- Uses `max_instances=1` and `coalesce=True` in APScheduler
- Prevents resource conflicts and data corruption

### Automatic Retries

Network requests automatically retry with exponential backoff:
- Up to 5 retry attempts
- Initial delay: 1 second
- Maximum delay: 30 seconds
- Retries on: timeout, connection errors, server errors

### Error Handling

- Individual job failures don't crash the scheduler
- All errors are logged with full stack traces
- Failed runs are logged but scheduler continues
- Graceful shutdown on `SIGINT` (Ctrl+C) or `SIGTERM`

### Logging

All operations are logged with timestamps:
- Job starts and completions
- Errors and warnings
- Pipeline execution times
- Data collection status

Set log level in `.env`:
```bash
LOG_LEVEL=INFO  # or DEBUG, WARNING, ERROR
```

## Architecture

### New Modules

#### `utils/pipeline.py`
- Orchestrates the complete analysis workflow
- Wraps existing `CryptoAnalyzer` logic
- Provides clean entry point for scheduling

#### `utils/scheduler.py`
- Manages APScheduler configuration
- Handles job scheduling and lifecycle
- Implements graceful shutdown
- Prevents overlapping executions

### Updated Modules

#### `main.py`
- New CLI arguments: `--run-once`, `--every`, `--config`
- Scheduler mode support
- Backwards compatible with original behavior

#### `data_collection/price_data.py`
- Added `@retry` decorator for network resilience
- Exponential backoff on failures
- Better error handling

## Advanced Usage

### Custom Kline Intervals

Control the Binance kline interval for data collection:

```bash
# Use 5-minute klines (default is 1h)
python main.py --symbol BTCUSDT --every 15m --kline-interval 5m

# Use 1-day klines
python main.py --config config.yaml --kline-interval 1d
```

Valid kline intervals: `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`, `3d`, `1w`, `1M`

### Running as a Service

#### Using systemd (Linux)

Create `/etc/systemd/system/crypto-analyzer.service`:

```ini
[Unit]
Description=Crypto Analyzer Scheduler
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/crypto_analyzer
ExecStart=/path/to/python main.py --config config.yaml --enable-rl
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable crypto-analyzer
sudo systemctl start crypto-analyzer
sudo systemctl status crypto-analyzer
```

#### Using Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "main.py", "--config", "config.yaml"]
```

Build and run:
```bash
docker build -t crypto-analyzer .
docker run -d --name crypto-analyzer crypto-analyzer
```

#### Using screen (Simple Background Process)

```bash
# Start in background
screen -dmS crypto-analyzer python main.py --config config.yaml

# Attach to view
screen -r crypto-analyzer

# Detach: Ctrl+A, then D
```

## Troubleshooting

### Scheduler Not Starting

Check that you have valid schedules:
```bash
python -c "import yaml; print(yaml.safe_load(open('config.yaml')))"
```

### Jobs Not Running

Enable debug logging in `.env`:
```bash
LOG_LEVEL=DEBUG
```

Check logs for errors and job scheduling info.

### Overlapping Runs

The scheduler prevents this by default. If you see warnings about skipped runs, it means the previous run is still executing. Consider:
- Increasing the interval
- Disabling chart generation (`--no-charts`)
- Using quick analysis mode

### Memory Issues

For long-running schedulers:
- Disable chart generation if not needed
- Reduce the number of scheduled symbols
- Increase the interval between runs
- Monitor with `top` or `htop`

## Configuration Reference

### config.yaml Structure

```yaml
schedules:
  - symbol: BTCUSDT    # Binance symbol (required)
    every: 15m         # Interval: 5m, 10m, 15m, 20m, 30m, 60m/1h (required)
  
  - symbol: ETHUSDT
    every: 30m
```

### Environment Variables (.env)

```bash
# Logging
LOG_LEVEL=INFO

# Binance (optional)
BINANCE_API_KEY=
BINANCE_API_SECRET=

# Other API keys (see .env.example for full list)
```

### CLI Arguments

```
Required (choose one mode):
  --symbol SYMBOL       Symbol for analysis
  --config CONFIG       Config file for multiple schedules
  --verify-predictions  Verify past predictions
  --monitor            Real-time monitoring mode

Scheduler Arguments:
  --run-once           Run once and exit
  --every INTERVAL     Schedule interval (5m, 10m, 15m, 20m, 30m, 60m)
  --kline-interval     Binance kline interval (default: 1h)

Analysis Options:
  --enable-rl          Enable RL predictions
  --no-charts          Skip chart generation
  --analysis TYPE      Analysis type: full or quick
```

## Performance Tips

1. **Use appropriate intervals**: 
   - High volatility coins: 5-15 minutes
   - Stable coins: 30-60 minutes

2. **Optimize for speed**:
   - Use `--no-charts` for faster runs
   - Use `--kline-interval 1h` or higher
   - Disable unnecessary API integrations

3. **Resource management**:
   - Limit concurrent symbols (3-5 recommended)
   - Stagger intervals in config
   - Monitor system resources

4. **Data storage**:
   - Reports are saved in `data/reports/`
   - Old reports can be archived/deleted
   - Database grows over time - plan for cleanup

## Migration from Original Behavior

The scheduler feature is **fully backwards compatible**:

```bash
# Old way (still works)
python main.py --symbol BTCUSDT

# New explicit way
python main.py --symbol BTCUSDT --run-once
```

Both commands work identically. The `--run-once` flag is optional when using `--symbol` without `--every`.

## Support

For issues or questions:
1. Check logs with `LOG_LEVEL=DEBUG`
2. Verify configuration with `--run-once` first
3. Review error messages and stack traces
4. Check that all dependencies are installed

## Summary

The scheduler feature enables automated, hands-free crypto analysis:
- ✅ Run at fixed intervals (5m to 60m)
- ✅ Schedule multiple symbols
- ✅ No overlapping runs
- ✅ Automatic error recovery
- ✅ Production-ready with systemd/Docker
- ✅ Backwards compatible
- ✅ Full logging and monitoring

Start scheduling today:
```bash
python main.py --symbol BTCUSDT --every 15m
```
