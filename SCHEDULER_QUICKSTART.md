# Scheduler Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Usage Examples

### 1. Run Once (Original Behavior)
```bash
python main.py --symbol BTCUSDT --run-once
```

### 2. Schedule Single Symbol
```bash
# Every 15 minutes
python main.py --symbol BTCUSDT --every 15m

# Every 30 minutes with RL
python main.py --symbol ETHUSDT --every 30m --enable-rl
```

### 3. Schedule Multiple Symbols
```bash
# Edit config.yaml first, then:
python main.py --config config.yaml
```

## Valid Intervals

`5m`, `10m`, `15m`, `20m`, `30m`, `60m` (or `1h`)

## Config File Example

```yaml
schedules:
  - symbol: BTCUSDT
    every: 15m
  
  - symbol: ETHUSDT
    every: 30m
  
  - symbol: SOLUSDT
    every: 1h
```

## Stop Scheduler

Press `Ctrl+C` to gracefully stop the scheduler.

## All CLI Options

```bash
# Scheduler modes
python main.py --symbol BTCUSDT --run-once              # Run once
python main.py --symbol BTCUSDT --every 15m            # Schedule one
python main.py --config config.yaml                    # Schedule many

# Additional options
--enable-rl          # Enable RL predictions
--no-charts          # Skip chart generation
--kline-interval 5m  # Set kline interval (default: 1h)

# Other modes (unchanged)
--verify-predictions # Verify past predictions
--monitor           # Real-time WebSocket monitoring
```

## Features

✅ No overlapping runs per symbol  
✅ Automatic retries on network errors  
✅ Graceful shutdown (Ctrl+C)  
✅ Full error logging  
✅ Backwards compatible  

## Troubleshooting

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py --config config.yaml

# Test single run first
python main.py --symbol BTCUSDT --run-once

# Validate config
python -c "import yaml; print(yaml.safe_load(open('config.yaml')))"
```

## Running in Background

### Linux (systemd)
```bash
sudo systemctl start crypto-analyzer
```

### macOS/Linux (screen)
```bash
screen -dmS crypto python main.py --config config.yaml
screen -r crypto  # to attach
```

### Docker
```bash
docker run -d crypto-analyzer python main.py --config config.yaml
```

---

For full documentation, see [SCHEDULER_README.md](SCHEDULER_README.md)
