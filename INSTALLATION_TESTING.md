# Installation and Testing Guide - Scheduler Feature

## Step 1: Install New Dependencies

The scheduler feature requires new Python packages. Install them using:

```bash
cd /Users/I1846/Documents/Projectwork/crypto_analyzer
conda activate venv_a2a
pip install apscheduler tenacity pyyaml python-dotenv
```

Or install all requirements at once:

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import apscheduler; import tenacity; import yaml; print('✅ All scheduler dependencies installed')"
```

## Step 2: Basic Testing

### Test 1: Run Once (Backwards Compatibility)

Test that the original behavior still works:

```bash
# Original command style (should still work)
python main.py --symbol BTCUSDT

# New explicit style
python main.py --symbol BTCUSDT --run-once
```

Expected: Analysis runs once and exits (same as before)

### Test 2: Single Symbol Scheduler

Test scheduling a single symbol:

```bash
# Schedule BTCUSDT every 5 minutes (without charts for faster testing)
python main.py --symbol BTCUSDT --every 5m --no-charts
```

Expected:
- Message: "⏰ SCHEDULER MODE - Single Symbol"
- Message: "📊 Scheduling BTCUSDT every 5 minutes"
- Analysis runs immediately
- Analysis repeats every 5 minutes
- Press Ctrl+C to stop

### Test 3: Config-Based Scheduler

Edit `config.yaml` for testing (short intervals):

```yaml
schedules:
  - symbol: BTCUSDT
    every: 5m
  
  - symbol: ETHUSDT
    every: 10m
```

Run the scheduler:

```bash
python main.py --config config.yaml --no-charts
```

Expected:
- Message: "⏰ SCHEDULER MODE - Config File"
- Message: "📋 Loaded 2 schedule(s)"
- Both analyses run on their schedules
- Press Ctrl+C to stop

## Step 3: Verify Features

### Feature 1: No Overlapping Runs

If a run takes longer than the interval, the next run should be skipped.

Test with a very short interval:

```bash
# This will likely skip some runs if analysis takes > 5 min
python main.py --symbol BTCUSDT --every 5m
```

Check logs for: "Execution of job ... was missed" or similar APScheduler messages

### Feature 2: Automatic Retries

Enable debug logging to see retry attempts:

```bash
export LOG_LEVEL=DEBUG
python main.py --symbol BTCUSDT --run-once
```

You should see retry attempts if there are any network issues.

### Feature 3: Graceful Shutdown

Start the scheduler and then press Ctrl+C:

```bash
python main.py --symbol BTCUSDT --every 10m
# Press Ctrl+C
```

Expected:
- Message: "Shutting down scheduler..."
- Clean exit without errors

## Step 4: Test with RL Enabled

```bash
python main.py --symbol SOLUSDT --every 15m --enable-rl --no-charts
```

Expected: RL components should work normally with scheduler

## Step 5: Production Testing

### Test Long-Running Scheduler

```bash
# Run for an extended period (30-60 minutes)
python main.py --config config.yaml --no-charts
```

Monitor:
- Memory usage (should remain stable)
- CPU usage (should be low between runs)
- Log files (should show successful runs)
- Reports generated in `data/reports/`

### Test Background Execution

Using screen:

```bash
screen -S crypto-test
python main.py --config config.yaml --no-charts
# Press Ctrl+A, then D to detach
screen -r crypto-test  # to reattach
```

## Step 6: Verify All Modes Still Work

### Verify Prediction Mode
```bash
python main.py --verify-predictions
```

### Verify Monitor Mode
```bash
python main.py --symbol BTCUSDT --monitor
# Press Ctrl+C to stop
```

## Troubleshooting

### Import Errors

If you see import errors for apscheduler, tenacity, etc.:

```bash
pip install apscheduler tenacity pyyaml python-dotenv
```

### YAML Parse Errors

If config.yaml fails to load:

```bash
python -c "import yaml; print(yaml.safe_load(open('config.yaml')))"
```

### Scheduler Not Starting

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python main.py --config config.yaml
```

### Jobs Not Running

Check the schedule configuration:
- Verify symbol names are correct (e.g., BTCUSDT, not BTC)
- Verify interval format (e.g., 15m, not 15)
- Check logs for job scheduling messages

### Memory Issues

If memory grows over time:
- Reduce number of scheduled symbols
- Increase intervals between runs
- Disable chart generation (`--no-charts`)
- Archive/delete old reports periodically

## Common Commands

```bash
# Quick test run
python main.py --symbol BTCUSDT --run-once --no-charts

# Schedule single symbol (5 min interval for testing)
python main.py --symbol BTCUSDT --every 5m --no-charts

# Schedule from config
python main.py --config config.yaml --no-charts

# With RL enabled
python main.py --symbol BTCUSDT --every 15m --enable-rl --no-charts

# Debug mode
export LOG_LEVEL=DEBUG
python main.py --config config.yaml

# Background with screen
screen -dmS crypto python main.py --config config.yaml
screen -r crypto  # to view
```

## Expected Output

### Successful Scheduler Start

```
⏰ SCHEDULER MODE - Config File
================================================================================
📋 Loaded 2 schedule(s) from config.yaml
   - BTCUSDT every 5m
   - ETHUSDT every 10m
2026-01-22 12:00:00 | INFO | utils.scheduler | Added schedule: BTCUSDT every 5 minutes (job_id: BTCUSDT_5m)
2026-01-22 12:00:00 | INFO | utils.scheduler | Added schedule: ETHUSDT every 10 minutes (job_id: ETHUSDT_10m)
2026-01-22 12:00:00 | INFO | utils.scheduler | Scheduler started for: BTCUSDT@5m, ETHUSDT@10m
```

### Successful Job Execution

```
2026-01-22 12:00:00 | INFO | utils.scheduler | [Scheduler] Starting scheduled job for BTCUSDT
2026-01-22 12:00:00 | INFO | utils.pipeline | [BTCUSDT] Pipeline start at 2026-01-22T12:00:00 interval=1h
...
2026-01-22 12:02:30 | INFO | utils.pipeline | [BTCUSDT] Pipeline completed successfully in 150.45s
```

### Successful Shutdown

```
^C
2026-01-22 12:15:00 | INFO | utils.scheduler | Received shutdown signal, stopping scheduler...
2026-01-22 12:15:00 | INFO | utils.scheduler | Shutting down scheduler...
2026-01-22 12:15:01 | INFO | utils.scheduler | Scheduler stopped
```

## Next Steps

After successful testing:

1. **Adjust Config**: Set production intervals in `config.yaml` (15m, 30m, 1h)
2. **Setup Service**: Configure systemd or Docker for production
3. **Monitor Logs**: Set up log rotation and monitoring
4. **Archive Reports**: Set up periodic cleanup of old reports
5. **Backup Database**: Regular backups of SQLite database

## Support

For issues during testing:

1. Check `LOG_LEVEL=DEBUG` output
2. Verify all dependencies are installed
3. Test with `--run-once` first
4. Review error messages in logs
5. Check system resources (memory, disk space)

## Success Checklist

- [ ] New dependencies installed successfully
- [ ] `--run-once` mode works (backwards compatibility)
- [ ] Single symbol scheduler works (`--every`)
- [ ] Config-based scheduler works (`--config`)
- [ ] No overlapping runs observed
- [ ] Graceful shutdown works (Ctrl+C)
- [ ] RL mode works with scheduler
- [ ] Other modes still work (--verify-predictions, --monitor)
- [ ] Long-running test completed (30+ min)
- [ ] Memory usage stable over time
- [ ] Reports generated correctly

Once all items are checked, the scheduler feature is ready for production use!
