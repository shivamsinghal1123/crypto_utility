
GPT 5
Below is a single, copy-paste “vibe coding” prompt you can drop into your AI coding tool to refactor your repo so your app runs automatically at fixed intervals (5, 10, 15, 20, 30, 60 minutes) without manual re-runs. It introduces a scheduler with proper error handling, prevents overlapping runs per symbol, and keeps your existing analysis and RL logic intact.

Vibe coding prompt (paste everything between the lines into your code assistant)
Goal: Refactor my existing crypto analysis app so it can run automatically on a schedule (every 5, 10, 15, 20, 30 minutes, or every 1 hour), without me re-running it manually. I pass a Binance symbol today; after this change, I want to:

Run once (manual mode) just like before, and
Run continuously (scheduler mode) at a fixed interval, or read a config with multiple symbols and intervals.
Ensure no overlapping runs per symbol (if a run takes longer than the interval, the next one should be skipped/coalesced).
Keep my existing logic for market data fetch, technical analysis, fundamental analysis, and RL as-is; just wrap it in a unified pipeline function.
Add robust logging, retries for transient network errors, and safe shutdown.
Provide simple CLI flags and a config file option for multiple schedules.
Keep dependencies minimal; do not break current features.
High-level changes:

Create a clear pipeline entry point that encapsulates the current workflow for one symbol:
fetch latest market data from Binance
technical analysis
fundamental analysis
RL step (unchanged)
persist results (CSV/SQLite/Parquet) and log outcomes
Call signature:
run_pipeline(symbol: str, interval: str = "1m", lookback: str = "1 day") -> None
This function should be idempotent for a given timestamp batch and safe to call repeatedly.
Add a scheduler using APScheduler:
Use BackgroundScheduler (or AsyncIOScheduler if our codebase is async).
Each symbol/interval pair is a job with:
coalesce=True, max_instances=1, misfire_grace_time=60
IntervalTrigger driven by minutes (5, 10, 15, 20, 30, 60).
Prevent overlapping jobs for the same symbol.
Graceful shutdown on SIGINT/SIGTERM.
CLI:
python main.py --symbol BTCUSDT --run-once # runs immediately once (current behavior)
python main.py --symbol BTCUSDT --every 15m # starts a scheduler for this symbol at 15 minutes
python main.py --config config.yaml # starts scheduler for multiple symbols/intervals from config
Valid values for --every: 5m, 10m, 15m, 20m, 30m, 60m.
If both --symbol and --every are provided, schedule that one symbol only.
If --config is provided, ignore --symbol/--every and run schedules from config.
Config:
Add config.yaml with a list of schedules:
schedules:
symbol: BTCUSDT
every: 5m
symbol: ETHUSDT
every: 15m
Also allow .env for API keys if needed (but public market data may not need keys). Keep keys optional.
Error handling and retries:
Use exponential backoff for HTTP/API calls (tenacity or custom).
Log errors with stack traces; do not crash the scheduler.
If a single job fails, schedule continues.
Persistence:
Keep current persistence approach (if any). If none, add simple CSV or SQLite write with timestamps under ./data/.
Ensure unique keys (symbol + timestamp) to avoid duplicates.
Keep the reinforcement learning component unchanged; just call it from the pipeline.
Provide minimal Docker/systemd/cron notes for running in background, but default approach is APScheduler inside the app.
Now implement the changes with the following files. If my repo structure is different, adapt names and imports accordingly and wire up to existing modules instead of stubs.

File: requirements.txt
apscheduler
tenacity
pyyaml
python-dotenv
pandas
numpy
binance-connector

File: .env.example
BINANCE_API_KEY=
BINANCE_API_SECRET=

File: config.yaml
schedules:

symbol: BTCUSDT
every: 5m
symbol: ETHUSDT
every: 15m
File: app/pipeline.py
import logging
from datetime import datetime
from app.services.market_data import get_market_data
from app.analysis.technical import run_technical_analysis
from app.analysis.fundamental import run_fundamental_analysis
from app.rl.agent import run_rl_step
from app.services.storage import persist_results

logger = logging.getLogger(name)

def run_pipeline(symbol: str, interval: str = "1m", lookback: str = "1 day") -> None:
"""
Orchestrates a full run of data fetch + analyses + RL for a single symbol.
interval: Binance kline interval (e.g., '1m', '5m')
lookback: human-readable or implementation-specific (adapt to your code)
"""
start_ts = datetime.utcnow()
logger.info(f"[{symbol}] Pipeline start at {start_ts.isoformat()} interval={interval} lookback={lookback}")

# Fetch market data
df = get_market_data(symbol=symbol, interval=interval, lookback=lookback)

# Technical analysis
ta_out = run_technical_analysis(df=df, symbol=symbol)

# Fundamental analysis (adapt to your approach; may be on-chain, news, etc.)
fa_out = run_fundamental_analysis(symbol=symbol)

# RL step (unchanged; pass in what it needs)
rl_out = run_rl_step(df=df, ta=ta_out, fa=fa_out, symbol=symbol)

# Persist
persist_results(symbol=symbol, df=df, ta=ta_out, fa=fa_out, rl=rl_out, run_ts=start_ts)

logger.info(f"[{symbol}] Pipeline completed successfully.")
File: app/services/market_data.py
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from binance.spot import Spot as BinanceSpot

logger = logging.getLogger(name)

def _init_client():
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
# Public endpoints don't need auth; providing keys is optional
if api_key and api_secret:
return BinanceSpot(key=api_key, secret=api_secret)
return BinanceSpot()

def _parse_lookback(lookback: str) -> int:
# Simplified: return number of minutes to look back; adapt as needed
lookback = lookback.lower().strip()
mapping = {
"1 day": 1440, "1d": 1440, "24h": 1440,
"12h": 720, "6h": 360, "4h": 240, "2h": 120, "1h": 60,
"30m": 30, "15m": 15, "5m": 5, "3d": 4320, "7d": 10080
}
return mapping.get(lookback, 1440)

@retry(
reraise=True,
stop=stop_after_attempt(5),
wait=wait_exponential(multiplier=1, min=1, max=30),
retry=retry_if_exception_type(Exception),
)
def get_market_data(symbol: str, interval: str, lookback: str) -> pd.DataFrame:
client = _init_client()
minutes = _parse_lookback(lookback)
end = datetime.utcnow()
start = end - timedelta(minutes=minutes)

# Binance klines, returns [[open_time, open, high, low, close, volume, close_time, ...], ...]
klines = client.klines(symbol, interval, startTime=int(start.timestamp() * 1000), endTime=int(end.timestamp() * 1000))
if not klines:
raise RuntimeError(f"No klines returned for {symbol} {interval}")

cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore"]
df = pd.DataFrame(klines, columns=cols)
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
numeric = ["open","high","low","close","volume","qav","taker_base_vol","taker_quote_vol"]
for c in numeric:
df[c] = pd.to_numeric(df[c], errors="coerce")
return df
File: app/analysis/technical.py
def run_technical_analysis(df, symbol: str):
# Placeholder; integrate your existing TA here
# Example: compute simple indicators and return dict or DataFrame
out = {
"rsi": None,
"macd": None,
"signal": None
}
return out

File: app/analysis/fundamental.py
def run_fundamental_analysis(symbol: str):
# Placeholder; integrate your existing fundamental analysis here
# Could be on-chain, news sentiment, tokenomics, etc.
return {"fa_score": None}

File: app/rl/agent.py
def run_rl_step(df, ta, fa, symbol: str):
# Placeholder; call your existing RL logic here unchanged
# Return whatever summary/outcomes are needed for persistence
return {"rl_policy": None, "action": None}

File: app/services/storage.py
import os
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(name)

def persist_results(symbol: str, df: pd.DataFrame, ta, fa, rl, run_ts: datetime):
os.makedirs("data", exist_ok=True)
# Minimal example: write latest row to CSV and a summary log
latest = df.iloc[-1:].copy()
latest.to_csv(f"data/{symbol}_latest.csv", index=False)
with open(f"data/{symbol}_runs.log", "a", encoding="utf-8") as f:
f.write(f"{run_ts.isoformat()} | TA={ta} | FA={fa} | RL={rl}\n")

File: scheduler/scheduler.py
import logging
import signal
import sys
from typing import List, Dict
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from datetime import timedelta
from app.pipeline import run_pipeline

logger = logging.getLogger(name)

VALID_MINUTES = {"5m":5, "10m":10, "15m":15, "20m":20, "30m":30, "60m":60, "1h":60}

def parse_every(every: str) -> int:
key = every.lower().strip()
if key not in VALID_MINUTES:
raise ValueError(f"Invalid every value: {every}. Use one of {list(VALID_MINUTES.keys())}")
return VALID_MINUTES[key]

def _job(symbol: str, kline_interval: str):
# Choose kline interval based on schedule if desired.
# For example, align kline_interval to '1m' or '5m' etc. Here we keep it simple.
run_pipeline(symbol=symbol, interval=kline_interval, lookback="1 day")

def start_scheduler(schedules: List[Dict], default_kline_interval: str = "1m"):
jobstores = {"default": MemoryJobStore()}
executors = {"default": ThreadPoolExecutor(max_workers=8)}
job_defaults = {"coalesce": True, "max_instances": 1, "misfire_grace_time": 60}
scheduler = BackgroundScheduler(jobstores=jobstores, executors=executors, job_defaults=job_defaults, timezone="UTC")
scheduler.start()

added = []
for sch in schedules:
symbol = sch["symbol"].upper()
minutes = parse_every(sch["every"])
trigger = IntervalTrigger(minutes=minutes)
job_id = f"{symbol}"
scheduler.add_job(
_job, trigger, id=job_id, replace_existing=True,
kwargs={"symbol": symbol, "kline_interval": default_kline_interval},
)
added.append(f"{symbol}@{minutes}m")

logger.info(f"Scheduler started for: {', '.join(added)}")

# Graceful shutdown
def shutdown(signum, frame):
logger.info("Shutting down scheduler...")
scheduler.shutdown(wait=False)
sys.exit(0)

for sig in (signal.SIGINT, signal.SIGTERM):
signal.signal(sig, shutdown)

# Keep process alive
try:
signal.pause()
except AttributeError:
# Windows fallback
import time
while True:
time.sleep(3600)
File: main.py
import argparse
import logging
import os
import sys
import yaml
from dotenv import load_dotenv
from app.pipeline import run_pipeline
from scheduler.scheduler import start_scheduler, parse_every

def setup_logging():
level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
level=level,
format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

def load_config(path: str):
with open(path, "r", encoding="utf-8") as f:
return yaml.safe_load(f)

def main():
load_dotenv()
setup_logging()

parser = argparse.ArgumentParser(description="Crypto analysis app with scheduler")
parser.add_argument("--symbol", type=str, help="Binance symbol, e.g., BTCUSDT")
parser.add_argument("--run-once", action="store_true", help="Run once and exit")
parser.add_argument("--every", type=str, help="Interval like 5m,10m,15m,20m,30m,60m")
parser.add_argument("--config", type=str, help="Path to config.yaml for multiple schedules")
parser.add_argument("--kline-interval", type=str, default="1m", help="Binance kline interval for pipeline (default 1m)")
parser.add_argument("--lookback", type=str, default="1 day", help="Lookback window (default '1 day')")
args = parser.parse_args()

# Run-once mode (preserves original behavior)
if args.run_once:
if not args.symbol:
print("--symbol is required with --run-once", file=sys.stderr)
sys.exit(1)
run_pipeline(symbol=args.symbol.upper(), interval=args.kline_interval, lookback=args.lookback)
return

# Config-driven multi-schedule mode
if args.config:
conf = load_config(args.config)
schedules = conf.get("schedules", [])
if not schedules:
print("No schedules found in config.yaml", file=sys.stderr)
sys.exit(1)
start_scheduler(schedules=schedules, default_kline_interval=args.kline_interval)
return

# Single-symbol scheduler mode
if args.symbol and args.every:
parse_every(args.every) # validate
schedules = [{"symbol": args.symbol.upper(), "every": args.every}]
start_scheduler(schedules=schedules, default_kline_interval=args.kline_interval)
return

# If nothing matched, print help
parser.print_help()
if name == "main":
main()

Integration notes:

Wire app.analysis.technical.run_technical_analysis, app.analysis.fundamental.run_fundamental_analysis, and app.rl.agent.run_rl_step to your existing implementations. Replace the placeholders with your actual logic.
If your app is async, swap BackgroundScheduler for AsyncIOScheduler and run the pipeline via asyncio. Also ensure get_market_data and other calls are async-compatible.
If you already have a data persistence layer, use that inside persist_results instead of the CSV stub.
If your code already had a single entry script, replace it with main.py above or merge the CLI/scheduler logic into it.
How to run:

Install deps: pip install -r requirements.txt
Run once (as before): python main.py --symbol BTCUSDT --run-once
Schedule one symbol: python main.py --symbol BTCUSDT --every 15m
Schedule via config: python main.py --config config.yaml
Optional deployment options:

Systemd: run python main.py --config config.yaml as a service.
Cron: not recommended for sub-5-minute intervals or overlapping prevention; APScheduler already manages overlaps and retries.
Docker: wrap the command in a container entrypoint and keep the process in foreground.
Please refactor my existing modules instead of the placeholder stubs where applicable. Preserve my RL and analysis behavior. Ensure no overlapping runs per symbol, log errors, and keep the process alive in scheduler mode. 