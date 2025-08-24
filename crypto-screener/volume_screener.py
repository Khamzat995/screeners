"""
Crypto Volume Spike Screener (Binance Futures) -> Telegram
---------------------------------------------------------
- Scans ALL USDT perpetual futures on Binance.
- Timeframes: 5m and 15m (configurable).
- Signals when the latest candle's quote-volume is a strong outlier vs prior average (z-score).
- Classifies as PUMP / DUMP using price change over the same candle.
- Sends alerts to Telegram.

Quick start:
  1) pip install -U ccxt pandas python-dotenv requests
  2) Set environment variables (or create a .env file in the same folder):
     TELEGRAM_BOT_TOKEN="7971906867:AAHUricvISDfoSdtAe9gNKg8o21LB3Z0E1o"
     TELEGRAM_CHAT_ID="552398882"
  3) python volume_screener.py

Notes:
  - Uses ccxt with Binance futures (defaultType=future).
  - Respects CCXT ratelimit (but you may still hit Binance limits if you scan too frequently).
  - Tune thresholds in CONFIG below.
"""
import os
import time
import math
import json
import asyncio
import datetime as dt
import traceback
from collections import defaultdict

import requests
import pandas as pd
from dotenv import load_dotenv

import ccxt.async_support as ccxt  # async version

# ====================== CONFIG ======================
CONFIG = {
    "exchange": "binance",
    "timeframes": ["5m", "15m"],   # you can add "1m","1h", etc.
    "lookback": 96,                # candles to pull (per symbol/timeframe)
    "z_threshold_up": 3.0,         # z-score >= threshold -> volume spike UP
    "z_threshold_down": 3.0,       # z-score <= -threshold -> volume spike DOWN (rare on volume)
    "min_quote_vol_usdt": 200_000, # filter: latest candle quote volume must exceed this
    "min_price_change_pct": 0.6,   # classify PUMP/DUMP only if abs(price_change_pct) >= this
    "poll_seconds": 60,            # run loop every N seconds
    "max_concurrent_symbols": 12,  # limit concurrency to be gentle on API
    "cooldown_minutes": 15,        # per (symbol,timeframe) cooldown to avoid spam
    "log_to_file": True,           # write events to ./screener.log
    "symbols_include": [],         # optional whitelist like ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    "symbols_exclude": ["USDC/USDT:USDT", "FDUSD/USDT:USDT"],  # noisy/pegged pairs to skip
}
# ====================================================

# Load ENV (.env optional)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("[WARN] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not set. Alerts will be printed only.")

LOG_PATH = os.path.join(os.path.dirname(__file__), "screener.log")

def now_utc():
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

def fmt_ts(ts_ms: int) -> str:
    # ms -> "YYYY-MM-DD HH:MM UTC"
    t = dt.datetime.utcfromtimestamp(ts_ms / 1000).replace(tzinfo=dt.timezone.utc)
    return t.strftime("%Y-%m-%d %H:%M UTC")

def log(msg: str):
    stamp = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    if CONFIG["log_to_file"]:
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log(f"[TELEGRAM_DISABLED] {text}")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            log(f"[TELEGRAM_ERROR] {r.status_code} {r.text}")
    except Exception as e:
        log(f"[TELEGRAM_EXCEPTION] {e}")

async def build_exchange():
    ex_id = CONFIG["exchange"]
    ex_class = getattr(ccxt, ex_id)
    exchange = ex_class({
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",  # Binance USDT-M futures
        }
    })
    await exchange.load_markets()
    return exchange

def pick_symbols(exchange) -> list:
    # All USDT perpetual linear swaps (not coin-margined)
    syms = []
    for m in exchange.markets.values():
        if not m.get("active", True):
            continue
        if m.get("type") != "swap":
            continue
        if m.get("linear") is not True:
            continue
        if "USDT" not in m.get("quote", ""):
            continue
        symbol = m["symbol"]
        if CONFIG["symbols_include"] and symbol not in CONFIG["symbols_include"]:
            continue
        if symbol in CONFIG["symbols_exclude"]:
            continue
        syms.append(symbol)
    # Optional: sort alphabetically for stable ordering
    syms.sort()
    return syms

async def fetch_ohlcv_safe(exchange, symbol: str, timeframe: str, limit: int):
    try:
        data = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return data
    except Exception as e:
        log(f"[OHLCV_ERROR] {symbol} {timeframe} -> {e}")
        return None

def compute_signal(ohlcv: list) -> dict:
    # ohlcv: [ [t, o, h, l, c, v], ... ]
    if not ohlcv or len(ohlcv) < 30:
        return {"ok": False}
    df = pd.DataFrame(ohlcv, columns=["t","o","h","l","c","v"])
    # quote-volume per candle (approx): base_vol * close
    df["qv"] = df["v"] * df["c"]
    # Use last candle as test; compute stats on previous N-1 candles
    last = df.iloc[-1]
    prev = df.iloc[:-1]
    if len(prev) < 20:
        return {"ok": False}
    mean_qv = prev["qv"].mean()
    std_qv  = prev["qv"].std(ddof=0)
    # avoid division by zero
    std_qv = std_qv if std_qv > 1e-9 else 1.0
    z = (last["qv"] - mean_qv) / std_qv
    price_change_pct = (last["c"] - last["o"]) / last["o"] * 100.0
    res = {
        "ok": True,
        "ts": int(last["t"]),
        "close": float(last["c"]),
        "qv": float(last["qv"]),
        "z": float(z),
        "price_change_pct": float(price_change_pct),
        "mean_qv": float(mean_qv),
        "std_qv": float(std_qv),
    }
    return res

def classify_event(z, price_change_pct, cfg):
    # We mainly care about high positive z (spike). Downside z is rare for volume but we keep it.
    tag = None
    if z >= cfg["z_threshold_up"]:
        if price_change_pct >= cfg["min_price_change_pct"]:
            tag = "PUMP"
        elif price_change_pct <= -cfg["min_price_change_pct"]:
            tag = "DUMP on HIGH VOL"
        else:
            tag = "VOLUME SPIKE"
    elif z <= -cfg["z_threshold_down"]:
        # low-volume anomaly (not very useful intraday)
        tag = "VOLUME DRY-UP"
    return tag

def should_alert(symbol, timeframe, event_ts_ms, cooldown_minutes, last_alert_map):
    key = (symbol, timeframe)
    last_ts = last_alert_map.get(key, 0)
    # avoid repeating within cooldown; one candle per timeframe is usually enough
    if event_ts_ms <= last_ts:
        return False
    # also cooldown by wall-clock minutes
    last_wall = last_alert_map.get((key, "wall"), 0)
    now_min = int(time.time() // 60)
    if now_min - last_wall < cooldown_minutes:
        return False
    return True

def mark_alert_sent(symbol, timeframe, event_ts_ms, last_alert_map):
    key = (symbol, timeframe)
    last_alert_map[key] = event_ts_ms
    last_alert_map[(key, "wall")] = int(time.time() // 60)

async def process_symbol(exchange, symbol, timeframe, cfg, last_alert_map):
    ohlcv = await fetch_ohlcv_safe(exchange, symbol, timeframe, limit=cfg["lookback"])
    if not ohlcv:
        return
    sig = compute_signal(ohlcv)
    if not sig["ok"]:
        return

    # Filters
    if sig["qv"] < cfg["min_quote_vol_usdt"]:
        return

    tag = classify_event(sig["z"], sig["price_change_pct"], cfg)
    if not tag:
        return

    if not should_alert(symbol, timeframe, sig["ts"], cfg["cooldown_minutes"], last_alert_map):
        return

    text = (
        f"🔔 <b>{tag}</b>\n"
        f"• <b>{symbol}</b>  ⏱ {timeframe}\n"
        f"• Time: {fmt_ts(sig['ts'])}\n"
        f"• Close: <code>{sig['close']:.4f}</code>\n"
        f"• Candle Quote Vol: <code>{sig['qv']:.0f}</code> USDT\n"
        f"• Vol z-score: <code>{sig['z']:.2f}</code>\n"
        f"• Price change: <code>{sig['price_change_pct']:.2f}%</code>\n"
        f"• Baseline QV (μ±σ): <code>{sig['mean_qv']:.0f} ± {sig['std_qv']:.0f}</code>\n"
    )
    send_telegram(text)
    log(f"{symbol} {timeframe} -> {tag} z={sig['z']:.2f} qv={sig['qv']:.0f} Δ%={sig['price_change_pct']:.2f}")
    mark_alert_sent(symbol, timeframe, sig["ts"], last_alert_map)

async def worker_loop():
    exchange = await build_exchange()
    try:
        symbols = pick_symbols(exchange)
        if CONFIG["symbols_include"]:
            log(f"[SYMBOLS] Using whitelist ({len(symbols)}): {symbols}")
        else:
            log(f"[SYMBOLS] USDT perpetual symbols: {len(symbols)}")

        sem = asyncio.Semaphore(CONFIG["max_concurrent_symbols"])
        last_alert_map = {}

        async def process_with_sem(sym, tf):
            async with sem:
                await process_symbol(exchange, sym, tf, CONFIG, last_alert_map)

        while True:
            start = time.time()
            tasks = []
            for tf in CONFIG["timeframes"]:
                for sym in symbols:
                    tasks.append(asyncio.create_task(process_with_sem(sym, tf)))
            # Let tasks run, but shield against total failure
            try:
                await asyncio.gather(*tasks)
            except Exception as e:
                log(f"[LOOP_ERROR] {e}\n{traceback.format_exc()}")

            # pacing
            elapsed = time.time() - start
            sleep_for = max(0.0, CONFIG["poll_seconds"] - elapsed)
            await asyncio.sleep(sleep_for)
    finally:
        try:
            await exchange.close()
        except Exception:
            pass

if __name__ == "__main__":
    log("Starting Crypto Volume Spike Screener...")
    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        log("Interrupted by user. Bye.")
    except Exception as e:
        log(f"FATAL: {e}\n{traceback.format_exc()}")
