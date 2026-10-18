
import os
import time
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import requests

NY = ZoneInfo("America/New_York")
BASE = "https://data.alpaca.markets/v2/stocks/bars"
KEY = os.environ["APCA_API_KEY_ID"]
SECRET = os.environ["APCA_API_SECRET_KEY"]
HEADERS = {
    "APCA-API-KEY-ID": KEY,
    "APCA-API-SECRET-KEY": SECRET,
}

def notify_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram not configured; see CSV report.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for i in range(0, len(message), 3800):
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": message[i:i+3800]},
            timeout=20,
        )
        r.raise_for_status()

def get_bars(symbols, start, end):
    """Fetch 5-minute bars from Alpaca's free IEX feed."""
    out = []
    for offset in range(0, len(symbols), 50):
        batch = symbols[offset:offset + 50]
        token = None
        while True:
            params = {
                "symbols": ",".join(batch),
                "timeframe": "5Min",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": 10000,
                "feed": "iex",
                "adjustment": "raw",
                "sort": "asc",
            }
            if token:
                params["page_token"] = token
            response = requests.get(
                BASE, headers=HEADERS, params=params, timeout=60
            )
            if response.status_code == 429:
                time.sleep(5)
                continue
            response.raise_for_status()
            payload = response.json()
            for symbol, bars in payload.get("bars", {}).items():
                for bar in bars:
                    out.append({
                        "symbol": symbol,
                        "t": bar["t"],
                        "open": bar["o"],
                        "high": bar["h"],
                        "low": bar["l"],
                        "close": bar["c"],
                        "volume": bar["v"],
                    })
            token = payload.get("next_page_token")
            if not token:
                break
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(NY)
    return df.sort_values(["symbol", "t"]).reset_index(drop=True)

def indicators(g):
    g = g.sort_values("t").copy()
    close, high, low = g.close, g.high, g.low

    g["ema10"] = close.ewm(span=10, adjust=False).mean()
    g["ema20"] = close.ewm(span=20, adjust=False).mean()
    g["ema30"] = close.ewm(span=30, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    g["rsi"] = 100 - 100 / (1 + rs)
    g.loc[(avg_loss == 0) & (avg_gain > 0), "rsi"] = 100

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * plus_dm.ewm(
        alpha=1/14, adjust=False, min_periods=14
    ).mean() / atr
    minus_di = 100 * minus_dm.ewm(
        alpha=1/14, adjust=False, min_periods=14
    ).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    g["adx"] = dx.ewm(
        alpha=1/14, adjust=False, min_periods=14
    ).mean()

    # Session VWAP, reset each New York trading date.
    session = g["t"].dt.date
    typical = (high + low + close) / 3
    g["vwap"] = (
        (typical * g.volume).groupby(session).cumsum()
        / g.volume.groupby(session).cumsum().replace(0, float("nan"))
    )
    g["session_date"] = session
    return g

def evaluate(symbol, raw, scan_date):
    g = indicators(raw)
    today = g[g.session_date == scan_date].copy()
    if today.empty:
        return None

    # Use the completed 9:40-9:45 bar, labelled 9:40 by Alpaca.
    today["clock"] = today["t"].dt.strftime("%H:%M")
    target = today[today.clock == "09:40"]
    if target.empty:
        return None
    current = target.iloc[-1]

    earlier = g[g.session_date < scan_date]
    if earlier.empty:
        return None

    prev_high = earlier.groupby("session_date").high.max().iloc[-1]

    # Opening volume = 09:30, 09:35, and 09:40 bars.
    g["clock"] = g["t"].dt.strftime("%H:%M")
    opening = g[g.clock.isin(["09:30", "09:35", "09:40"])]
    by_day = opening.groupby("session_date").volume.sum()
    history = by_day[by_day.index < scan_date].tail(20)

    if len(history) < 10 or len(opening[opening.session_date == scan_date]) < 3:
        return None

    avg_opening_volume = history.mean()
    current_opening_volume = by_day.get(scan_date, float("nan"))
    rel_vol = current_opening_volume / avg_opening_volume if avg_opening_volume else 0

    # Require a fresh close crossing at the scan candle.
    previous_bars = today[today.t < current.t]
    if previous_bars.empty:
        return None
    previous_close = previous_bars.iloc[-1].close

    breakout = current.close > prev_high and previous_close <= prev_high
    vol_ok = rel_vol >= 1.5
    adx_ok = pd.notna(current.adx) and current.adx > 25
    ema_ok = current.ema10 > current.ema20 > current.ema30
    vwap_ok = current.close > current.vwap
    rsi_ok = pd.notna(current.rsi) and current.rsi > 50

    checks = [breakout, vol_ok, adx_ok, ema_ok, vwap_ok, rsi_ok]
    return {
        "symbol": symbol,
        "scan_date": str(scan_date),
        "price": round(float(current.close), 4),
        "previous_day_high": round(float(prev_high), 4),
        "opening_volume": int(current_opening_volume),
        "avg_opening_volume_20d": round(float(avg_opening_volume), 1),
        "relative_volume": round(float(rel_vol), 2),
        "adx14": round(float(current.adx), 2) if pd.notna(current.adx) else None,
        "ema10": round(float(current.ema10), 4),
        "ema20": round(float(current.ema20), 4),
        "ema30": round(float(current.ema30), 4),
        "vwap": round(float(current.vwap), 4),
        "rsi14": round(float(current.rsi), 2) if pd.notna(current.rsi) else None,
        "breakout_pass": bool(breakout),
        "volume_pass": bool(vol_ok),
        "adx_pass": bool(adx_ok),
        "ema_pass": bool(ema_ok),
        "vwap_pass": bool(vwap_ok),
        "rsi_pass": bool(rsi_ok),
        "signal": all(checks),
    }

def main():
    now = datetime.now(NY)
    scan_date = now.date()

    # Skip US market holidays.
    cal = mcal.get_calendar("NYSE")
    if cal.schedule(start_date=scan_date, end_date=scan_date).empty:
        print(f"{scan_date}: US market holiday; no scan.")
        return

    # Free historical endpoint only exposes data through at least 15 minutes ago.
    # Query exactly through the completed 9:45 AM ET candle.
    end = datetime.combine(scan_date, dtime(9, 45), tzinfo=NY)
    start = end - timedelta(days=45)

    symbols = []
    with open("symbols.txt", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if s and not s.startswith("#"):
                symbols.append(s)
    symbols = sorted(set(symbols))
    if not symbols:
        raise RuntimeError("symbols.txt is empty")

    print(f"Fetching IEX bars for {len(symbols)} symbols.")
    bars = get_bars(symbols, start, end)
    results = []
    if not bars.empty:
        for symbol, raw in bars.groupby("symbol"):
            try:
                row = evaluate(symbol, raw, scan_date)
                if row:
                    results.append(row)
            except Exception as exc:
                print(f"{symbol}: skipped ({exc})")

    report = pd.DataFrame(results)
    if report.empty:
        report = pd.DataFrame(columns=[
            "symbol", "scan_date", "price", "previous_day_high",
            "relative_volume", "adx14", "ema10", "ema20", "ema30",
            "vwap", "rsi14", "signal"
        ])
    else:
        report = report.sort_values(
            ["signal", "relative_volume"], ascending=[False, False]
        )

    os.makedirs("reports", exist_ok=True)
    path = f"reports/usa_breakout_{scan_date}.csv"
    report.to_csv(path, index=False)

    matches = report[report.signal == True] if "signal" in report else pd.DataFrame()
    msg = [
        f"USA Bullish Breakout Scanner — {scan_date}",
        "Data: Alpaca IEX only; not consolidated US-market volume.",
        f"Scanned: {len(symbols)} configured symbols",
        f"Matches: {len(matches)}",
    ]
    if not matches.empty:
        for _, r in matches.iterrows():
            msg.append(
                f"{r['symbol']} | ${r['price']} | RVOL {r['relative_volume']}x "
                f"| ADX {r['adx14']} | RSI {r['rsi14']}"
            )
    else:
        msg.append("No symbols passed all six conditions.")
    print("\n".join(msg))
    notify_telegram("\n".join(msg))
    print(f"CSV report saved to {path}")

if __name__ == "__main__":
    main()
