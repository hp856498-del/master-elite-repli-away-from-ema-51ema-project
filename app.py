import streamlit as st
import pandas as pd
import ta
import ccxt
from datetime import datetime, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from zoneinfo import ZoneInfo

INDIA_TZ = ZoneInfo("Asia/Kolkata")
CANDLE_MINUTES = {
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "3h": 180,
    "4h": 240,
    "1d": 1440,
}
REFERENCE_EMA_PERIODS = {
    "1h": (51,),
    "4h": (51, 101),
}

# Page Setup
st.set_page_config(page_title="Crypto EMA Screener", layout="wide")
st.title("📊 Crypto Away From EMA Screener")

# Initialize Binance via CCXT
exchange = ccxt.binance({'enableRateLimit': True})
_worker_exchange = threading.local()


def get_worker_exchange():
    """Create one CCXT client per worker thread for safe parallel requests."""
    if not hasattr(_worker_exchange, "exchange"):
        _worker_exchange.exchange = ccxt.binance({'enableRateLimit': True})
    return _worker_exchange.exchange


def fetch_ohlcv_for_scan(
    worker_exchange,
    symbol,
    selected_tf,
    scan_mode,
    selected_ema,
    start_datetime,
    end_datetime,
):
    """Fetch enough candles for the selected date range, including past dates."""
    if scan_mode != "Date & Time Range Scan":
        return worker_exchange.fetch_ohlcv(symbol, timeframe=selected_tf, limit=200)

    now_ist = datetime.now(INDIA_TZ)
    target_end = min(end_datetime, now_ist)
    if start_datetime >= target_end:
        return []

    candle_minutes = CANDLE_MINUTES[selected_tf]
    candle_ms = candle_minutes * 60 * 1000
    warmup = pd.Timedelta(minutes=candle_minutes * (selected_ema + 5))
    since_ms = int((start_datetime - warmup).timestamp() * 1000)
    end_ms = int(target_end.timestamp() * 1000)
    candles = []

    # Binance returns at most 1000 candles per request. Continue from the
    # last returned candle so a full month is available on smaller timeframes.
    while since_ms <= end_ms:
        batch = worker_exchange.fetch_ohlcv(
            symbol,
            timeframe=selected_tf,
            since=since_ms,
            limit=1000,
        )
        if not batch:
            break

        candles.extend(batch)
        last_timestamp = batch[-1][0]
        if last_timestamp >= end_ms or len(batch) < 1000:
            break

        next_since = last_timestamp + candle_ms
        if next_since <= since_ms:
            break
        since_ms = next_since

    return [candle for candle in candles if candle[0] <= end_ms]


def ohlcv_to_closed_dataframe(ohlcv, selected_tf):
    """Convert exchange candles to a dataframe containing closed candles only."""
    if not ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(
        ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['datetime'] = pd.to_datetime(
        df['timestamp'], unit='ms', utc=True
    ).dt.tz_convert(INDIA_TZ)

    candle_duration = pd.Timedelta(minutes=CANDLE_MINUTES[selected_tf])
    now_ist = pd.Timestamp.now(tz=INDIA_TZ)
    return df[
        (df['datetime'] + candle_duration) <= now_ist
    ].copy()


def get_reference_ema_data(
    worker_exchange,
    symbol,
    scan_mode,
    start_datetime,
    end_datetime,
):
    """Fetch the 1h/4h 51 EMA values needed for each scanned candle."""
    reference_data = {}
    for reference_tf, ema_periods in REFERENCE_EMA_PERIODS.items():
        warmup_period = max(ema_periods)
        ohlcv = fetch_ohlcv_for_scan(
            worker_exchange,
            symbol,
            reference_tf,
            scan_mode,
            warmup_period,
            start_datetime,
            end_datetime,
        )
        reference_df = ohlcv_to_closed_dataframe(ohlcv, reference_tf)
        if len(reference_df) < warmup_period:
            continue

        reference_columns = ["datetime"]
        for ema_period in ema_periods:
            ema_column = f"ema_{ema_period}"
            reference_df[ema_column] = ta.trend.ema_indicator(
                close=reference_df["close"],
                window=ema_period,
            )
            reference_columns.append(ema_column)

        reference_data[reference_tf] = reference_df[reference_columns].dropna()

    return reference_data


def format_ema_proximity(candle, ema):
    """Return the closest edge with its signed percentage from the EMA."""
    if pd.isna(ema) or ema == 0:
        return "Unavailable"

    high_gap = (candle["high"] - ema) / abs(ema) * 100
    low_gap = (candle["low"] - ema) / abs(ema) * 100
    nearest_edge, nearest_gap = min(
        [("High", high_gap), ("Low", low_gap)],
        key=lambda item: abs(item[1]),
    )
    return f"{nearest_edge} ({nearest_gap:+.2f}%)"


def highlight_close_coin(row):
    """Highlight a coin when either reference EMA is less than 1% away."""
    styles = pd.Series("", index=row.index)
    is_close = False
    for reference_tf in REFERENCE_EMA_PERIODS:
        proximity = row.get(f"{reference_tf} 51 EMA Proximity", "")
        if not isinstance(proximity, str) or "(" not in proximity:
            continue
        try:
            gap = float(proximity.rsplit("(", 1)[1].rstrip("%)"))
        except ValueError:
            continue
        if abs(gap) < 1:
            is_close = True
            break

    if is_close:
        styles["Coin"] = "color: red; font-weight: 700"
    return styles


@st.cache_data(ttl=300)
def get_top_pairs(limit):
    tickers = exchange.fetch_tickers()
    usdt_pairs = [
        symbol for symbol in tickers.keys() 
        if symbol.endswith('/USDT') and 'UP/' not in symbol and 'DOWN/' not in symbol
    ]
    sorted_pairs = sorted(
        usdt_pairs, 
        key=lambda x: tickers[x].get('quoteVolume', 0) if tickers[x].get('quoteVolume') else 0, 
        reverse=True
    )
    return sorted_pairs[:limit]

# Sidebar Controls
st.sidebar.header("⚙️ Screener Controls")

# 1. Total Coins Selection
limit_options = [50, 100, 200, 400, 600, 700]
selected_limit = st.sidebar.selectbox("Top Coins Limit (by Volume)", limit_options, index=1)

# 2. Timeframe Selection
tf_map = {
    "15min": "15m",
    "30minute": "30m",
    "1 hour": "1h",
    "2hour": "2h",
    "3hour": "3h",
    "4 hour": "4h",
    "1 day": "1d"
}
selected_tf_label = st.sidebar.selectbox("Timeframe", list(tf_map.keys()), index=2)
selected_tf = tf_map[selected_tf_label]

# 3. EMA Selection
selected_ema = st.sidebar.selectbox("EMA Period", [2, 3, 4, 5, 6], index=1)

# 4. Scan Mode
scan_mode = st.sidebar.radio(
    "Scan Mode", 
    ["Latest Closed Candle", "Date & Time Range Scan"]
)

start_datetime = None
end_datetime = None

st.sidebar.subheader("📅 Date & Time Filter (IST)")
st.sidebar.caption("All dates and times use India Standard Time.")
selected_date = st.sidebar.date_input("Select Date", datetime.now(INDIA_TZ).date())

use_time_range = st.sidebar.checkbox("Filter by Time Range?", value=True)
if use_time_range:
    col1, col2 = st.sidebar.columns(2)
    start_t = col1.time_input("Start Time", time(15, 30))
    end_t = col2.time_input("End Time", time(19, 30))
    start_datetime = datetime.combine(selected_date, start_t, tzinfo=INDIA_TZ)
    end_datetime = datetime.combine(selected_date, end_t, tzinfo=INDIA_TZ)
else:
    start_datetime = datetime.combine(selected_date, time(0, 0), tzinfo=INDIA_TZ)
    end_datetime = datetime.combine(selected_date, time(23, 59, 59), tzinfo=INDIA_TZ)

def check_away_from_ema(df, ema_col):
    df['max_val'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['min_val'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    df['is_away'] = (df['min_val'] > df[ema_col]) | (df['max_val'] < df[ema_col])
    return df


def scan_symbol(symbol, selected_tf, selected_ema, scan_mode, start_datetime, end_datetime):
    """Fetch and scan one symbol. Called in parallel by the screener."""
    try:
        worker_exchange = get_worker_exchange()
        ohlcv = fetch_ohlcv_for_scan(
            worker_exchange,
            symbol,
            selected_tf,
            scan_mode,
            selected_ema,
            start_datetime,
            end_datetime,
        )
        if not ohlcv:
            return []

        # Binance can return the currently forming candle as the last row.
        # Remove it before calculating EMA or returning a match.
        closed_df = ohlcv_to_closed_dataframe(ohlcv, selected_tf)
        if len(closed_df) < selected_ema + 5:
            return []
        df = closed_df

        ema_name = f"EMA_{selected_ema}"
        df[ema_name] = ta.trend.ema_indicator(
            close=df['close'],
            window=selected_ema
        )
        df = check_away_from_ema(df, ema_name)

        if scan_mode == "Latest Closed Candle":
            rows_to_scan = [df.iloc[-1]]
        else:
            filtered_df = df[
                (df['datetime'] >= start_datetime)
                & (df['datetime'] <= end_datetime)
            ]
            rows_to_scan = filtered_df[filtered_df['is_away'] == True].to_dict('records')

        reference_data = get_reference_ema_data(
            worker_exchange,
            symbol,
            scan_mode,
            start_datetime,
            end_datetime,
        )
        matched_rows = []
        for row in rows_to_scan:
            if not row['is_away']:
                continue
            result = {
                "Coin": symbol,
                "Time": row['datetime'].strftime('%Y-%m-%d %H:%M'),
                "Open": row['open'],
                "High": row['high'],
                "Low": row['low'],
                "Close": row['close'],
                ema_name: round(row[ema_name], 4),
                "Position": "Above EMA" if row['low'] > row[ema_name] else "Below EMA"
            }

            # Match each scanned candle to the most recent completed 1h/4h
            # candle so historical range scans use the EMA from that moment.
            for reference_tf in REFERENCE_EMA_PERIODS:
                reference_df = reference_data.get(reference_tf)
                ema_value = None
                reference_match = None
                if reference_df is not None and not reference_df.empty:
                    reference_match = pd.merge_asof(
                        pd.DataFrame({"datetime": [row["datetime"]]}),
                        reference_df.sort_values("datetime"),
                        on="datetime",
                        direction="backward",
                    )
                    ema_value = reference_match.iloc[0]["ema_51"]

                result[f"{reference_tf} 51 EMA Proximity"] = (
                    format_ema_proximity(row, ema_value)
                )

                if reference_tf == "4h":
                    trend = "Unavailable"
                    coin_marker = ""
                    if reference_match is not None:
                        ema_51 = reference_match.iloc[0]["ema_51"]
                        ema_101 = reference_match.iloc[0]["ema_101"]
                        if pd.notna(ema_51) and pd.notna(ema_101):
                            trend = "Bullish" if ema_51 > ema_101 else "Bearish"
                            coin_marker = "🟢" if trend == "Bullish" else "🔴"

                    result["4h 51/101 EMA Trend"] = trend
                    result["Coin"] = (
                        f"{coin_marker} {symbol}".strip()
                        if coin_marker
                        else symbol
                    )

            matched_rows.append(result)
        return matched_rows
    except Exception:
        return []


# Main Execution
if st.button("🚀 Run Screener"):
    with st.spinner("Scanning coins..."):
        top_symbols = get_top_pairs(selected_limit)
        matched_results = []
        progress_bar = st.progress(0)

        # Eight workers keep the public API responsive while avoiding an
        # excessive number of concurrent requests.
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    scan_symbol,
                    symbol,
                    selected_tf,
                    selected_ema,
                    scan_mode,
                    start_datetime,
                    end_datetime,
                )
                for symbol in top_symbols
            ]
            for idx, future in enumerate(as_completed(futures), start=1):
                matched_results.extend(future.result())
                progress_bar.progress(idx / len(futures))

        st.subheader("📋 Scan Results")
        if matched_results:
            result_df = pd.DataFrame(matched_results)
            st.success(f"Total {len(result_df)} instances paye gaye!")
            st.dataframe(
                result_df.style.apply(highlight_close_coin, axis=1),
                use_container_width=True,
            )
        else:
            st.warning("Is condition me koi coin nahi mila.")