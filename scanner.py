from datetime import datetime
import json
import time
import numpy as np
import pandas as pd
import requests


BASE_URL = "https://fapi.bitunix.com"

EXCLUDE = {
    "BTC", "ETH", "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDE", "USDS", "USD1", "PYUSD", "RLUSD",
    "XAU", "XAG", "XAUT", "PAXG", "GLD", "SLV", "USDG", "USDGO", "USDY",
    "CL", "NG", "HG", "SI", "GC", "BZ", "WTI", "BRENT",
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AMD", "INTC", "NFLX",
    "SPY", "QQQ", "IWM", "DIA", "SOXL", "SOXS", "TQQQ", "SQQQ", "SPCX",
    "SKHYNIX", "SNDK", "ASML", "TSM", "BABA", "JD", "PDD", "NIO",
    "ES", "NQ", "YM", "RTY",
    "AAOI", "ABBV", "ACN", "ADBE", "AKAM", "ALAB", "AMAT", "AMGN", "ANET", "APLD",
    "FIGR_HELOC", "WBT", "LEO", "4STOCK", "MU",
}


def get_tickers():
    url = f"{BASE_URL}/api/v1/futures/market/tickers"
    res = requests.get(url, timeout=15)
    res.raise_for_status()
    data = res.json()
    if data.get("code") != 0:
        return []

    tickers = []
    for t in data.get("data", []):
        symbol = t.get("symbol", "")
        if not symbol.endswith("USDT"):
            continue
        base = symbol[:-4]
        if base in EXCLUDE or any(kw in base for kw in ["STOCK", "ETF"]):
            continue
        quote_vol = float(t.get("quoteVol", 0) or 0)
        if quote_vol < 2_000_000:
            continue
        open_p = float(t.get("open", 0) or 0)
        last_p = float(t.get("lastPrice", 0) or 0)
        change = ((last_p - open_p) / open_p * 100) if open_p > 0 else 0
        tickers.append({
            "symbol": symbol, "base": base, "price": last_p,
            "change_24h": change, "volume_24h": quote_vol,
            "high_24h": float(t.get("high", 0) or 0),
            "low_24h": float(t.get("low", 0) or 0),
        })
    tickers.sort(key=lambda x: x["volume_24h"], reverse=True)
    return tickers[:30]


def get_klines(symbol, interval="1h", limit=80):
    url = f"{BASE_URL}/api/v1/futures/market/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if data.get("code") != 0:
            return None
        raw = data.get("data", [])
        if not raw or len(raw) < 30:
            return None
        df = pd.DataFrame(raw)
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        return df
    except Exception:
        return None


def calc_indicators(df):
    if df is None or len(df) < 30:
        return None
    close = df["close"]
    high = df["high"]
    low = df["low"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50).values

    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = (macd_line - signal).values

    low_min = low.rolling(9, min_periods=1).min()
    high_max = high.rolling(9, min_periods=1).max()
    rsv = ((close - low_min) / (high_max - low_min).replace(0, np.nan) * 100).fillna(50)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()

    return {"rsi": rsi, "macd_hist": macd_hist, "k": k.values, "d": d.values, "close": close.values}


def score_timeframe(ind, change_24h):
    if ind is None:
        return 0, []
    score = 0
    triggered = []
    rsi, macd_hist, k, d, close = ind["rsi"], ind["macd_hist"], ind["k"], ind["d"], ind["close"]

    if len(rsi) >= 5 and close[-1] < close[-4] and rsi[-1] > rsi[-4] + 2:
        score += 2
        triggered.append("RSI背離")
    if rsi[-1] < 32:
        score += 2
        triggered.append("RSI超賣")
    elif 32 <= rsi[-1] <= 45:
        score += 1

    if len(macd_hist) >= 4:
        if macd_hist[-1] > macd_hist[-2] > macd_hist[-3]:
            score += 2
            triggered.append("MACD向上")
        if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
            score += 1
            triggered.append("MACD翻正")

    if k[-1] > d[-1] and k[-2] <= d[-2]:
        score += 2
        triggered.append("KD金叉")
    if k[-1] < 30:
        score += 1
        triggered.append("KD低檔")

    if change_24h > 10:
        score -= 1
    if rsi[-1] > 72:
        score -= 2
    return max(0, score), triggered


def simple_backtest(ind):
    if ind is None or len(ind["k"]) < 25:
        return None
    k, d, close = ind["k"], ind["d"], ind["close"]
    results = []
    for i in range(10, len(k) - 5):
        if k[i] > d[i] and k[i-1] <= d[i-1]:
            results.append((close[i+5] - close[i]) / close[i] * 100)
    return round(float(np.mean(results)), 2) if results else None


def scan_market():
    print("正在從 Bitunix 取得行情...")
    tickers = get_tickers()
    print(f"取得 {len(tickers)} 檔，開始分析...\n")

    results = []
    for i, t in enumerate(tickers, 1):
        try:
            df_1h = get_klines(t["symbol"], "1h", 80)
            ind_1h = calc_indicators(df_1h)
            score_1h, trig_1h = score_timeframe(ind_1h, t["change_24h"])

            df_4h = get_klines(t["symbol"], "4h", 60)
            ind_4h = calc_indicators(df_4h)
            score_4h, trig_4h = score_timeframe(ind_4h, t["change_24h"])

            total_score = max(0, min(int(score_1h * 0.6 + score_4h * 0.4), 10))
            bt = simple_backtest(ind_1h)
            all_triggered = list(set(trig_1h + trig_4h))

            if total_score < 3:
                continue

            status = "重點關注" if total_score >= 8 else ("蓄勢待發" if total_score >= 6 else "觀察中")

            results.append({
                "code": t["base"], "symbol": t["symbol"],
                "score": total_score, "score_1h": score_1h, "score_4h": score_4h,
                "status": status, "indicators": all_triggered,
                "current_price": round(t["price"], 6),
                "change_24h": round(t["change_24h"], 2),
                "volume_24h": round(t["volume_24h"]),
                "backtest_avg": bt,
            })
            print(f"[{i:2d}] {t['base']:<8} 總分:{total_score}  1h:{score_1h}  4h:{score_4h}")
            time.sleep(0.15)
        except Exception as e:
            print(f"[{i}] {t['base']} 失敗: {e}")

    results.sort(key=lambda x: (x["score"], x["volume_24h"]), reverse=True)
    top = results[:15]
    print(f"\n成功選出 {len(top)} 檔")
    save_data(top)


def save_data(results):
    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_count": len(results),
        "source": "Bitunix Futures (Multi-TF)",
        "stocks": results,
    }
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"資料寫入完成，共 {len(results)} 檔")


if __name__ == "__main__":
    scan_market()
