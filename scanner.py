from datetime import datetime
import json
import time
import numpy as np
import pandas as pd
import requests


BASE_URL = "https://api.binance.com"
EXCLUDE = {"BTC", "ETH", "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDE", "USDS", "USD1", "PYUSD", "RLUSD"}


def get_24h_tickers():
    """取得所有 USDT 交易對的 24h 行情"""
    url = f"{BASE_URL}/api/v3/ticker/24hr"
    res = requests.get(url, timeout=15)
    res.raise_for_status()
    data = res.json()

    tickers = []
    for t in data:
        symbol = t["symbol"]
        if not symbol.endswith("USDT"):
            continue
        base = symbol[:-4]
        if base in EXCLUDE:
            continue

        quote_volume = float(t.get("quoteVolume", 0))
        if quote_volume < 5_000_000:  # 成交量過低略過
            continue

        tickers.append({
            "symbol": symbol,
            "base": base,
            "price": float(t["lastPrice"]),
            "change_24h": float(t["priceChangePercent"]),
            "high_24h": float(t["highPrice"]),
            "low_24h": float(t["lowPrice"]),
            "volume_24h": quote_volume,
            "trades": int(t.get("count", 0)),
        })

    # 依成交量排序，取前 40 檔做進一步分析
    tickers.sort(key=lambda x: x["volume_24h"], reverse=True)
    return tickers[:40]


def get_klines(symbol, interval="1h", limit=80):
    """取得 K 線資料"""
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    res = requests.get(url, params=params, timeout=10)
    res.raise_for_status()
    raw = res.json()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def calc_indicators(df):
    """計算 RSI、MACD、KD"""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50).values

    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = (macd_line - signal).values

    # KD
    low_min = low.rolling(9, min_periods=1).min()
    high_max = high.rolling(9, min_periods=1).max()
    rsv = ((close - low_min) / (high_max - low_min).replace(0, np.nan) * 100).fillna(50)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()

    return {
        "rsi": rsi,
        "macd_hist": macd_hist,
        "k": k.values,
        "d": d.values,
        "close": close.values,
    }


def score_coin(ind, change_24h):
    """根據指標給分"""
    score = 0
    triggered = []
    rsi = ind["rsi"]
    macd_hist = ind["macd_hist"]
    k = ind["k"]
    d = ind["d"]
    close = ind["close"]

    # RSI
    if len(rsi) >= 5 and close[-1] < close[-4] and rsi[-1] > rsi[-4] + 2:
        score += 3
        triggered.append("RSI背離")
    if rsi[-1] < 30:
        score += 2
        triggered.append("RSI超賣")
    elif 30 <= rsi[-1] <= 42:
        score += 1

    # MACD
    if len(macd_hist) >= 4:
        if macd_hist[-1] > macd_hist[-2] > macd_hist[-3]:
            score += 2
            triggered.append("MACD動能向上")
        if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
            score += 2
            triggered.append("MACD翻正")

    # KD
    if k[-1] > d[-1] and k[-2] <= d[-2]:
        score += 2
        triggered.append("KD金叉")
    if k[-1] < 30 and d[-1] < 35:
        score += 1
        triggered.append("KD低檔")

    # 過熱懲罰
    if change_24h > 12:
        score -= 2
    if rsi[-1] > 75:
        score -= 2
        triggered.append("RSI過熱")

    return score, triggered


def scan_market():
    print("正在從 Binance 取得行情...")
    tickers = get_24h_tickers()
    print(f"取得 {len(tickers)} 檔 USDT 交易對，開始分析...\n")

    results = []
    for i, t in enumerate(tickers, 1):
        try:
            df = get_klines(t["symbol"], interval="1h", limit=80)
            if len(df) < 30:
                continue

            ind = calc_indicators(df)
            score, triggered = score_coin(ind, t["change_24h"])

            if score < 3:
                continue

            if score >= 8:
                status = "重點關注"
            elif score >= 6:
                status = "蓄勢待發"
            else:
                status = "觀察中"

            results.append({
                "code": t["base"],
                "symbol": t["symbol"],
                "score": int(score),
                "status": status,
                "indicators": triggered,
                "current_price": round(t["price"], 6),
                "change_24h": round(t["change_24h"], 2),
                "volume_24h": round(t["volume_24h"]),
                "high_24h": round(t["high_24h"], 6),
                "low_24h": round(t["low_24h"], 6),
            })

            print(f"[{i:2d}/{len(tickers)}] {t['base']:<8} 分數:{score:2d}  {status}")

            time.sleep(0.15)  # 避免請求過快

        except Exception as e:
            print(f"[{i:2d}] {t['base']} 分析失敗: {e}")
            continue

    results.sort(key=lambda x: (x["score"], x["volume_24h"]), reverse=True)
    top_results = results[:15]

    print(f"\n處理完成，成功選出 {len(top_results)} 檔標的。")
    save_data(top_results)


def save_data(results):
    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_count": len(results),
        "source": "Binance Public API",
        "stocks": results,
    }
    with open("/workspace/artifacts/data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"資料寫入完成！共 {len(results)} 檔標的。")
    print(f"檔案位置: /workspace/artifacts/data.json")


if __name__ == "__main__":
    scan_market()
