from datetime import datetime
import json
import numpy as np
import pandas as pd
import requests


BASE_URL = "https://fapi.bitunix.com"

# 排除清單：主流幣 + 穩定幣 + 傳統金融 / 商品 / 股票
EXCLUDE = {
    # 主流與穩定幣
    "BTC", "ETH", "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDE", "USDS", "USD1", "PYUSD", "RLUSD",
    # 黃金 / 白銀 / 商品
    "XAU", "XAG", "XAUT", "PAXG", "GLD", "SLV", "USDG", "USDGO", "USDY",
    # 股票 / ETF / 指數
    "SOXL", "SKHYNIX", "SNDK", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "CL", "NG", "HG", "SI", "GC", "ES", "NQ", "YM", "RTY",
    # 其他非加密
    "FIGR_HELOC", "WBT", "LEO",
}


def get_tickers():
    """取得 Bitunix 所有合約交易對的 24h 行情"""
    url = f"{BASE_URL}/api/v1/futures/market/tickers"
    res = requests.get(url, timeout=15)
    res.raise_for_status()
    data = res.json()

    if data.get("code") != 0:
        print(f"API 錯誤: {data.get('msg')}")
        return []

    tickers = []
    for t in data.get("data", []):
        symbol = t.get("symbol", "")
        if not symbol.endswith("USDT"):
            continue
        base = symbol[:-4]
        if base in EXCLUDE:
            continue

        quote_vol = float(t.get("quoteVol", 0) or 0)
        if quote_vol < 1_000_000:
            continue

        tickers.append({
            "symbol": symbol,
            "base": base,
            "price": float(t.get("lastPrice", 0) or 0),
            "change_24h": 0,
            "high_24h": float(t.get("high", 0) or 0),
            "low_24h": float(t.get("low", 0) or 0),
            "open_24h": float(t.get("open", 0) or 0),
            "volume_24h": quote_vol,
        })

    for t in tickers:
        if t["open_24h"] > 0:
            t["change_24h"] = (t["price"] - t["open_24h"]) / t["open_24h"] * 100

    tickers.sort(key=lambda x: x["volume_24h"], reverse=True)
    return tickers[:40]


def score_ticker(t):
    """簡易評分（基於 24h 數據）"""
    score = 0
    triggered = []
    change = t["change_24h"]
    high = t["high_24h"]
    low = t["low_24h"]
    price = t["price"]

    if -15 < change < -3:
        score += 2
        triggered.append("跌深")
    elif -3 <= change <= 3:
        score += 1
        triggered.append("盤整")

    if high > 0 and low > 0 and price > 0:
        span = (high - low) / price
        if span < 0.06:
            score += 2
            triggered.append("波動收斂")

    if t["volume_24h"] > 50_000_000:
        score += 2
        triggered.append("量能活躍")
    elif t["volume_24h"] > 10_000_000:
        score += 1

    if change > 12:
        score -= 2

    score = max(0, min(score, 10))
    return score, triggered


def scan_market():
    print("正在從 Bitunix 取得行情...")
    tickers = get_tickers()
    print(f"取得 {len(tickers)} 檔交易對\n")

    results = []
    for t in tickers:
        score, triggered = score_ticker(t)
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

    results.sort(key=lambda x: (x["score"], x["volume_24h"]), reverse=True)
    top_results = results[:15]

    print(f"成功選出 {len(top_results)} 檔標的")
    save_data(top_results)
    return top_results


def save_data(results):
    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_count": len(results),
        "source": "Bitunix Futures",
        "stocks": results,
    }
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"資料寫入完成！共 {len(results)} 檔標的。")


if __name__ == "__main__":
    scan_market()
