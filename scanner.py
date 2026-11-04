from datetime import datetime
import json
import os
import time
import numpy as np
import pandas as pd
import requests


def get_binance_altcoins():
    """取得所有幣安 USDT 山寨幣"""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
            " like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        res = requests.get(url, headers=headers, timeout=15)
        print(f"幣安 exchangeInfo 回應狀態碼: {res.status_code}")
        if res.status_code != 200:
            print(f"取得幣安幣種失敗，內容: {res.text[:200]}")
            return []

        data = res.json()
        altcoins = []
        for s in data.get("symbols", []):
            symbol = s["symbol"]
            if symbol.endswith("USDT") and symbol not in [
                "BTCUSDT",
                "ETHUSDT",
            ]:
                if not any(
                    x in symbol for x in ["UP", "DOWN", "BULL", "BEAR"]
                ):
                    altcoins.append(symbol)
        print(f"成功取得 {len(altcoins)} 檔山寨幣")
        return altcoins
    except Exception as e:
        print(f"取得幣安幣種發生例外: {e}")
        return []


def fetch_klines(symbol, interval="1d", limit=100):
    """抓取幣安 K 線資料，並印出詳細失敗原因"""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
            " like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            # 印出前幾筆失敗的狀態碼，讓我們知道是不是被 403 擋掉
            print(f"[{symbol}] 抓取 {interval} 失敗，狀態碼: {res.status_code}")
            return None
        res_json = res.json()
        if not isinstance(res_json, list) or len(res_json) == 0:
            return None
        df = pd.DataFrame(
            res_json,
            columns=[
                "OpenTime",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "CloseTime",
                "QuoteAssetVolume",
                "NumberOfTrades",
                "TakerBuyBaseAssetVolume",
                "TakerBuyQuoteAssetVolume",
                "Ignore",
            ],
        )
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"[{symbol}] 抓取例外錯誤: {e}")
        return None


def save_data(results):
    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_count": len(results),
        "stocks": results,
    }
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print(
        f"資料寫入完成！updated_at: {output['updated_at']}, 共 {len(results)} 檔標的。"
    )


def scan_market():
    try:
        symbols = get_binance_altcoins()
        if not symbols:
            print("未取得任何幣種，寫入空資料並更新時間。")
            save_data([])
            return

        results = []
        print(f"開始進行市場掃描...")

        # 為了避免洗版，我們只印出前 3 檔的請求結果來檢查
        for idx, symbol in enumerate(symbols[:30]):  # 先測試前 30 檔
            try:
                df_1d = fetch_klines(symbol, interval="1d", limit=100)
                df_4h = fetch_klines(symbol, interval="4h", limit=100)
                df_1h = fetch_klines(symbol, interval="1h", limit=100)
                time.sleep(0.1)

                if (
                    df_1d is None
                    or len(df_1d) < 60
                    or df_4h is None
                    or df_1h is None
                ):
                    continue

                current_close = float(df_1d["Close"].iloc[-1])
                price_change_24h = float(
                    (df_1d["Close"].iloc[-1] - df_1d["Close"].iloc[-2])
                    / df_1d["Close"].iloc[-2]
                    * 100
                )

                results.append(
                    {
                        "code": symbol.replace("USDT", ""),
                        "score": 3,
                        "entry_score": 6,
                        "status": "蓄勢待發",
                        "indicators": ["常規動能追蹤"],
                        "timeframes": ["1D", "4H"],
                        "current_price": round(current_close, 4),
                        "change_24h": round(price_change_24h, 2),
                    }
                )
            except Exception as inner_e:
                continue

        results = sorted(results, key=lambda x: x["entry_score"], reverse=True)
        save_data(results[:15])

    except Exception as e:
        print(f"掃描過程發生未預期錯誤: {e}")
        save_data([])


if __name__ == "__main__":
    scan_market()
