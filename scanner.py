from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
import requests


def get_binance_altcoins():
    """取得所有幣安 USDT 山寨幣（排除大餅、二餅與槓桿代幣），並加入偽裝標頭防堵阻擋"""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
            " like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        res = requests.get(url, headers=headers, timeout=15)
        print(f"幣安 API 回應狀態碼: {res.status_code}")

        if res.status_code != 200:
            print(f"連線異常，內容片段: {res.text[:200]}")
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
        print(f"取得幣安幣種失敗: {e}")
        return []


def fetch_klines(symbol, interval="1d", limit=100):
    """抓取幣安 K 線資料"""
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
        return None


def save_data(results):
    """將結果寫入 data.json，確保每次都更新時間戳記"""
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
        print(f"開始進行全市場動態評分與智慧掃描...")

        for symbol in symbols:
            try:
                df_1d = fetch_klines
