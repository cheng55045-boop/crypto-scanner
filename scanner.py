from datetime import datetime
import json
import numpy as np
import pandas as pd
import requests


def scan_market():
    """CoinGecko 山寨幣掃描（單次執行版）"""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 80,
        "page": 1,
        "sparkline": "true",
        "price_change_percentage": "24h,7d",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        res = requests.get(url, params=params, headers=headers, timeout=15)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] CoinGecko 狀態碼: {res.status_code}")

        if res.status_code != 200:
            print(f"取得資料失敗: {res.text[:200]}")
            return []

        data = res.json()
        results = []

        for item in data:
            symbol = item["symbol"].upper()
            if symbol in ["BTC", "ETH", "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS", "USD1", "PYUSD", "RLUSD"]:
                continue

            current_price = float(item.get("current_price", 0) or 0)
            if current_price <= 0:
                continue

            change_24h = float(item.get("price_change_percentage_24h", 0) or 0)
            change_7d = float(item.get("price_change_percentage_7d_in_currency", 0) or 0)
            total_volume = float(item.get("total_volume", 0) or 0)
            market_cap = float(item.get("market_cap", 0) or 0)
            high_24h = float(item.get("high_24h", current_price * 1.01) or current_price)
            low_24h = float(item.get("low_24h", current_price * 0.99) or current_price)

            if total_volume < 2_000_000:
                continue
            if change_24h > 25:
                continue
            if market_cap < 20_000_000:
                continue

            sparkline = item.get("sparkline_in_7d", {}).get("price", [])
            if len(sparkline) < 12:
                continue

            df = pd.DataFrame({"Close": sparkline})
            df["High"] = df["Close"] * (1 + abs(change_24h) / 200)
            df["Low"] = df["Close"] * (1 - abs(change_24h) / 200)

            prices = df["Close"].values

            # RSI
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 
