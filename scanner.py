from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
import requests


def get_binance_altcoins():
    """取得所有幣安 USDT 山寨幣（排除大餅、二餅與槓桿代幣）"""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        res = requests.get(url, timeout=15).json()
        altcoins = []
        for s in res.get("symbols", []):
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
    try:
        res = requests.get(url, timeout=10).json()
        if not isinstance(res, list) or len(res) == 0:
            return None
        df = pd.DataFrame(
            res,
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
        df["Open"] = df["Open"].astype(float)
        df["High"] = df["High"].astype(float)
        df["Low"] = df["Low"].astype(float)
        df["Close"] = df["Close"].astype(float)
        df["Volume"] = df["Volume"].astype(float)
        return df
    except Exception as e:
        return None


def scan_market():
    symbols = get_binance_altcoins()
    if not symbols:
        print("未取得任何幣種，結束掃描。")
        # 即使沒抓到幣種，也強制更新時間
        save_data([])
        return

    results = []
    print(f"開始進行高質量多週期 (1D, 4H, 1H) 掃描...")

    for symbol in symbols:
        try:
            df_1d = fetch_klines(symbol, interval="1d", limit=100)
            df_4h = fetch_klines(symbol, interval="4h", limit=100)
            df_1h = fetch_klines(symbol, interval="1h", limit=100)

            if (
                df_1d is None
                or len(df_1d) < 60
                or df_4h is None
                or df_1h is None
            ):
                continue

            prices = df_1d["Close"].values
            lows_1d = df_1d["Low"].values
            volumes = df_1d["Volume"].values
            current_close = float(df_1d["Close"].iloc[-1])

            price_change_24h = float(
                (df_1d["Close"].iloc[-1] - df_1d["Close"].iloc[-2])
                / df_1d["Close"].iloc[-2]
                * 100
            )

            # 技術指標計算 (1D)
            delta = df_1d
