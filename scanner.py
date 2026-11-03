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
        return

    results = []
    print(f"開始進行多週期 (1D, 4H, 1H) 掃描...")

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

            # 指標計算 (1D)
            delta = df_1d["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = (100 - (100 / (1 + (gain / loss)))).values

            exp1 = df_1d["Close"].ewm(span=12, adjust=False).mean()
            exp2 = df_1d["Close"].ewm(span=26, adjust=False).mean()
            macd_hist = ((exp1 - exp2) - (exp1 - exp2).ewm(span=9).mean()).values

            obv = (
                (
                    pd.Series(
                        np.where(
                            df_1d["Close"].diff() > 0,
                            volumes,
                            np.where(
                                df_1d["Close"].diff() < 0, -volumes, 0
                            ),
                        )
                    )
                )
                .cumsum()
                .values
            )

            low_min = df_1d["Low"].rolling(window=9).min()
            high_max = df_1d["High"].rolling(window=9).max()
            rsv = (
                (df_1d["Close"] - low_min) / (high_max - low_min) * 100
            ).fillna(50)
            k = rsv.ewm(com=2, adjust=False).mean().values
            d = k.ewm(com=2, adjust=False).mean().values

            def check_div(ind):
                if len(prices) < 30:
                    return False
                p_mins = []
                for i in range(2, len(prices) - 2):
                    if lows_1d[i] == min(lows_1d[i - 2 : i + 3]):
                        p_mins.append(i)
                if len(p_mins) < 2:
                    return False
                idx1, idx2 = p_mins[-2], p_mins[-1]
                if len(prices) - idx2 > 12:
                    return False
                return (prices[idx2] < prices[idx1]) and (
                    ind[idx2] > ind[idx1]
                )

            score = 0
            triggered = []
            active_timeframes = ["1D"]

            if check_div(obv):
                score +=
