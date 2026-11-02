import json
from datetime import datetime
import pandas as pd
import requests


def get_binance_altcoins():
    """取得所有幣安 USDT 山寨幣（排除大餅、二餅與槓桿代幣）"""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        res = requests.get(url, timeout=10).json()
        altcoins = []
        for s in res["symbols"]:
            symbol = s["symbol"]
            if symbol.endswith("USDT") and symbol not in [
                "BTCUSDT",
                "ETHUSDT",
            ]:
                if not any(
                    x in symbol for x in ["UP", "DOWN", "BULL", "BEAR"]
                ):
                    altcoins.append(symbol)
        return altcoins
    except Exception as e:
        print(f"取得幣安幣種失敗: {e}")
        return []


def fetch_klines(symbol, interval="1h", limit=200):
    """抓取幣安 K 線資料"""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        res = requests.get(url, timeout=10).json()
        if not isinstance(res, list):
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
    print(f"總共取得 {len(symbols)} 檔山寨幣，開始掃描...")

    system_a_results = []
    system_b_results = []

    for symbol in symbols:
        df_1h = fetch_klines(symbol, interval="1h", limit=200)
        df_1d = fetch_klines(symbol, interval="1d", limit=100)

        if df_1h is None or len(df_1h) < 100 or df_1d is None:
            continue

        # ==========================================
        # 🟢 系統 A：1H 底底高 + 收盤高於前一根最高點
        # ==========================================
        try:
            lows = df_1h["Low"].values
            highs = df_1h["High"].values
            closes = df_1h["Close"].values

            # 用純粹的滾動視窗尋找近期的兩個波段低點（底底高判斷）
            # 尋找局部低點（左右各 3 根 K 棒最低）
            local_mins = []
            for i in range(3, len(lows) - 3):
                if lows[i] == min(lows[i - 3 : i + 4]):
                    local_mins.append((i, lows[i]))

            if len(local_mins) >= 2:
                prev_low_idx, prev_low_val = local_mins[-2]
                last_low_idx, last_low_val = local_mins[-1]

                # 條件 1：1H 必須呈現「底底高」（第二個底 > 第一個底）
                is_higher_low = last_low_val > prev_low_val

                # 條件 2：最近一根 1H 收盤價，高於前一根 1H 的最高點
                current_close_1h = closes[-1]
                prev_high_1h = highs[-2]
                is_breakout = current_close_1h > prev_high_1h

                # 確保最近這個底發生在相對近期（例如 36 小時內）
                recent_low_fresh = (len(df_1h) - last_low_idx) <= 36

                if is_higher_low and is_breakout and recent_low_fresh:
                    prev_close_1d = df_1d["Close"].iloc[-2]
                    change_pct = round(
                        (
                            (current_close_1h - prev_close_1d)
                            / prev_close_1d
                            * 100
                        ),
                        2,
                    )
                    exit_price = float(last_low_val)  # 防守價：以最近的底為防守

                    system_a_results.append(
                        {
                            "code": symbol.replace("USDT", ""),
                            "close": round(current_close_1h, 4),
                            "change_pct": change_pct,
                            "entry_price": round(current_close_1h, 4),
                            "exit_price": round(exit_price, 4),
                        }
                    )
        except Exception as e:
            pass

        # ==========================================
        # 🔵 系統 B：背離評分系統 (OBV +3, MACD +2,
