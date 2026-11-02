import json
from datetime import datetime
import numpy as np
import pandas as pd
import requests
from scipy.signal import argrelextrema


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


def check_divergence_at_point(prices, indicator, order=5):
    """檢測底背離 (Price Lower Low + Indicator Higher Low)"""
    if len(prices) < 50:
        return False
    price_mins = argrelextrema(prices, np.less, order=order)[0]
    if len(price_mins) < 2:
        return False
    idx1 = price_mins[-2]
    idx2 = price_mins[-1]
    if len(prices) - idx2 > 15:
        return False
    return (prices[idx2] < prices[idx1]) and (indicator[idx2] > indicator[idx1])


def scan_market():
    symbols = get_binance_altcoins()
    print(f"總共取得 {len(symbols)} 檔山寨幣，開始掃描...")

    system_a_results = []
    system_b_results = []

    for symbol in symbols:
        # 抓取 1H 與 1D 資料
        df_1h = fetch_klines(symbol, interval="1h", limit=200)
        df_1d = fetch_klines(symbol, interval="1d", limit=100)

        if df_1h is None or len(df_1h) < 100 or df_1d is None:
            continue

        # ==========================================
        # 🟢 系統 A：1H 底底高 + 收盤高於前一根最高點
        # ==========================================
        try:
            prices_1h = df_1h["Close"].values
            lows_1h = df_1h["Low"].values

            # 尋找 1H 的波段低點 (order=5)
            min_indices = argrelextrema(lows_1h, np.less, order=5)[0]

            if len(min_indices) >= 2:
                # 取得最近兩個波段低點的值
                val_prev_low = lows_1h[min_indices[-2]]
                val_last_low = lows_1h[min_indices[-1]]

                # 條件 1：1H 必須呈現「底底高」（後一個低點 > 前一個低點）
                is_higher_low = val_last_low > val_prev_low

                # 條件 2：最近一根 1H 收盤價，高於前一根 1H 的最高點
                current_close_1h = df_1h["Close"].iloc[-1]
                prev_high_1h = df_1h["High"].iloc[-2]
                is_breakout = current_close_1h > prev_high_1h

                # 確保最後一個低點發生在相對近期，避免抓到過期的形態
                recent_low_fresh = (len(df_1h) - min_indices[-1]) <= 24

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
                    exit_price = float(
                        val_last_low
                    )  # 防守價：以最近的那個底（低點）作為防守

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
        # 🔵 系統 B：背離評分系統 (OBV +3, MACD +2, RSI/KD +1)
        # ==========================================
        try:
            prices = df_1d["Close"].values
            current_close = df_1d["Close"].iloc[-1]

            delta = df_1d["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = (100 - (100 / (1 + (gain / loss)))).values

            exp1 = df_1d["Close"].ewm(span=12, adjust=False).mean()
            exp2 = df_1d["Close"].ewm(span=26, adjust=False).mean()
            macd_hist = ((exp1 - exp2) - (exp1 - exp2).ewm(span=9).mean()).values

            obv = (
                (np.sign(df_1d["Close"].diff()) * df_1d["Volume"])
                .fillna(0)
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

            score = 0
            triggered = []

            if check_divergence_at_point(prices, obv, order=5):
                score += 3
                triggered.append("OBV")
            if check_divergence_at_point(prices, macd_hist, order=5):
                score += 2
                triggered.append("MACD")
            if check_divergence_at_point(prices, rsi, order=5):
                score += 1
                triggered.append("RSI")
            if check_divergence_at_point(prices, k, order=5):
                score += 1
                triggered.append("KD")

            momentum_reversal = (macd_hist[-1] > macd_hist[-2]) or (
                k[-1] > d[-1] and k[-2] <= d[-2]
            )

            if score >= 4 and momentum_reversal:
                recent_mins = argrelextrema(prices, np.less, order=5)[0]
                exit_price = (
                    prices[recent_mins[-1]] * 0.98
                    if len(recent_mins) > 0
                    else current_close * 0.95
                )

                system_b_results.append(
                    {
                        "code": symbol.replace("USDT", ""),
                        "score": score,
                        "indicators": triggered,
                        "entry_price": round(current_close, 4),
                        "exit_price": round(exit_price, 4),
                    }
                )
        except Exception as e:
            pass

    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_a": system_a_results,
        "system_b": system_b_results,
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(
        f"掃描完成！系統 A 找到 {len(system_a_results)} 檔，系統 B 找到 {len(system_b_results)} 檔。"
    )


if __name__ == "__main__":
    scan_market()
