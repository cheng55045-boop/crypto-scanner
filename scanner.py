import json
from datetime import datetime
import numpy as np
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


def fetch_klines(symbol, interval="1d", limit=100):
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

    system_b_results = []

    for symbol in symbols:
        df_1d = fetch_klines(symbol, interval="1d", limit=100)

        if df_1d is None or len(df_1d) < 60:
            continue

        # ==========================================
        # 🔵 系統 B：背離評分系統 (OBV +3, MACD +2, RSI/KD +1)
        # ==========================================
        try:
            prices = df_1d["Close"].values
            current_close_1d = df_1d["Close"].iloc[-1]
            lows_1d = df_1d["Low"].values
            volumes = df_1d["Volume"].values

            # 計算指標
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

            if check_div(obv):
                score += 3
                triggered.append("OBV")
            if check_div(macd_hist):
                score += 2
                triggered.append("MACD")
            if check_div(rsi):
                score += 1
                triggered.append("RSI")
            if check_div(k):
                score += 1
                triggered.append("KD")

            momentum_reversal = (macd_hist[-1] > macd_hist[-2]) or (
                k[-1] > d[-1] and k[-2] <= d[-2]
            )

            if score >= 4 and momentum_reversal:
                exit_price = current_close_1d * 0.95
                system_b_results.append(
                    {
                        "code": symbol.replace("USDT", ""),
                        "score": score,
                        "indicators": triggered,
                        "entry_price": round(current_close_1d, 4),
                        "exit_price": round(exit_price, 4),
                    }
                )
        except Exception as e:
            pass

    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_b": system_b_results,
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"掃描完成！系統 B 找到 {len(system_b_results)} 檔。")


if __name__ == "__main__":
    scan_market()
