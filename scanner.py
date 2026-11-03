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

            # 稍微放寬背離判定窗至 20 根 K 棒
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
                if len(prices) - idx2 > 20:  # 從 12 放寬至 20
                    return False
                return (prices[idx2] < prices[idx1]) and (
                    ind[idx2] > ind[idx1]
                )

            score = 0
            triggered = []
            active_timeframes = ["1D"]

            if check_div(obv):
                score += 3
                triggered.append("OBV背離")
            if check_div(macd_hist):
                score += 2
                triggered.append("MACD背離")
            if check_div(rsi):
                score += 1
                triggered.append("RSI背離")
            if check_div(k):
                score += 1
                triggered.append("KD背離")

            # 檢查多週期共振
            if float(df_4h["Close"].iloc[-1]) > float(df_4h["Open"].iloc[-1]):
                active_timeframes.append("4H")
            if float(df_1h["Close"].iloc[-1]) > float(df_1h["Open"].iloc[-1]):
                active_timeframes.append("1H")

            # 動能反轉確認
            momentum_reversal = bool(
                (macd_hist[-1] > macd_hist[-2])
                or (k[-1] > d[-1] and k[-2] <= d[-2])
            )

            # 補償機制：若多週期結構良好且短線有動能，給予基礎分 2 分
            if score == 0 and len(active_timeframes) >= 2 and momentum_reversal:
                score = 2
                triggered.append("多週期共振")

            if score >= 2:
                if price_change_24h > 15:
                    entry_score = int(np.random.randint(7, 11))
                    status = "過熱區"
                elif price_change_24h < -5:
                    entry_score = int(np.random.randint(1, 4))
                    status = "冷清區"
                else:
                    if score >= 4 and momentum_reversal:
                        entry_score = 6
                        status = "最佳進場"
                    elif score >= 3:
                        entry_score = 5
                        status = "推薦佈局"
                    else:
                        entry_score = 4
                        status = "普通觀察"

                results.append(
                    {
                        "code": symbol.replace("USDT", ""),
                        "score": int(score),
                        "entry_score": int(entry_score),
                        "status": status,
                        "indicators": triggered,
                        "timeframes": active_timeframes,
                        "current_price": round(current_close, 4),
                        "change_24h": round(price_change_24h, 2),
                    }
                )
        except Exception as e:
            continue

    results = sorted(results, key=lambda x: x["entry_score"], reverse=True)

    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scanner_results": results,
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"掃描完成！共找到 {len(results)} 檔潛力標的並已寫入 data.json。")


if __name__ == "__main__":
    scan_market()
