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
        print(
            f"開始進行完整多維度（背離 + 多週期共振 + 4H壓縮吸籌）掃描..."
        )

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

                current_close = float(df_1d["Close"].iloc[-1])
                price_change_24h = float(
                    (df_1d["Close"].iloc[-1] - df_1d["Close"].iloc[-2])
                    / df_1d["Close"].iloc[-2]
                    * 100
                )

                # ==========================================
                # 1. 完整技術指標計算 (1D)
                # ==========================================
                prices = df_1d["Close"].values
                lows_1d = df_1d["Low"].values
                volumes = df_1d["Volume"].values

                # RSI
                delta = df_1d["Close"].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rsi = (100 - (100 / (1 + (gain / loss)))).values

                # MACD
                exp1 = df_1d["Close"].ewm(span=12, adjust=False).mean()
                exp2 = df_1d["Close"].ewm(span=26, adjust=False).mean()
                macd_hist = (
                    (exp1 - exp2) - (exp1 - exp2).ewm(span=9).mean()
                ).values

                # OBV
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

                # KD (RSV, K, D)
                low_min = df_1d["Low"].rolling(window=9).min()
                high_max = df_1d["High"].rolling(window=9).max()
                rsv = (
                    (df_1d["Close"] - low_min) / (high_max - low_min) * 100
                ).fillna(50)
                k = rsv.ewm(com=2, adjust=False).mean().values
                d = k.ewm(com=2, adjust=False).mean().values

                # 背離偵測函式（20 根 K 棒窗）
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
                    if len(prices) - idx2 > 20:
                        return False
                    return (prices[idx2] < prices[idx1]) and (
                        ind[idx2] > ind[idx1]
                    )

                score = 0
                triggered = []
                active_timeframes = ["1D"]

                # 累計原本的背離分數
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

                # ==========================================
                # 2. 多週期共振檢查 (4H, 1H)
                # ==========================================
                if float(df_4h["Close"].iloc[-1]) > float(
                    df_4h["Open"].iloc[-1]
                ):
                    active_timeframes.append("4H")
                if float(df_1h["Close"].iloc[-1]) > float(
                    df_1h["Open"].iloc[-1]
                ):
                    active_timeframes.append("1H")

                momentum_reversal = bool(
                    (macd_hist[-1] > macd_hist[-2])
                    or (k[-1] > d[-1] and k[-2] <= d[-2])
                )

                if (
                    score == 0
                    and len(active_timeframes) >= 2
                    and momentum_reversal
                ):
                    score = 2
                    triggered.append("多週期共振")

                # ==========================================
                # 3. 新增：4H 壓縮盤整與資金增量（主力吸籌）偵測
                # ==========================================
                h_4h = df_4h["High"].iloc[-15:].max()
                l_4h = df_4h["Low"].iloc[-15:].min()
                c_4h = df_4h["Close"].iloc[-1]
                span_ratio = (h_4h - l_4h) / c_4h

                vol_recent = df_4h["Volume"].iloc[-5:].mean()
                vol_past = df_4h["Volume"].iloc[-25:-5].mean()
                vol_surge = vol_recent > (vol_past * 1.3)

                is_compression_accumulation = (span_ratio < 0.08) and vol_surge

                if is_compression_accumulation:
                    score += 3
                    triggered.append("4H壓縮吸籌")
                    if "4H" not in active_timeframes:
                        active_timeframes.append("4H")

                # ==========================================
                # 4. 評分與狀態分級
                # ==========================================
                if score >= 2:
                    if is_compression_accumulation:
                        entry_score = 6
                        status = "蓄勢待發"
                    elif price_change_24h > 15:
                        entry_score = int(np.random.randint(7, 10))
                        status = "過熱區"
                    elif score >= 4 and momentum_reversal:
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
            except Exception as inner_e:
                continue

        results = sorted(results, key=lambda x: x["entry_score"], reverse=True)
        save_data(results)

    except Exception as e:
        print(f"掃描過程發生未預期錯誤: {e}")
        save_data([])


if __name__ == "__main__":
    scan_market()
