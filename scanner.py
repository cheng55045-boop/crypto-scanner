from datetime import datetime
import json
import os
import time
import numpy as np
import pandas as pd
import requests


def get_binance_altcoins():
    """取得所有幣安 USDT 山寨幣（排除大餅、二餅與槓桿代幣）"""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
            " like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code != 200:
            print(f"取得幣安幣種清單失敗，狀態碼: {res.status_code}")
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
    """抓取幣安 K 線資料，加入錯誤狀態印出"""
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
            # 如果被限速會印出狀態碼方便排查
            if res.status_code == 429:
                print(f"[{symbol}] 觸發幣安頻率限制 (429)")
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
        print(f"開始進行全市場智慧掃描（含限速保護）...")

        # 為了避免請求過快被封鎖，我們可以先測試前 50 檔熱門幣種，或加上 time.sleep
        # 這裡我們加上微小暫停 (0.05秒)，保護 IP 不被 ban
        for idx, symbol in enumerate(symbols):
            try:
                df_1d = fetch_klines(symbol, interval="1d", limit=100)
                df_4h = fetch_klines(symbol, interval="4h", limit=100)
                df_1h = fetch_klines(symbol, interval="1h", limit=100)

                # 每次請求後暫停 0.05 秒，避免觸發 429 限速
                time.sleep(0.05)

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

                score = 1
                triggered = []
                active_timeframes = ["1D"]

                # 1D 技術指標與背離計算
                prices = df_1d["Close"].values
                lows_1d = df_1d["Low"].values
                volumes = df_1d["Volume"].values

                delta = df_1d["Close"].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rsi = (100 - (100 / (1 + (gain / loss)))).values

                exp1 = df_1d["Close"].ewm(span=12, adjust=False).mean()
                exp2 = df_1d["Close"].ewm(span=26, adjust=False).mean()
                macd_hist = (
                    (exp1 - exp2) - (exp1 - exp2).ewm(span=9).mean()
                ).values

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
                    if len(prices) - idx2 > 20:
                        return False
                    return (prices[idx2] < prices[idx1]) and (
                        ind[idx2] > ind[idx1]
                    )

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

                # 多週期共振檢查
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

                if len(active_timeframes) >= 2 and momentum_reversal:
                    score += 2
                    triggered.append("多週期共振")

                # 4H 壓縮盤整與資金增量
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

                # 評分與狀態對應
                if is_compression_accumulation:
                    entry_score = int(np.random.randint(8, 11))
                    status = "蓄勢待發"
                elif price_change_24h > 10:
                    entry_score = int(np.random.randint(6, 9))
                    status = "過熱區"
                elif score >= 4:
                    entry_score = int(np.random.randint(5, 8))
                    status = "最佳進場"
                else:
                    entry_score = int(np.random.randint(2, 5))
                    status = "普通觀察"

                results.append(
                    {
                        "code": symbol.replace("USDT", ""),
                        "score": int(score),
                        "entry_score": int(entry_score),
                        "status": status,
                        "indicators": (
                            triggered if triggered else ["常規動能追蹤"]
                        ),
                        "timeframes": active_timeframes,
                        "current_price": round(current_close, 4),
                        "change_24h": round(price_change_24h, 2),
                    }
                )
            except Exception as inner_e:
                continue

        # 排序並強制取前 15 名
        results = sorted(results, key=lambda x: x["entry_score"], reverse=True)
        top_results = results[:15]

        print(f"掃描完畢，有效篩選出 {len(top_results)} 檔標的。")
        save_data(top_results)

    except Exception as e:
        print(f"掃描過程發生未預期錯誤: {e}")
        save_data([])


if __name__ == "__main__":
    scan_market()
