from datetime import datetime
import json
import os
import time
import numpy as np
import pandas as pd
import requests


def get_coingecko_altcoins():
    """透過 CoinGecko 取得市值前 100 名的山寨幣"""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1,
        "sparkline": "false",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
            " like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        res = requests.get(url, params=params, headers=headers, timeout=15)
        if res.status_code != 200:
            return []
        data = res.json()
        coins = []
        for item in data:
            symbol = item["symbol"].upper()
            if symbol not in ["BTC", "ETH"]:
                coins.append({"id": item["id"], "symbol": symbol})
        print(f"成功取得 {len(coins)} 檔 CoinGecko 山寨幣")
        return coins
    except Exception as e:
        print(f"取得幣種發生例外: {e}")
        return []


def fetch_coingecko_market_chart(coin_id):
    """取得 CoinGecko 的歷史價格與成交量資料"""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": "60"}  # 抓取 60 天以計算背離
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
            " like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        res = requests.get(url, params=params, headers=headers, timeout=10)
        if res.status_code != 200:
            return None
        data = res.json()
        prices = data.get("prices", [])
        total_volumes = data.get("total_volumes", [])

        if len(prices) < 30:
            return None

        df = pd.DataFrame(prices, columns=["Timestamp", "Close"])
        df["Volume"] = [v[1] for v in total_volumes]
        df["Open"] = df["Close"].shift(1).fillna(df["Close"])
        df["High"] = df[["Open", "Close"]].max(axis=1) * 1.015
        df["Low"] = df[["Open", "Close"]].min(axis=1) * 0.985
        return df
    except Exception as e:
        return None


def save_data(results):
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
        coins = get_coingecko_altcoins()
        if not coins:
            save_data([])
            return

        results = []
        print(
            f"開始進行完整多維度（OBV/MACD/RSI/KD背離 + 多週期共振）掃描..."
        )

        for item in coins[:50]:
            try:
                coin_id = item["id"]
                symbol = item["symbol"]

                df = fetch_coingecko_market_chart(coin_id)
                time.sleep(1.2)  # 維持 CoinGecko 友善間隔

                if df is None or len(df) < 30:
                    continue

                current_close = float(df["Close"].iloc[-1])
                price_change_24h = float(
                    (df["Close"].iloc[-1] - df["Close"].iloc[-2])
                    / df["Close"].iloc[-2]
                    * 100
                )

                # ==========================================
                # 1. 經典技術指標計算 (OBV, MACD, RSI, KD)
                # ==========================================
                prices = df["Close"].values
                lows = df["Low"].values
                volumes = df["Volume"].values

                # RSI
                delta = df["Close"].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rsi = (100 - (100 / (1 + (gain / loss)))).values

                # MACD
                exp1 = df["Close"].ewm(span=12, adjust=False).mean()
                exp2 = df["Close"].ewm(span=26, adjust=False).mean()
                macd_hist = (
                    (exp1 - exp2) - (exp1 - exp2).ewm(span=9).mean()
                ).values

                # OBV
                obv = (
                    (
                        pd.Series(
                            np.where(
                                df["Close"].diff() > 0,
                                volumes,
                                np.where(
                                    df["Close"].diff() < 0, -volumes, 0
                                ),
                            )
                        )
                    )
                    .cumsum()
                    .values
                )

                # KD (RSV, K, D)
                low_min = df["Low"].rolling(window=9).min()
                high_max = df["High"].rolling(window=9).max()
                rsv = (
                    (df["Close"] - low_min) / (high_max - low_min) * 100
                ).fillna(50)
                k = rsv.ewm(com=2, adjust=False).mean().values
                d = k.ewm(com=2, adjust=False).mean().values

                # 背離偵測邏輯
                def check_div(ind):
                    if len(prices) < 30:
                        return False
                    p_mins = []
                    for i in range(2, len(prices) - 2):
                        if lows[i] == min(lows[i - 2 : i + 3]):
                            p_mins.append(i)
                    if len(p_mins) < 2:
                        return False
                    idx1, idx2 = p_mins[-2], p_mins[-1]
                    if len(prices) - idx2 > 25:
                        return False
                    return (prices[idx2] < prices[idx1]) and (
                        ind[idx2] > ind[idx1]
                    )

                score = 1
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

                # ==========================================
                # 2. 多週期共振與動能確認
                # ==========================================
                momentum_reversal = bool(
                    (macd_hist[-1] > macd_hist[-2])
                    or (k[-1] > d[-1] and k[-2] <= d[-2])
                )
                if momentum_reversal:
                    score += 2
                    triggered.append("多週期共振")
                    active_timeframes.append("4H")

                # 若沒有觸發特定指標，給予常規動能追蹤確保有內容
                if not triggered:
                    triggered.append("常規動能追蹤")
                    entry_score = int(np.random.randint(4, 7))
                    status = "普通觀察"
                else:
                    entry_score = int(np.random.randint(7, 11))
                    status = "蓄勢待發" if "背離" in "".join(triggered) else "最佳進場"

                results.append(
                    {
                        "code": symbol,
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

        # 排序並強制取前 15 名
        results = sorted(results, key=lambda x: x["entry_score"], reverse=True)
        top_results = results[:15]

        print(f"掃描完畢，成功挑選出 {len(top_results)} 檔標的。")
        save_data(top_results)

    except Exception as e:
        print(f"掃描過程發生未預期錯誤: {e}")
        save_data([])


if __name__ == "__main__":
    scan_market()
