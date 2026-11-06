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
            gain = delta.where(delta > 0, 0).rolling(7, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(7, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = (100 - (100 / (1 + rs))).fillna(50).values

            # MACD
            exp1 = df["Close"].ewm(span=6, adjust=False).mean()
            exp2 = df["Close"].ewm(span=13, adjust=False).mean()
            macd_line = exp1 - exp2
            signal = macd_line.ewm(span=5, adjust=False).mean()
            macd_hist = (macd_line - signal).values

            # KD
            low_min = df["Low"].rolling(5, min_periods=1).min()
            high_max = df["High"].rolling(5, min_periods=1).max()
            rsv = ((df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100).fillna(50)
            k = rsv.ewm(com=1, adjust=False).mean()
            d = k.ewm(com=1, adjust=False).mean()
            k_vals = k.values
            d_vals = d.values

            score = 0
            triggered = []

            if len(prices) >= 6:
                if prices[-1] < prices[-4] and rsi[-1] > rsi[-4] + 3:
                    score += 3
                    triggered.append("RSI latent背離")
                if rsi[-1] < 32:
                    score += 2
                    triggered.append("RSI超賣")
                elif 32 <= rsi[-1] <= 45:
                    score += 1

            if len(macd_hist) >= 4:
                if macd_hist[-1] > macd_hist[-2] > macd_hist[-3]:
                    score += 2
                    triggered.append("MACD動能向上")
                if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
                    score += 2
                    triggered.append("MACD翻正")

            if k_vals[-1] > d_vals[-1] and k_vals[-2] <= d_vals[-2]:
                score += 2
                triggered.append("KD金叉")
            if k_vals[-1] < 35 and d_vals[-1] < 40:
                score += 1
                triggered.append("KD低檔")

            span_ratio = (high_24h - low_24h) / current_price
            if span_ratio < 0.045 and change_24h > -3:
                score += 3
                triggered.append("盤整壓縮")

            vol_mcap_ratio = total_volume / market_cap if market_cap > 0 else 0
            if vol_mcap_ratio > 0.15:
                score += 2
                triggered.append("量能活躍")
            elif vol_mcap_ratio > 0.08:
                score += 1

            if -8 < change_7d < 12:
                score += 1
            if change_7d < -12:
                score += 1
                triggered.append("7日跌深")

            if change_24h > 9:
                score -= 2
            if rsi[-1] > 72:
                score -= 2
                triggered.append("RSI過熱")

            if score < 3:
                continue

            if score >= 9:
                status = "重點關注"
            elif score >= 6:
                status = "蓄勢待發"
            else:
                status = "觀察中"

            results.append({
                "code": symbol,
                "name": item.get("name", symbol),
                "score": int(score),
                "status": status,
                "indicators": triggered,
                "current_price": round(current_price, 6),
                "change_24h": round(change_24h, 2),
                "change_7d": round(change_7d, 2),
                "volume_24h": round(total_volume),
                "market_cap": round(market_cap),
            })

        results = sorted(results, key=lambda x: (x["score"], x["volume_24h"]), reverse=True)
        return results[:15]

    except Exception as e:
        print(f"掃描錯誤: {e}")
        return []


def save_data(results):
    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_count": len(results),
        "source": "CoinGecko (GitHub Actions)",
        "stocks": results,
    }
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"已更新 data.json，共 {len(results)} 檔")


if __name__ == "__main__":
    print("開始掃描...")
    results = scan_market()
    if results:
        save_data(results)
        print("前 5 名：")
        for i, r in enumerate(results[:5], 1):
            print(f"  {i}. {r['code']:<6} 分數:{r['score']}  {r['status']}  24h:{r['change_24h']:+.2f}%")
    else:
        print("本次無符合標的")
