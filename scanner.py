from datetime import datetime
import json
import numpy as np
import pandas as pd
import requests


def scan_market():
    """CoinGecko 山寨幣掃描（改進版）- 客觀評分、無隨機"""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 80,          # 多抓一點，後面再過濾
        "page": 1,
        "sparkline": "true",
        "price_change_percentage": "24h,7d",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        res = requests.get(url, params=params, headers=headers, timeout=15)
        print(f"CoinGecko API 回應狀態碼: {res.status_code}")

        if res.status_code != 200:
            print(f"取得資料失敗: {res.text[:200]}")
            save_data([])
            return

        data = res.json()
        results = []

        for item in data:
            symbol = item["symbol"].upper()
            if symbol in ["BTC", "ETH", "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS", "USD1", "PYUSD"]:
                continue  # 排除主流與穩定幣

            current_price = float(item.get("current_price", 0) or 0)
            if current_price <= 0:
                continue

            change_24h = float(item.get("price_change_percentage_24h", 0) or 0)
            change_7d = float(item.get("price_change_percentage_7d_in_currency", 0) or 0)
            total_volume = float(item.get("total_volume", 0) or 0)
            market_cap = float(item.get("market_cap", 0) or 0)
            high_24h = float(item.get("high_24h", current_price * 1.01) or current_price)
            low_24h = float(item.get("low_24h", current_price * 0.99) or current_price)

            # 基本風險過濾
            if total_volume < 5_000_000:          # 成交量太小
                continue
            if change_24h > 18:                   # 已經噴太多，追高風險高
                continue
            if market_cap < 50_000_000:           # 市值太小
                continue

            sparkline = item.get("sparkline_in_7d", {}).get("price", [])
            if len(sparkline) < 12:
                continue

            df = pd.DataFrame({"Close": sparkline})
            # 用真實 24h 高低點來模擬近期波動（比單純 *1.01 好一點）
            df["High"] = df["Close"] * (1 + abs(change_24h) / 200)
            df["Low"] = df["Close"] * (1 - abs(change_24h) / 200)
            df["Volume"] = total_volume / len(sparkline)

            prices = df["Close"].values

            # ========== 技術指標 ==========
            # RSI (較短週期)
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(7, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(7, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = (100 - (100 / (1 + rs))).fillna(50).values

            # MACD Histogram
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

            # ========== 評分系統（完全客觀）==========
            score = 0
            triggered = []

            # 1. RSI 背離 / 超賣反彈
            if len(prices) >= 6:
                # 價格創新低但 RSI 沒有 →  latent 背離
                if prices[-1] < prices[-4] and rsi[-1] > rsi[-4] + 3:
                    score += 3
                    triggered.append("RSI latent背離")
                # 單純超賣
                if rsi[-1] < 32:
                    score += 2
                    triggered.append("RSI超賣")
                elif 32 <= rsi[-1] <= 45:
                    score += 1

            # 2. MACD 動能轉強
            if len(macd_hist) >= 4:
                if macd_hist[-1] > macd_hist[-2] > macd_hist[-3]:
                    score += 2
                    triggered.append("MACD動能向上")
                if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
                    score += 2
                    triggered.append("MACD翻正")

            # 3. KD 金叉 + 低檔
            if k_vals[-1] > d_vals[-1] and k_vals[-2] <= d_vals[-2]:
                score += 2
                triggered.append("KD金叉")
            if k_vals[-1] < 35 and d_vals[-1] < 40:
                score += 1
                triggered.append("KD低檔")

            # 4. 盤整壓縮（波動收斂）
            span_ratio = (high_24h - low_24h) / current_price
            if span_ratio < 0.045 and change_24h > -3:
                score += 3
                triggered.append("盤整壓縮")

            # 5. 量能相對強度（用市值比粗略判斷）
            vol_mcap_ratio = total_volume / market_cap if market_cap > 0 else 0
            if vol_mcap_ratio > 0.15:
                score += 2
                triggered.append("量能活躍")
            elif vol_mcap_ratio > 0.08:
                score += 1

            # 6. 7日表現（不要已經大漲太多）
            if -8 < change_7d < 12:
                score += 1
            if change_7d < -12:          # 跌深有機會
                score += 1
                triggered.append("7日跌深")

            # 7. 懲罰：已經短期過熱
            if change_24h > 9:
                score -= 2
            if rsi[-1] > 72:
                score -= 2
                triggered.append("RSI過熱")

            # 最低門檻
            if score < 4:
                continue

            # 狀態判斷
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

        # 依分數由高到低排序，取前 15
        results = sorted(results, key=lambda x: (x["score"], x["volume_24h"]), reverse=True)
        top_results = results[:15]

        print(f"\n處理完成，成功選出 {len(top_results)} 檔標的：\n")
        for i, r in enumerate(top_results, 1):
            print(f"{i:2d}. {r['code']:<8} 分數:{r['score']:2d}  {r['status']:<6}  "
                  f"24h:{r['change_24h']:+6.2f}%  7d:{r['change_7d']:+6.2f}%  "
                  f"{' | '.join(r['indicators'])}")

        save_data(top_results)

    except Exception as e:
        print(f"發生未預期錯誤: {e}")
        import traceback
        traceback.print_exc()
        save_data([])


def save_data(results):
    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_count": len(results),
        "stocks": results,
    }
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"\n資料寫入完成！updated_at: {output['updated_at']}，共 {len(results)} 檔標的。")


if __name__ == "__main__":
    scan_market()
