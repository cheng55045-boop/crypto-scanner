import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import json

# 設定台灣時區 (UTC+8)
TZ_TAIPEI = timezone(timedelta(hours=8))

def fetch_binance_symbols():
    """取得幣安合約所有 USDT 交易對，排除穩定幣"""
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        symbols = []
        exclude_coins = {'USDCUSDT', 'FDUSDUSDT', 'TUSDUSDT', 'USDTUSDT'}
        
        for s in data['symbols']:
            symbol = s['symbol']
            if symbol.endswith('USDT') and symbol not in exclude_coins and s['status'] == 'TRADING':
                symbols.append(symbol)
        return symbols
    except Exception as e:
        print(f"取得交易對清單失敗: {e}")
        return []

def fetch_klines(symbol, interval, limit=50):
    """抓取指定幣種與時框的 K 棒數據"""
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            raw_data = response.json()
            df = pd.DataFrame(raw_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df
    except Exception:
        pass
    return None

def analyze_altcoins():
    symbols = fetch_binance_symbols()
    print(f"總共抓取到 {len(symbols)} 個 USDT 交易對，開始進行多時框策略掃描...")
    
    selected_stocks = []
    
    for symbol in symbols:
        df_1d = fetch_klines(symbol, '1d', limit=30)
        df_1h = fetch_klines(symbol, '1h', limit=10)
        
        if df_1d is None or len(df_1d) < 10 or df_1h is None or len(df_1h) < 5:
            continue
            
        # 過濾 24H 換手量過小的冷門幣
        total_volume_usdt = (df_1h['close'] * df_1h['volume']).sum()
        if total_volume_usdt < 500000:
            continue

        # 策略邏輯：
        # 1D 日線最新收盤價高於前一根日線最高點
        cond_1d = df_1d['close'].iloc[-1] > df_1d['high'].iloc[-2]
        if not cond_1d:
            continue

        # 1H 最近 3 根 K 棒底底高 (Higher Lows) 且最新收盤扎實突破前高
        h1_lows = df_1h['low'].iloc[-3:]
        h1_closes = df_1h['close'].iloc[-3:]
        
        is_higher_lows = (h1_lows.iloc[1] > h1_lows.iloc[0]) and (h1_lows.iloc[2] > h1_lows.iloc[1])
        cond_1h_breakout = h1_closes.iloc[-1] > df_1h['high'].iloc[-2]

        if is_higher_lows and cond_1h_breakout:
            prev_close = df_1h['close'].iloc[-2]
            curr_close = df_1h['close'].iloc[-1]
            change_pct = ((curr_close - prev_close) / prev_close) * 100
            
            entry_price = curr_close
            exit_price = float(h1_lows.min()) # 3根 1H 最低點作為防守價
            
            coin_info = {
                "code": symbol.replace('USDT', ''),
                "name": f"{symbol} Perpetual",
                "close": round(curr_close, 4),
                "change_pct": round(change_pct, 2),
                "volume": round(total_volume_usdt, 0),
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "has_warning": False
            }
            selected_stocks.append(coin_info)
            print(f"🎯 捕捉到符合標的: {symbol} | 進場價: {entry_price}")

    output_data = {
        "updated_at": datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S"),
        "total_count": len(selected_stocks),
        "stocks": selected_stocks
    }
    
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"掃描完成！共找到 {len(selected_stocks)} 檔符合條件的山寨幣，已更新至 data.json")

if __name__ == "__main__":
    analyze_altcoins()