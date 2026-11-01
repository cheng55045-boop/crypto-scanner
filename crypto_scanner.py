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
    print(f"總共抓取到 {len(symbols)} 個 USDT 交易對，開始進行波段底底高與突破策略掃描...")
    
    selected_stocks = []
    
    for symbol in symbols:
        df_1d = fetch_klines(symbol, '1d', limit=30)
        df_1h = fetch_klines(symbol, '1h', limit=30) # 拉長 1H 樣本以捕捉波段結構
        
        if df_1d is None or len(df_1d) < 10 or df_1h is None or len(df_1h) < 15:
            continue
            
        # 過濾 24H 換手量過小的冷門幣
        total_volume_usdt = (df_1h['close'] * df_1h['volume']).sum()
        if total_volume_usdt < 500000:
            continue

        # 1. 大週期方向：1D 日線最新收盤價高於前一根日線最高點
        cond_1d = df_1d['close'].iloc[-1] > df_1d['high'].iloc[-2]
        if not cond_1d:
            continue

        # 2. 1H 波段結構分析：尋找上漲、回檔未破前低、再次上漲突破的結構
        # 我們取最近幾根 1H K棒來尋找局部低點與高點
        highs = df_1h['high'].values
        lows = df_1h['low'].values
        closes = df_1h['close'].values

        # 尋找波段特徵：
        # 檢查是否有形成「底底高」：比對近期的波段低點是否逐步墊高
        # 簡單且穩健的寫法：把最近區間分割為前半段與後半段，確認回檔低點高於前期低點
        recent_lows = lows[-15:]
        # 尋找前半段最低點與後半段最低點
        mid = len(recent_lows) // 2
        early_low = recent_lows[:mid].min()
        late_low = recent_lows[mid:-1].min()
        
        # 條件 A：回檔低點沒有過低（底底高結構）
        is_higher_low = late_low > early_low

        # 條件 B：最新一根 1H K棒的收盤價，必須高於前一根 K 棒的最高點（突破確認）
        is_breakout_close = closes[-1] > highs[-2]

        if is_higher_low and is_breakout_close:
            prev_close = closes[-2]
            curr_close = closes[-1]
            change_pct = ((curr_close - prev_close) / prev_close) * 100
            
            # 進場價：突破當根 1H K 棒的收盤價
            entry_price = curr_close
            
            # 出場/停損防守價：收盤高過前一根最高點的那根突破 K 棒的最低點
            exit_price = float(lows[-1])
            
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
            print(f"🎯 捕捉到波段突破標的: {symbol} | 進場價: {entry_price} | 防守價: {exit_price}")

    output_data = {
        "updated_at": datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S"),
        "total_count": len(selected_stocks),
        "stocks": selected_stocks
    }
    
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"掃描完成！共找到 {len(selected_stocks)} 檔符合波段突破條件的山寨幣，已更新至 data.json")

if __name__ == "__main__":
    analyze_altcoins()
