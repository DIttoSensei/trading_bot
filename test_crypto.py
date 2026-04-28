import requests
import pandas as pd

def get_binance_data(symbol='BTCUSDT', limit=100):
    """Get crypto data directly from Binance"""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': '1m',
        'limit': limit
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if isinstance(data, dict) and 'code' in data:
        print(f"Error: {data}")
        return None
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# Test it
print("Testing Binance API...")
df = get_binance_data('BTCUSDT', 50)

if df is not None and len(df) > 0:
    print(f"✅ SUCCESS! Got {len(df)} minutes of data")
    print(f"Latest Bitcoin price: ${df['close'].iloc[-1]:,.2f}")
    print("\nLast 3 minutes:")
    print(df.tail(3))
else:
    print("❌ Failed to get data")