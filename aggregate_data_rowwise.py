import pandas as pd
import numpy as np

def parse_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(';')
            if len(parts) >= 4:
                datetime_parts = parts[0].split(' ')
                if len(datetime_parts) == 3:
                    timestamp = ' '.join(datetime_parts)
                    data.append([timestamp] + parts[1:])
    df = pd.DataFrame(data, columns=['timestamp', 'price1', 'price2', 'price3', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d %H%M%S %f', errors='coerce')
    df[['price1', 'price2', 'price3']] = df[['price1', 'price2', 'price3']].apply(pd.to_numeric)
    df['volume'] = df['volume'].astype(int)
    return df

def aggregate_ticks(data, ticks_per_bar=100):
    tick_counter = 0
    buy_volume = 0
    sell_volume = 0
    high_price = -float('inf')
    low_price = float('inf')
    volumes = []
    prices = []
    bars = []
    for index, row in data.iterrows():
        tick_counter += 1
        if row['price1'] == row['price2']:
            buy_volume += row['volume']
        elif row['price1'] == row['price3']:
            sell_volume += row['volume']
        high_price = max(high_price, row['price1'])
        low_price = min(low_price, row['price1'])
        prices.append(row['price1'])
        volumes.append(row['volume'])
        if tick_counter >= ticks_per_bar:
            close_price = row['price1']
            bar = {
                'timestamp': row['timestamp'],
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'prices': prices,
                'volumes': volumes
            }
            bars.append(bar)
            tick_counter = 0
            buy_volume = 0
            sell_volume = 0
            high_price = -float('inf')
            low_price = float('inf')
            prices = []
            volumes = []
    return bars

def calculate_market_profile(bars, value_area_volume_percentage=0.7):
    results = []
    for bar in bars:
        prices = bar['prices']
        volumes = bar['volumes']
        volume_profile = pd.Series(volumes, index=prices).groupby(level=0).sum()
        poc_price = volume_profile.idxmax()
        total_volume = volume_profile.sum()
        value_area_volume = total_volume * value_area_volume_percentage
        volume_cumsum = volume_profile.sort_values(ascending=False).cumsum()
        value_area_prices = volume_profile[volume_cumsum <= value_area_volume]
        if value_area_prices.empty:
            vah = val = np.nan
        else:
            vah = value_area_prices.idxmax()
            val = value_area_prices.idxmin()
        last_price = bar['close']
        dist_poc = last_price - poc_price if poc_price is not None else np.nan
        dist_vah = last_price - vah if vah is not None else np.nan
        dist_val = last_price - val if val is not None else np.nan
        results.append({
            'timestamp': bar['timestamp'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'buy_volume': bar['buy_volume'],
            'sell_volume': bar['sell_volume'],
            'poc': poc_price,
            'vah': vah,
            'val': val,
            'dist_poc': dist_poc,
            'dist_vah': dist_vah,
            'dist_val': dist_val
        })
    return pd.DataFrame(results)

def main(file_path):
    data = parse_file(file_path)
    tick_bars = aggregate_ticks(data, ticks_per_bar=1000)
    market_profile_data = calculate_market_profile(tick_bars)
    print(market_profile_data.head())

# Specify the path to your data file
file_path = 'ES DEC23.Last.csv'
main(file_path)
