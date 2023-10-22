import pandas as pd
import numpy as np

def parse_file(file_path):
    # Read the file directly into a DataFrame
    data = pd.read_csv(
        file_path, 
        delimiter=';', 
        header=None, 
        names=['datetime', 'price1', 'price2', 'price3', 'volume'],
        dtype={'datetime': str, 'price1': float, 'price2': float, 'price3': float, 'volume': int}
    )
    
    # Convert the 'datetime' column to a pandas datetime object
    data['datetime'] = pd.to_datetime(data['datetime'], format='%Y%m%d %H%M%S %f', errors='coerce')
    
    return data

def aggregate_ticks_vectorized(data, ticks_per_bar=100):
    # Calculate buy/sell volumes
    data['buy_volume'] = np.where(data['price1'] == data['price2'], data['volume'], 0)
    data['sell_volume'] = np.where(data['price1'] == data['price3'], data['volume'], 0)
    
    # Create a grouping key for aggregating every 'ticks_per_bar' records
    data['group_key'] = data.index // ticks_per_bar
    
    # Define the aggregation dictionary
    aggregation_funcs = {
        'datetime': 'last',
        'price1': ['max', 'min', 'last'],
        'buy_volume': 'sum',
        'sell_volume': 'sum',
        'volume': 'sum'  # we need the total volume for each bar for market profile calculations
    }
    
    # Group by 'group_key' and aggregate
    aggregated = data.groupby('group_key').agg(aggregation_funcs)
    aggregated.columns = ['timestamp', 'high', 'low', 'close', 'buy_volume', 'sell_volume', 'total_volume']
    
    return aggregated

def calculate_market_profile(bars_df, value_area_volume_percentage=0.7):
    results = []
    for _, bar in bars_df.iterrows():
        # For the market profile, we need individual transactions, which we don't have after aggregation.
        # As an approximation, we distribute the total volume evenly across the price range (high to low) of the bar.
        prices = np.linspace(bar['low'], bar['high'], num=int(bar['total_volume']))
        volumes = np.full_like(prices, 1)  # since we're distributing evenly, each "transaction" is 1 volume unit

        # Calculate volume profile
        volume_profile = pd.Series(volumes, index=prices).groupby(level=0).sum()
        poc_price = volume_profile.idxmax()
        total_volume = volume_profile.sum()
        value_area_volume = total_volume * value_area_volume_percentage
        volume_cumsum = volume_profile.sort_values(ascending=False).cumsum()
        value_area_prices = volume_profile[volume_cumsum <= value_area_volume]

        # Check if value_area_prices is empty
        if value_area_prices.empty:
            vah = val = np.nan  # or some other indicator of missing data
        else:
            vah = value_area_prices.idxmax()
            val = value_area_prices.idxmin()

        # Calculate distances
        last_price = bar['close']
        dist_poc = last_price - poc_price if poc_price is not None else np.nan
        dist_vah = last_price - vah if vah is not None else np.nan
        dist_val = last_price - val if val is not None else np.nan

        # Append results
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
    tick_bars = aggregate_ticks_vectorized(data, ticks_per_bar=1000)
    market_profile_data = calculate_market_profile(tick_bars)
    print(market_profile_data.head())  # Display the first few rows

# Specify the path to your data file
file_path = r'ES DEC23.Last.csv'
main(file_path)
