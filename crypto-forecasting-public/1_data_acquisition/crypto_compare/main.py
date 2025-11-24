''' Script to download Bitcoin (BTC) historical data from the cryptocompare API. '''

import pandas as pd
from functions import auth_key, get_data, get_balance_data

if __name__=='__main__':

    #--------------------------------------------------------------------------
    # Fetch Bitcoin (BTC) data
    #--------------------------------------------------------------------------

    # get (todays) price data
    feature = 'histoday'
    params = {'fsym': 'BTC',
              'tsym': 'EUR'}
    drop_columns = ['btc_price_volumefrom',
                    'btc_price_volumeto',
                    'btc_price_conversionType',
                    'btc_price_conversionSymbol']
    prefix = 'btc_price_'

    btc_price = get_data(feature, params, 'btc', prefix).drop(columns=drop_columns)
    
    #--------------------------------------------------------------------------

    # get (todays) aggregated exchange volume for various currencies
    feature = 'histoday'
    currencies = ['EUR', 'USD']
    btc_currency_vol = pd.DataFrame()

    for i in currencies:
        params = {'fsym': 'BTC',
                  'tsym': i}
        prefix = 'btc_' + i + '_'
        drop_columns = [prefix + 'high',
                        prefix + 'low',
                        prefix + 'open',
                        prefix + 'close',
                        prefix + 'conversionType',
                        prefix + 'conversionSymbol']
        data = get_data(feature, params, 'btc', prefix).drop(columns=drop_columns)
        btc_currency_vol = pd.concat([data, btc_currency_vol], axis=1)

    #--------------------------------------------------------------------------

    # get (todays) disaggregated exchange volume
    exchanges = ('Binance', 'BTSE', 'Bitci', 'Coinbase', 'Kraken', 'Bitfinex')
    btc_exchange_vol = pd.DataFrame()

    for i in exchanges:
        feature = 'exchange/symbol/histoday'
        params = {'fsym': 'BTC',
                'tsym': 'EUR',
                'e': i}
        prefix = 'btc_exchange_' + i + '_'
        data = get_data(feature, params, 'btc', prefix)
        btc_exchange_vol = pd.concat([data, btc_exchange_vol], axis=1)
        
    #--------------------------------------------------------------------------

    # get (yesterdays) blockchain data (requires paid API key)
    btc_blockchain = None
    try:
        feature = 'blockchain/histo/day'
        params = {'fsym': 'BTC',
                  'auth_key': auth_key}
        drop_columns = ['btc_id', 'btc_symbol']

        btc_blockchain = (get_data(feature, params, 'btc', 'btc_', itype=1)
                          .drop(columns=drop_columns))
        btc_blockchain.loc[max(btc_blockchain.index)+86400, :] = None
        btc_blockchain = btc_blockchain.shift(1)
        print("✓ Blockchain data fetched successfully")
    except Exception as e:
        print(f"⚠ Blockchain data skipped (requires paid API key): {e}")

    # --------------------------------------------------------------------------

    # get (yesterdays) balance data (requires paid API key)
    balances = None
    try:
        feature = 'blockchain/balancedistribution/histo/day'
        params = {'fsym': 'BTC',
                  'auth_key': auth_key}
        prefix = 'btc_balance_distribution_'

        balances = get_balance_data(feature, params, prefix)
        balances.loc[max(balances.index)+86400, :] = None
        balances = balances.shift(1)
        print("✓ Balance distribution data fetched successfully")
    except Exception as e:
        print(f"⚠ Balance distribution data skipped (requires paid API key): {e}")

    #------------------------------------------------------------------------------

    # concatenate all dataframes into one and save
    dataframes = [btc_price,
                  btc_currency_vol,
                  btc_exchange_vol]
    
    # Add optional data if available
    if btc_blockchain is not None:
        dataframes.append(btc_blockchain)
    if balances is not None:
        dataframes.append(balances)

    btc_data = pd.concat(dataframes, axis=1).sort_index()
    btc_data.to_parquet('btc_data.parquet.gzip', compression='gzip')

    print("=" * 60)
    print("Bitcoin data acquisition complete!")
    print(f"Saved {len(btc_data)} data points to btc_data.parquet.gzip")
    print("=" * 60)

    #--------------------------------------------------------------------------
    # Fetch crypto index data (requires paid API key)
    #--------------------------------------------------------------------------

    # get daily crypto indices
    try:
        index_names = ('MVDA', 'MVDALC', 'MVDAMC', 'MVDASC')
        indices = pd.DataFrame()

        for i in index_names:
            feature = 'index/histo/day'
            params = {'indexName': i,
                      'auth_key': auth_key}
            prefix = 'index_' + i + '_'
            indices = pd.concat(
                [get_data(feature, params, 'btc', prefix), indices], axis=1)

        indices.sort_index().to_parquet('indices_data.parquet.gzip', compression='gzip')
        print("✓ Crypto indices data fetched successfully")
    except Exception as e:
        print(f"⚠ Crypto indices data skipped (requires paid API key): {e}")

    # get hourly BTC volatility data
    try:
        feature = 'index/histo/hour'
        params = {'indexName': 'BVIN',
                  'auth_key': auth_key}
        prefix = 'btc_volatility_index_'

        btc_volatility = get_data(feature, params, 'btc', prefix)
        btc_volatility.to_parquet(
            'btc_volatility_hourly.parquet.gzip', compression='gzip')
        print("✓ BTC volatility index data fetched successfully")
    except Exception as e:
        print(f"⚠ BTC volatility index data skipped (requires paid API key): {e}")
