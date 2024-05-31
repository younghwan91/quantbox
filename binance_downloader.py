import os
import csv
import pandas as pd
from binance.client import Client
from binance.enums import HistoricalKlinesType
from tqdm import tqdm

class BinanceDownloader:
    def __init__(self, config, client):
        # self.client = self.initialize_client(config)
        self.client = client
        self.saving_dir = config["SAVING_DIR"]
        self.trading_coins = self.get_trading_coins(self.client, config["QUOTE_COIN"], config["N_COINS"])
        self.config = config

    def initialize_client(self, config):
        return Client(api_key=config["API_KEY"], api_secret=config["API_SECRET"])

    def get_trading_coins(self, client, quote_coin, n_coins):
        futures_exchange_info = client.futures_exchange_info()
        trading_coins = [info['symbol'] for info in futures_exchange_info['symbols'] if info['symbol'].endswith(quote_coin)]

        assert n_coins <= len(trading_coins), "N_COINS is greater than the number of trading coins"
        return trading_coins[:n_coins]

    def save_trading_coins(self):
        with open(os.path.join(self.saving_dir, self.config["CSV_FILENAME"]), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.trading_coins)

    def fetch_and_save_klines(self, coin, interval, start_str, end_str):
        
        """
        Fetch and save klines for a given coin and interval.

        Interval options:
        - '1m': 1 minute
        - '3m': 3 minutes
        - '5m': 5 minutes
        - '15m': 15 minutes
        - '30m': 30 minutes
        - '1h': 1 hour
        - '2h': 2 hours
        - '4h': 4 hours
        - '6h': 6 hours
        - '8h': 8 hours
        - '12h': 12 hours
        - '1d': 1 day
        - '3d': 3 days
        - '1w': 1 week
        - '1M': 1 month
        """

        klines = self.client.get_historical_klines(
            symbol=coin,
            interval=interval,
            start_str=start_str,
            end_str=end_str,
            limit=1000,
            klines_type=HistoricalKlinesType.FUTURES
        )
        df = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 
                            'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 
                            'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
        df = df.astype(float)
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df.set_index('Open Time', inplace=True)
        df.to_pickle(os.path.join(self.saving_dir, f'{coin}.pkl'))

    def download_data(self):
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)

        self.save_trading_coins()

        for coin in tqdm(self.trading_coins):
            self.fetch_and_save_klines(coin, self.config["INTERVAL"], self.config["START_STR"], self.config["END_STR"])
