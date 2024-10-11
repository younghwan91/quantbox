import os
import csv
import pandas as pd
from binance.client import Client
from binance.enums import HistoricalKlinesType
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger('BinanceDownloader')

class BinanceDownloader:
    def __init__(self, config, client):
        # self.client = self.initialize_client(config)
        self.client = client
        self.saving_dir = config["SAVING_DIR"]
        self.trading_coins = self.get_trading_coins(self.client, config["QUOTE_COIN"], config["N_COINS"])
        self.config = config

    def get_trading_coins(self, client, quote_coin, n_coins, sort_by_volume=True):
        futures_exchange_info = client.futures_exchange_info()
        trading_coins = [info['symbol'] for info in futures_exchange_info['symbols'] if info['symbol'].endswith(quote_coin) and info['status'] == 'TRADING']

        print(f"Number of trading pairs: {len(trading_coins)}")
        assert n_coins <= len(trading_coins), "N_COINS is greater than the number of trading coins"

        if sort_by_volume:
            logger.info("Fetching 24-hour trading volume for each coin")
            volume_dict = {}
            for coin in trading_coins:
                try:
                    ticker = client.get_ticker(symbol=coin)
                    volume_dict[coin] = float(ticker['volume'])
                except Exception as e:
                    logger.error(f"Error fetching ticker for {coin}: {e}")
                    continue

            # Sort coins based on trading volume in descending order
            sorted_coins = sorted(volume_dict.items(), key=lambda x: x[1], reverse=True)
            num_coins = len(sorted_coins)
            logger.info(f"Number of trading pairs: {num_coins}")

            logger.info(f"Selecting top {n_coins} coins based on trading volume")
            top_coins = [coin for coin, volume in sorted_coins[:n_coins]]
        else:
            logger.info(f"Selecting first {n_coins} coins")
            top_coins = trading_coins[:n_coins]

        return top_coins


    def save_trading_coins(self):
        with open(os.path.join(self.saving_dir, self.config["CSV_FILENAME"]), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.trading_coins)

    def fetch_and_save_klines(self, coin, interval, start_str, end_str):
        
        """
        Fetch and save klines for a given coin and interval.
        1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
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