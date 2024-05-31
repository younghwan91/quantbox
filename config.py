import datetime as dt
import os

CONFIG = {
    "FOLDER_NAME": 'futures',
    "SAVING_DIR": 'D:\crypto_historical',
    "CSV_FILENAME": "pairs_list.csv",
    "N_COINS": 50,
    "INTERVAL": '5m',
    "START_STR": '2 day ago',
    "END_STR": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "MAX_N_NPD": None,
    "TRAIN_RATIO": 0.9,
    "COINT_THRESHOLD": 0.01,
    "QUOTE_COIN": 'USDT',
    "TRADING_CHARGE": 0.0005,
    "TRADAING_TAX": 0,
    "MAX_TRADE_PORTION": 0.02,
}
saving_dir = os.path.join(CONFIG["SAVING_DIR"], CONFIG["FOLDER_NAME"], CONFIG["INTERVAL"])
# add saving_dir to CONFIG
CONFIG["SAVING_DIR"] = saving_dir