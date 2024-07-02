import datetime as dt
import os

hour_per_day = 24
total_hours = hour_per_day*3

CONFIG = {
    "FOLDER_NAME": 'futures',
    "SAVING_DIR": 'data',
    "CSV_FILENAME": "pairs_list.csv",
    "N_COINS": 100,
    "INTERVAL": '5m',
    "START_STR": f'{total_hours} hour ago',
    "END_STR": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "COINT_THRESHOLD": 0.01,
    "TRAIN_RATIO": 1.0,
    "QUOTE_COIN": 'USDT',
    "TRADING_CHARGE": 0.0005,
    "TRADAING_TAX": 0,
    "MAX_TRADE_PORTION": 0.02,
    "CORR_THRESHOLD": 0.3,
}
saving_dir = os.path.join(CONFIG["SAVING_DIR"], CONFIG["FOLDER_NAME"], CONFIG["INTERVAL"])
# add saving_dir to CONFIG
CONFIG["SAVING_DIR"] = saving_dir