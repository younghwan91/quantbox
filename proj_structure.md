your_project/
│
├── binance_downloader.py       # Contains BinanceDownloader class
├── backtest_preprocessor.py    # Contains BacktestPreprocessor class
├── pair_trading_model.py       # Contains PairTradingModel class
├── main_backtest.py            # Main script to run the workflow
├── config.py                   # Configuration parameters (optional)
├── binance.json                # Binance API credentials
├── crypto_downloader.log       # Log file (generated after running)
├── data/                       # Data directory
│   ├── trading_coins.csv       # CSV of trading coins
│   ├── backtest/               # Backtest data
│   │   ├── pairs/              # Pairs data
│   │   └── pairs_for_backtest.csv
│   └── ...                     # Other data files
├── requirements.txt            # Python dependencies
└── .gitignore                  # To exclude sensitive files like binance.json

