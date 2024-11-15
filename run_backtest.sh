#!/bin/bash

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Navigate to the project directory
cd "$(dirname "$0")"

# Run the main_backtest.py script with desired arguments
python main_backtest.py \
   --preprocess-data \
    --days 30 \
    --formation-days 10 \
    --trading-window 48 \
    --interval 30m \
    --hour-shift 11 \
    --n-coins 50 \
   # --download-data \

# Deactivate virtual environment if it was activated
# deactivate
