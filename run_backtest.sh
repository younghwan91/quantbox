#!/bin/bash

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Navigate to the project directory
cd "$(dirname "$0")"

# Run the main_backtest.py script with desired arguments
python main_backtest.py \
   --preprocess-data \
    --days 90 \
    --formation-days 20 \
    --trading-window 120 \
    --interval 1h \
    --hour-shift 12 \
    --n-coins 200 \
   --download-data \

# Deactivate virtual environment if it was activated
# deactivate
