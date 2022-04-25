# algo_trader

## Scripts
1. collect_data.py
  - Generates datasets by pair, interval, number of candles
2. predict.py
  - Generates deeps neural network according to the previous script's parameters
3. trade.py
  - Generates a buy or sell prediction based on the above
4. algo_trader.py (1 and 2 combined)
  - Combines `collect_data` and `predict` above into one step for scheduler use
5. scheduler.py
  - Creating scheduled jobs for use in the cloude (Heroku)
6. check_trades.py
  - Cleanup predictions table and update with actual results of predicted candles

## Data Storage
- datasets
- models
- predictions

All stored in Google Cloud Storage