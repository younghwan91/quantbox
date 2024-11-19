import pickle

data = pickle.load(open('./data/backtest/precomputed_values.pkl', 'rb'))

# cut the data to last 1000 rows
data = data[-1000:]

# save
pickle.dump(data, open('./data/backtest/precomputed_values.pkl', 'wb'))
