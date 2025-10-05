import pickle
pkl_path = './data/backtest/precomputed_values.pkl'
data = pickle.load(open(pkl_path, 'rb'))
print(len(data))
data = data[::30]
print(len(data))

# save

pickle.dump(data, open(pkl_path, 'wb'))
