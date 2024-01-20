import pandas as pd
from sklearn.metrics import mean_squared_error

results = pd.read_csv("results.csv").dropna(axis=0)

for model in results.columns[4:]:
    print(model)
    print(mean_squared_error(results[model], results['market']))