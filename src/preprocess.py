import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()

# Convert to pandas dataframe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target column
df['target'] = iris.target

# Save dataset to data folder
df.to_csv('../data/data.csv', index=False)

print("Dataset saved to data/data.csv")