import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import joblib

# Load dataset
data = pd.read_csv('../data/data.csv')

# Split features and target
X = data.drop('target', axis=1)
y = data['target']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print("Model MSE:", mse)

# Save model
joblib.dump(model, '../models/iris_model.pkl')

print("Model saved!")