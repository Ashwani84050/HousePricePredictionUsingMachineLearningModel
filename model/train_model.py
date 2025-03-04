import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load Dataset (Using California Housing Dataset)
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target

# Select Features and Target
X = df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup']]
y = df['Price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
with open('model/model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training complete! Saved as 'model.pkl'.")
