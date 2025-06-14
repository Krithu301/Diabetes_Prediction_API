import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Create sample training data
data = {
    "age": [50, 30, 40],
    "bmi": [25, 30, 28],
    "bp": [80, 70, 90],
    "s1": [100, 110, 105],
    "s2": [90, 95, 100],
    "s3": [85, 80, 88],
    "s4": [4.5, 4.2, 4.8],
    "s5": [4.2, 4.1, 4.3],
    "s6": [90, 85, 88],
    "target": [150, 120, 130]
}

df = pd.DataFrame(data)

X = df.drop("target", axis=1)
y = df["target"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved")