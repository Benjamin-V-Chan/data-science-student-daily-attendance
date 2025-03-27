import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("outputs/cleaned_data.csv", parse_dates=["Date"])
df = df.sort_values(by=["School DBN", "Date"])

# Use one school for modeling
school = df["School DBN"].value_counts().idxmax()
school_df = df[df["School DBN"] == school].copy()

# Create lag features
school_df["Lag1"] = school_df["AttendanceRate"].shift(1)
school_df["Lag7"] = school_df["AttendanceRate"].shift(7)
school_df = school_df.dropna()

X = school_df[["Lag1", "Lag7"]]
y = school_df["AttendanceRate"]

# Train/test split
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

rmse = mean_squared_error(y_test, preds, squared=False)
print(f"Test RMSE: {rmse:.4f}")

plt.figure()
plt.plot(y_test.index, y_test.values, label="Actual")
plt.plot(y_test.index, preds, label="Predicted")
plt.legend()
plt.title(f"Random Forest Prediction - {school}")
plt.tight_layout()
plt.savefig("outputs/results/prediction_vs_actual.png")
