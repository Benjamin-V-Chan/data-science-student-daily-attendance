import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pathlib import Path
import joblib

# Load and prepare data
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

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Evaluate
rmse = mean_squared_error(y_test, preds, squared=False)
print(f"Test RMSE: {rmse:.4f}")

# Ensure output folders exist
Path("outputs/results").mkdir(parents=True, exist_ok=True)
Path("outputs/models").mkdir(parents=True, exist_ok=True)

# Save plot
plt.figure()
plt.plot(y_test.index, y_test.values, label="Actual")
plt.plot(y_test.index, preds, label="Predicted")
plt.legend()
plt.title(f"Random Forest Prediction - {school}")
plt.tight_layout()
plt.savefig("outputs/results/prediction_vs_actual.png")

# Save predictions to CSV
results_df = pd.DataFrame({
    "Date": school_df.iloc[y_test.index]["Date"].values,
    "Actual": y_test.values,
    "Predicted": preds
})
results_df.to_csv("outputs/results/predictions.csv", index=False)

# Save RMSE score
with open("outputs/results/rmse.txt", "w") as f:
    f.write(f"Test RMSE: {rmse:.4f}")

# Save trained model
joblib.dump(model, "outputs/models/random_forest_model.pkl")
