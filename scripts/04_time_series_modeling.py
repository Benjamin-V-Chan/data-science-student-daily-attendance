import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path

df = pd.read_csv("outputs/cleaned_data.csv", parse_dates=["Date"])

# Choose a school with lots of records
top_school = df["School DBN"].value_counts().idxmax()
school_df = df[df["School DBN"] == top_school].set_index("Date")

# Weekly average attendance
ts = school_df["AttendanceRate"].resample("W").mean()

# ARIMA model
model = ARIMA(ts, order=(2, 1, 2)).fit()
forecast = model.predict(start=ts.index[-20], end=ts.index[-1] + pd.Timedelta(weeks=10))

plt.figure()
ts.plot(label="Actual")
forecast.plot(label="Forecast")
plt.title(f"Attendance Forecast - {top_school}")
plt.legend()
plt.tight_layout()
Path("outputs/models").mkdir(parents=True, exist_ok=True)
plt.savefig(f"outputs/models/arima_forecast_{top_school}.png")
