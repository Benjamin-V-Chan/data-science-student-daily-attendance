import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("outputs/cleaned_data.csv", parse_dates=["Date"])

Path("outputs/plots").mkdir(parents=True, exist_ok=True)

# Average attendance rate over time
daily_avg = df.groupby("Date")["AttendanceRate"].mean()

plt.figure()
daily_avg.plot(title="Average Attendance Rate Over Time")
plt.ylabel("Attendance Rate")
plt.tight_layout()
plt.savefig("outputs/plots/attendance_trend.png")

# Histogram of attendance and absence rates
plt.figure()
df["AttendanceRate"].hist(bins=50)
plt.title("Distribution of Attendance Rate")
plt.xlabel("Attendance Rate")
plt.tight_layout()
plt.savefig("outputs/plots/attendance_hist.png")

plt.figure()
df["AbsenceRate"].hist(bins=50)
plt.title("Distribution of Absence Rate")
plt.xlabel("Absence Rate")
plt.tight_layout()
plt.savefig("outputs/plots/absence_hist.png")

# Scatter: Enrolled vs Absent
plt.figure()
plt.scatter(df["Enrolled"], df["Absent"], alpha=0.3)
plt.title("Enrolled vs Absent")
plt.xlabel("Enrolled")
plt.ylabel("Absent")
plt.tight_layout()
plt.savefig("outputs/plots/enrolled_vs_absent.png")
