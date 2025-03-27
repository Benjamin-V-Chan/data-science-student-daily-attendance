import pandas as pd
from pathlib import Path

# Load raw data
df = pd.read_csv("data/2018-2019_Daily_Attendance_20240429.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")

# Remove rows where Enrolled is zero or negative
df = df[df["Enrolled"] > 0]

# Calculate attendance and absence rates
df["AttendanceRate"] = df["Present"] / df["Enrolled"]
df["AbsenceRate"] = df["Absent"] / df["Enrolled"]

# Sort by School DBN and Date
df = df.sort_values(by=["School DBN", "Date"])

# Save cleaned data
Path("outputs").mkdir(exist_ok=True)
df.to_csv("outputs/cleaned_data.csv", index=False)
