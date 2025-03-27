import pandas as pd
from pathlib import Path

df = pd.read_csv("outputs/cleaned_data.csv", parse_dates=["Date"])

# Summary statistics
summary = df[["AttendanceRate", "AbsenceRate"]].describe()

# Schools with highest/lowest avg attendance
school_avg = df.groupby("School DBN")["AttendanceRate"].mean().sort_values()
lowest = school_avg.head(5)
highest = school_avg.tail(5)

# Group by date: avg enrolled, present, absent
daily_summary = df.groupby("Date")[["Enrolled", "Present", "Absent"]].mean()

# Count of rows per school
school_counts = df["School DBN"].value_counts()

# Save to CSV
Path("outputs/results").mkdir(parents=True, exist_ok=True)
summary.to_csv("outputs/results/summary_statistics.csv")
lowest.to_csv("outputs/results/lowest_attendance_schools.csv")
highest.to_csv("outputs/results/highest_attendance_schools.csv")
daily_summary.to_csv("outputs/results/daily_enrollment_summary.csv")
school_counts.to_csv("outputs/results/school_row_counts.csv")
