# data-science-student-daily-attendance

# Project Overview

This project explores and models daily student attendance data from schools using a robust data science pipeline. It includes data cleaning, exploratory analysis, visualization, time series forecasting, clustering, and predictive modeling. The goal is to extract patterns and build tools to understand and predict student attendance behavior over time.

---

# Folder Structure

```
project-root/
├── data/                    # Raw dataset
│   └── 2018-2019_Daily_Attendance_20240429.csv
├── scripts/                 # Python scripts for each stage
│   ├── 01_load_and_clean.py
│   ├── 02_exploratory_analysis.py
│   ├── 03_visualizations.py
│   ├── 04_time_series_modeling.py
│   ├── 05_clustering_days.py
│   └── 06_predictive_modeling.py
├── outputs/                # Generated outputs
│   ├── cleaned_data.csv
│   ├── plots/
│   ├── models/
│   └── results/
├── requirements.txt
└── README.md
```

---

# Usage

## 1. Setup the Project:

Clone the repository.  
Ensure you have Python installed.  
Install required dependencies using the requirements.txt file:

```bash
pip install -r requirements.txt
```

## 2. Run Scripts Sequentially:

### Load and Clean the Data
```bash
python scripts/01_load_and_clean.py
```

### Perform Exploratory Analysis
```bash
python scripts/02_exploratory_analysis.py
```

### Generate Visualizations
```bash
python scripts/03_visualizations.py
```

### Time Series Modeling on Attendance Data
```bash
python scripts/04_time_series_modeling.py
```

### Cluster Days Based on Attendance Patterns
```bash
python scripts/05_clustering_days.py
```

### Predict Future Attendance Rates
```bash
python scripts/06_predictive_modeling.py
```

---

# Requirements

- Python >= 3.8
- pandas
- matplotlib
- scikit-learn
- statsmodels

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

# Acknowledgments

dataset name: School Student Daily Attendance  
dataset author: Sahir Maharaj  
dataset source: https://www.kaggle.com/datasets/sahirmaharajj/school-student-daily-attendance