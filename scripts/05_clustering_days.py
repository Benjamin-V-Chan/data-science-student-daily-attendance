import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("outputs/cleaned_data.csv")

features = df[["AttendanceRate", "AbsenceRate", "Released"]]
kmeans = KMeans(n_clusters=3, random_state=42).fit(features)
df["Cluster"] = kmeans.labels_

centers = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns)

plt.figure()
centers.plot(kind="bar", title="Cluster Centers")
plt.xticks(range(3), [f"Cluster {i}" for i in range(3)], rotation=0)
plt.tight_layout()
Path("outputs/results").mkdir(parents=True, exist_ok=True)
plt.savefig("outputs/results/cluster_centers.png")

df.to_csv("outputs/results/clustered_days.csv", index=False)
