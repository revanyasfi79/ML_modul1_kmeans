import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =====================================================================
# 1. MENYIAPKAN DATASET ALTERNATIF (Mall Customers Dataset)
# =====================================================================
# Membuat synthetic/simulated dataset yang strukturnya sama persis dengan
# Mall_Customers dataset asli untuk memastikan kode langsung bisa dijalankan.
np.random.seed(42)
data_size = 200

mall_data = {
    "CustomerID": range(1, data_size + 1),
    "Gender": np.random.choice(["Male", "Female"], size=data_size),
    "Age": np.random.randint(18, 70, size=data_size),
    "Annual Income (k$)": np.random.randint(15, 137, size=data_size),
    "Spending Score (1-100)": np.random.randint(1, 100, size=data_size),
}

df = pd.DataFrame(mall_data)
print("--- 5 Data Pertama Dataset Mall Customers ---")
print(df.head(), "\n")

# =====================================================================
# 2. SELEKSI FITUR (FEATURE SELECTION)
# =====================================================================
# Kita memilih dua fitur utama untuk clustering: Annual Income dan Spending Score
X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

# =====================================================================
# 3. PENENTUAN JUMLAH CLUSTER OPTIMAL (ELBOW METHOD)
# =====================================================================
wcss = []  # Within-Cluster Sum of Squares
cluster_range = range(1, 11)

for i in cluster_range:
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Visualisasi Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, wcss, marker="o", linestyle="--", color="b")
plt.title("Elbow Method untuk Menentukan Jumlah Cluster Optimal")
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("WCSS (Inertia)")
plt.grid(True)
plt.show()

# =====================================================================
# 4. MEMBUAT MODEL K-MEANS CLUSTERING (Ditentukan k=5 berdasarkan Elbow)
# =====================================================================
optimal_k = 5
kmeans_model = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42)
y_kmeans = kmeans_model.fit_predict(X)

# Menambahkan hasil label cluster ke dataframe asal
df["Cluster"] = y_kmeans

print("--- Hasil Pengelompokkan Cluster pada Data (5 Teratas) ---")
print(df.head(), "\n")

# =====================================================================
# 5. VISUALISASI HASIL CLUSTERING DAN CENTROID
# =====================================================================
plt.figure(figsize=(10, 7))

# Plot masing-masing cluster
colors = ["red", "blue", "green", "cyan", "magenta"]
cluster_labels = [
    "Cluster 1 (Pendapatan Rendah, Pengeluaran Tinggi)",
    "Cluster 2 (Pendapatan Sedang, Pengeluaran Sedang)",
    "Cluster 3 (Pendapatan Tinggi, Pengeluaran Tinggi)",
    "Cluster 4 (Pendapatan Rendah, Pengeluaran Rendah)",
    "Cluster 5 (Pendapatan Tinggi, Pengeluaran Rendah)",
]

for i in range(optimal_k):
    plt.scatter(
        X[y_kmeans == i, 0],
        X[y_kmeans == i, 1],
        s=100,
        c=colors[i],
        label=cluster_labels[i],
    )

# Plot Titik Pusat Cluster (Centroid)
plt.scatter(
    kmeans_model.cluster_centers_[:, 0],
    kmeans_model.cluster_centers_[:, 1],
    s=300,
    c="yellow",
    marker="*",
    edgecolors="black",
    label="Centroids (Pusat Cluster)",
)

plt.title("Segmentasi Pelanggan Menggunakan K-Means Clustering")
plt.xlabel("Annual Income (k$) / Pendapatan Tahunan")
plt.ylabel("Spending Score (1-100) / Skor Pengeluaran")
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
plt.grid(True)
plt.tight_layout()
plt.show()