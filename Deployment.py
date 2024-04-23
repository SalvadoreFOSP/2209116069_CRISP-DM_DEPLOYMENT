import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

URL = 'https://raw.githubusercontent.com/SalvadoreFOSP/Praktikum_DataMining/main/DataCleaned.csv'

df = pd.read_csv(URL)
df.head()

st.header("Dataset")
st.write(df)

# Menampilkan Panah Elbow
x_final = df.drop("Time", axis=1)

scaler = MinMaxScaler()

x_final_norm = scaler.fit_transform(x_final)

inertia_values = []

k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_final_norm)
    inertia_values.append(kmeans.inertia_)

plt.plot(k_range, inertia_values, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_range)
plt.show()

st.set_option("deprecation.showPyplotGlobalUse", False)
elbo_plot = st.pyplot()

# Sidebar
st.sidebar.subheader("Nilai Jumlah K")
clust = st.sidebar.slider("Pilih Jumlah Cluster :", 2,10,3,1)

from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

def k_means(n_clust):
    kmeans = KMeans(n_clusters=n_clust)

    # Membuat data contoh
    X, _ = make_blobs(n_samples=1000, centers=3, n_features=10, random_state=42)

    # Inisialisasi model PCA dengan menjaga 95% varians
    pca = PCA(n_components=0.95)

    # Mengurangi dimensi fitur menggunakan PCA
    X_pca = pca.fit_transform(X)

    # Melatih model KMeans dengan fitur yang telah direduksi
    kmeans.fit(X_pca)

    # Visualisasi data asli dengan dua fitur pertama
    plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', edgecolor='k', label='Data Asli')

    # Visualisasi pusat klaster setelah PCA
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centroid')

    plt.title('Data Asli dan Pusat Klaster setelah PCA')
    plt.xlabel('Rank')
    plt.ylabel('Feature')
    plt.legend()
    plt.grid(True)
    plt.show()

    st.header("Cluster Plot")
    st.pyplot()
    st.write(X_pca)

k_means(clust)
