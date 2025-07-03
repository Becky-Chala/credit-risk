# src/proxy_target.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime

# 1. Calculate RFM (Recency, Frequency, Monetary)
def calculate_rfm(df, snapshot_date):
    import pandas as pd

    # Ensure TransactionStartTime is parsed as UTC timezone-aware
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce', utc=True)

    # Calculate RFM values
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
        'TransactionId': 'count',  # Frequency
        'Amount': 'sum'  # Monetary
    }).reset_index()

    # Rename columns
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']

    return rfm

# 2. Scale RFM data
def scale_rfm(rfm_df):
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    return scaled_rfm

# 3. Run K-Means clustering
def run_kmeans(rfm_scaled, n_clusters=3, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(rfm_scaled)
    return clusters

# 4. Analyze clusters (optional - for checking cluster centers)
def analyze_clusters(rfm_df, cluster_labels):
    # Add the cluster labels to the RFM dataframe
    rfm_df['Cluster'] = cluster_labels

    # Select only numeric columns for analysis (exclude CustomerId)
    numeric_cols = ['Recency', 'Frequency', 'Monetary']

    # Calculate mean values for each cluster
    cluster_summary = rfm_df.groupby('Cluster')[numeric_cols].mean()

    return cluster_summary

# 5. Label High Risk Customers
def label_high_risk_customers(rfm_df, high_risk_cluster_id):
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster_id).astype(int)
    return rfm_df[['CustomerId', 'is_high_risk']]

# 6. Merge the target back with main dataset
def merge_target_with_data(df, target_df):
    return df.merge(target_df, on='CustomerId', how='left')
