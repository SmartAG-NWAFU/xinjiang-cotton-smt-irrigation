import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from utils.csv_convert_raster import RasterConverter


def calculate_correlation_matrix(data):
    """Calculate the correlation matrix for specified columns."""
    return data[['mean_tmin', 'mean_tmax', 'mean_prcp', 'mean_et', 'mean_gdd']].corr()


def plot_heatmap(correlation_matrix, output_dir):
    """Plot and save a heatmap of the correlation matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    output_file = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def standardize_data(data):
    """Standardize the specified columns of the data."""
    scaler = StandardScaler()
    return scaler.fit_transform(data[['mean_tmin', 'mean_tmax', 'mean_prcp', 'mean_et', 'mean_gdd']])

def determine_optimal_clusters(scaled_data, output_dir):
    """Determine the optimal number of clusters using the elbow method."""
    inertia = []
    for n in range(1, 11):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.savefig(os.path.join(output_dir, 'determine_optimal_clusters.png'), dpi=300, bbox_inches='tight')


def perform_clustering(data, scaled_data, n_clusters):
    """Perform KMeans clustering and add the cluster labels to the data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(scaled_data)
    return data

def save_clustered_data(data, output_dir):
    """Save the clustered data to a CSV file."""
    output_data = data[['ID', 'Cluster']]
    output_data.to_csv(os.path.join(output_dir, 'cotton_areas_clusted_data.csv'), index=False)
    return output_data

# Convert the results to raster
def convert_results_to_raster(results, output_file_dir):
    cotton_units = pd.read_csv('../data/grid_units/units.csv')
    results = pd.merge(results, cotton_units, on='ID', how='left')

    value_columns = ['Cluster']
    converter = RasterConverter(
        df=results,
        value_columns=value_columns,
        res=0.05,
        output_dir=output_file_dir,
        lat_column='lat',
        lon_column='lon',
        crs='EPSG:4326'
    )
    converter.csv_to_tif()

def main():
    # Load data
    data = pd.read_csv('../results/xinjiang_weather/grid_units_aquacrop_weather_summary.csv')
    
    # Output files paths
    output_dir_path = '../results/cotton_area_clustered'
    # Check if directory exists, if not create it
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    # Calculate and plot correlation matrix
    correlation_matrix = calculate_correlation_matrix(data)
    plot_heatmap(correlation_matrix, output_dir_path)
    
    # Standardize data
    scaled_data = standardize_data(data)
    # Determine optimal clusters
    determine_optimal_clusters(scaled_data, output_dir_path)
    
    # Input optimal clusters
    optimal_clusters = 2
    # Perform clustering
    clustered_data = perform_clustering(data, scaled_data, optimal_clusters)
    # Save clustered data
    cluster_results = save_clustered_data(clustered_data, output_dir_path)
    convert_results_to_raster(cluster_results, output_dir_path)

if __name__ == "__main__":
    main()