"""
Exploratory Analysis Methods

This module implements exploratory data analysis methods like PCA, UMAP,
t-SNE, and clustering techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def perform_pca_analysis(dataset_data: Dict[str, pd.DataFrame],
                        params: Dict[str, Any],
                        progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Perform Principal Component Analysis on spectral data.
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - n_components: Number of components (default 3)
            - scaling: Scaler type ('StandardScaler', 'MinMaxScaler', 'None')
            - show_loadings: Show PC loadings plot
            - show_scree: Show scree plot
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary containing:
            - primary_figure: Scores plot (PC1 vs PC2)
            - secondary_figure: Loadings/scree plot
            - data_table: PC scores DataFrame
            - summary_text: Analysis summary
            - raw_results: Full PCA results
    """
    if progress_callback:
        progress_callback(10)
    
    # Get parameters
    n_components = params.get("n_components", 3)
    scaling_type = params.get("scaling", "StandardScaler")
    show_loadings = params.get("show_loadings", True)
    show_scree = params.get("show_scree", True)
    
    # Combine all datasets
    all_spectra = []
    labels = []
    
    for dataset_name, df in dataset_data.items():
        spectra_matrix = df.values.T  # Shape: (n_spectra, n_wavenumbers)
        all_spectra.append(spectra_matrix)
        labels.extend([dataset_name] * spectra_matrix.shape[0])
    
    X = np.vstack(all_spectra)
    wavenumbers = dataset_data[list(dataset_data.keys())[0]].index.values
    
    if progress_callback:
        progress_callback(30)
    
    # Apply scaling
    if scaling_type == "StandardScaler":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaling_type == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    if progress_callback:
        progress_callback(50)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)
    
    if progress_callback:
        progress_callback(70)
    
    # Create primary figure: PC1 vs PC2 scores plot
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, dataset_label in enumerate(unique_labels):
        mask = np.array([l == dataset_label for l in labels])
        ax1.scatter(scores[mask, 0], scores[mask, 1],
                   c=[colors[i]], label=dataset_label,
                   alpha=0.7, s=50)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                   fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                   fontsize=12)
    ax1.set_title('PCA Score Plot', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Create secondary figure: Loadings or Scree plot
    if show_loadings and show_scree:
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig2, ax2a = plt.subplots(figsize=(10, 6))
        ax2b = None
    
    if show_loadings:
        # Plot loadings for PC1 and PC2
        ax2a.plot(wavenumbers, pca.components_[0], label='PC1', linewidth=1.5)
        ax2a.plot(wavenumbers, pca.components_[1], label='PC2', linewidth=1.5)
        if n_components >= 3:
            ax2a.plot(wavenumbers, pca.components_[2], label='PC3', linewidth=1.5)
        ax2a.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
        ax2a.set_ylabel('Loading', fontsize=12)
        ax2a.set_title('PCA Loadings', fontsize=14, fontweight='bold')
        ax2a.legend()
        ax2a.grid(True, alpha=0.3)
        ax2a.invert_xaxis()
    
    if show_scree and ax2b is not None:
        # Scree plot
        variance_ratio = pca.explained_variance_ratio_ * 100
        cumulative_variance = np.cumsum(variance_ratio)
        
        ax2b.bar(range(1, len(variance_ratio) + 1), variance_ratio,
                alpha=0.7, label='Individual')
        ax2b.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                 'ro-', linewidth=2, markersize=8, label='Cumulative')
        ax2b.set_xlabel('Principal Component', fontsize=12)
        ax2b.set_ylabel('Variance Explained (%)', fontsize=12)
        ax2b.set_title('Scree Plot', fontsize=14, fontweight='bold')
        ax2b.legend()
        ax2b.grid(True, alpha=0.3)
    elif show_scree and not show_loadings:
        variance_ratio = pca.explained_variance_ratio_ * 100
        cumulative_variance = np.cumsum(variance_ratio)
        
        ax2a.bar(range(1, len(variance_ratio) + 1), variance_ratio,
                alpha=0.7, label='Individual')
        ax2a.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                 'ro-', linewidth=2, markersize=8, label='Cumulative')
        ax2a.set_xlabel('Principal Component', fontsize=12)
        ax2a.set_ylabel('Variance Explained (%)', fontsize=12)
        ax2a.set_title('Scree Plot', fontsize=14, fontweight='bold')
        ax2a.legend()
        ax2a.grid(True, alpha=0.3)
    
    if progress_callback:
        progress_callback(90)
    
    # Create data table
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    scores_df = pd.DataFrame(scores, columns=pc_columns)
    scores_df['Dataset'] = labels
    
    # Summary text
    total_variance = np.sum(pca.explained_variance_ratio_[:3]) * 100
    summary = f"PCA completed with {n_components} components.\n"
    summary += f"First 3 PCs explain {total_variance:.1f}% of variance.\n"
    summary += f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, "
    summary += f"PC2: {pca.explained_variance_ratio_[1]*100:.1f}%"
    if n_components >= 3:
        summary += f", PC3: {pca.explained_variance_ratio_[2]*100:.1f}%"
    
    return {
        "primary_figure": fig1,
        "secondary_figure": fig2 if show_loadings or show_scree else None,
        "data_table": scores_df,
        "summary_text": summary,
        "detailed_summary": f"Scaling: {scaling_type}\nTotal spectra: {X.shape[0]}",
        "raw_results": {
            "pca_model": pca,
            "scores": scores,
            "loadings": pca.components_,
            "explained_variance": pca.explained_variance_ratio_
        }
    }


def perform_umap_analysis(dataset_data: Dict[str, pd.DataFrame],
                         params: Dict[str, Any],
                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Perform UMAP dimensionality reduction.
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - n_neighbors: Number of neighbors (default 15)
            - min_dist: Minimum distance (default 0.1)
            - n_components: Number of components (default 2)
            - metric: Distance metric (default 'euclidean')
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with embedding plot and results
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is not installed. Install with: pip install umap-learn")
    
    if progress_callback:
        progress_callback(10)
    
    # Get parameters
    n_neighbors = params.get("n_neighbors", 15)
    min_dist = params.get("min_dist", 0.1)
    n_components = params.get("n_components", 2)
    metric = params.get("metric", "euclidean")
    
    # Combine all datasets
    all_spectra = []
    labels = []
    
    for dataset_name, df in dataset_data.items():
        spectra_matrix = df.values.T
        all_spectra.append(spectra_matrix)
        labels.extend([dataset_name] * spectra_matrix.shape[0])
    
    X = np.vstack(all_spectra)
    
    if progress_callback:
        progress_callback(30)
    
    # Perform UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=42
    )
    embedding = reducer.fit_transform(X)
    
    if progress_callback:
        progress_callback(80)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, dataset_label in enumerate(unique_labels):
        mask = np.array([l == dataset_label for l in labels])
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                  c=[colors[i]], label=dataset_label,
                  alpha=0.7, s=50)
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('UMAP Projection', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Create data table
    embedding_df = pd.DataFrame(
        embedding,
        columns=[f'UMAP{i+1}' for i in range(n_components)]
    )
    embedding_df['Dataset'] = labels
    
    summary = f"UMAP completed with {n_components} components.\n"
    summary += f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}"
    
    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": embedding_df,
        "summary_text": summary,
        "detailed_summary": f"Total spectra: {X.shape[0]}",
        "raw_results": {"embedding": embedding, "reducer": reducer}
    }


def perform_tsne_analysis(dataset_data: Dict[str, pd.DataFrame],
                         params: Dict[str, Any],
                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Perform t-SNE dimensionality reduction.
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - perplexity: Perplexity parameter (default 30)
            - learning_rate: Learning rate (default 200)
            - n_iter: Number of iterations (default 1000)
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with embedding plot and results
    """
    if progress_callback:
        progress_callback(10)
    
    # Get parameters
    perplexity = params.get("perplexity", 30)
    learning_rate = params.get("learning_rate", 200)
    n_iter = params.get("n_iter", 1000)
    
    # Combine all datasets
    all_spectra = []
    labels = []
    
    for dataset_name, df in dataset_data.items():
        spectra_matrix = df.values.T
        all_spectra.append(spectra_matrix)
        labels.extend([dataset_name] * spectra_matrix.shape[0])
    
    X = np.vstack(all_spectra)
    
    if progress_callback:
        progress_callback(30)
    
    # Perform t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=42
    )
    embedding = tsne.fit_transform(X)
    
    if progress_callback:
        progress_callback(80)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, dataset_label in enumerate(unique_labels):
        mask = np.array([l == dataset_label for l in labels])
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                  c=[colors[i]], label=dataset_label,
                  alpha=0.7, s=50)
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE Projection', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Create data table
    embedding_df = pd.DataFrame(embedding, columns=['tSNE1', 'tSNE2'])
    embedding_df['Dataset'] = labels
    
    summary = f"t-SNE completed with 2 components.\n"
    summary += f"Parameters: perplexity={perplexity}, learning_rate={learning_rate}, n_iter={n_iter}"
    
    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": embedding_df,
        "summary_text": summary,
        "detailed_summary": f"Total spectra: {X.shape[0]}",
        "raw_results": {"embedding": embedding}
    }


def perform_hierarchical_clustering(dataset_data: Dict[str, pd.DataFrame],
                                   params: Dict[str, Any],
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Perform hierarchical clustering analysis.
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - linkage_method: Linkage method (default 'ward')
            - distance_metric: Distance metric (default 'euclidean')
            - n_clusters: Number of clusters to color (optional)
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with dendrogram and results
    """
    if progress_callback:
        progress_callback(10)
    
    # Get parameters
    linkage_method = params.get("linkage_method", "ward")
    distance_metric = params.get("distance_metric", "euclidean")
    n_clusters = params.get("n_clusters", None)
    
    # Combine all datasets
    all_spectra = []
    labels = []
    
    for dataset_name, df in dataset_data.items():
        spectra_matrix = df.values.T
        all_spectra.append(spectra_matrix)
        labels.extend([dataset_name] * spectra_matrix.shape[0])
    
    X = np.vstack(all_spectra)
    
    if progress_callback:
        progress_callback(40)
    
    # Perform hierarchical clustering
    if linkage_method == 'ward':
        Z = linkage(X, method='ward')
    else:
        distances = pdist(X, metric=distance_metric)
        Z = linkage(distances, method=linkage_method)
    
    if progress_callback:
        progress_callback(70)
    
    # Create dendrogram
    fig, ax = plt.subplots(figsize=(12, 8))
    
    dendrogram(Z, ax=ax, labels=labels, leaf_font_size=8)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
    
    summary = f"Hierarchical clustering completed.\n"
    summary += f"Linkage: {linkage_method}, Distance metric: {distance_metric}\n"
    summary += f"Total spectra: {X.shape[0]}"
    
    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": None,
        "summary_text": summary,
        "detailed_summary": f"Linkage matrix shape: {Z.shape}",
        "raw_results": {"linkage_matrix": Z, "labels": labels}
    }


def perform_kmeans_clustering(dataset_data: Dict[str, pd.DataFrame],
                              params: Dict[str, Any],
                              progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Perform K-means clustering analysis.
    
    Args:
        dataset_data: Dictionary of {dataset_name: DataFrame}
        params: Analysis parameters
            - n_clusters: Number of clusters (default 3)
            - max_iter: Maximum iterations (default 300)
            - n_init: Number of initializations (default 10)
            - show_pca: Show clusters in PCA space
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with cluster visualization and results
    """
    if progress_callback:
        progress_callback(10)
    
    # Get parameters
    n_clusters = params.get("n_clusters", 3)
    max_iter = params.get("max_iter", 300)
    n_init = params.get("n_init", 10)
    show_pca = params.get("show_pca", True)
    
    # Combine all datasets
    all_spectra = []
    labels = []
    
    for dataset_name, df in dataset_data.items():
        spectra_matrix = df.values.T
        all_spectra.append(spectra_matrix)
        labels.extend([dataset_name] * spectra_matrix.shape[0])
    
    X = np.vstack(all_spectra)
    
    if progress_callback:
        progress_callback(30)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter,
                   n_init=n_init, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    if progress_callback:
        progress_callback(60)
    
    # Create visualization
    if show_pca:
        # Project to PCA space for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        centers_pca = pca.transform(kmeans.cluster_centers_)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot clusters
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        for i in range(n_clusters):
            mask = cluster_labels == i
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[colors[i]], label=f'Cluster {i+1}',
                      alpha=0.7, s=50)
        
        # Plot centroids
        ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
                  c='red', marker='X', s=200, edgecolors='black',
                  linewidths=2, label='Centroids')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title('K-means Clustering (PCA Projection)', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    else:
        # Just show cluster assignments
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(cluster_labels)), cluster_labels)
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Cluster ID', fontsize=12)
        ax.set_title('K-means Cluster Assignments', fontsize=14, fontweight='bold')
    
    if progress_callback:
        progress_callback(90)
    
    # Create data table
    results_df = pd.DataFrame({
        'Dataset': labels,
        'Cluster': cluster_labels
    })
    
    # Calculate cluster statistics
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    
    summary = f"K-means clustering completed with {n_clusters} clusters.\n"
    for i in range(n_clusters):
        count = cluster_counts.get(i, 0)
        pct = count / len(cluster_labels) * 100
        summary += f"Cluster {i+1}: {count} spectra ({pct:.1f}%)\n"
    
    return {
        "primary_figure": fig,
        "secondary_figure": None,
        "data_table": results_df,
        "summary_text": summary,
        "detailed_summary": f"Inertia: {kmeans.inertia_:.2f}\nIterations: {kmeans.n_iter_}",
        "raw_results": {
            "kmeans_model": kmeans,
            "cluster_labels": cluster_labels,
            "cluster_centers": kmeans.cluster_centers_
        }
    }
