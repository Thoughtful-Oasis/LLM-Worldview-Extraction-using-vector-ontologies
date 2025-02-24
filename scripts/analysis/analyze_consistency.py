"""
Spotify Genre Analysis and Visualization Module

This module provides tools for analyzing and visualizing the consistency of Spotify genre searches
and their distribution in feature space. It performs several types of analysis:

1. Genre Cluster Analysis:
   - PCA-based visualization of genre clusters
   - Convex hull generation for genre boundaries
   - Genre midpoint analysis in both original and PCA space

2. Consistency Analysis:
   - Calculation of various distance metrics (pairwise, centroid)
   - Volume coverage analysis in n-dimensional space
   - Statistical comparison against random baselines
   - Generation of multiple visualization plots

3. Random Projections:
   - Alternative 2D visualizations using random projections
   - Distance preservation analysis

The module expects data from a SQLite database containing Spotify search results
and a bin_edges.json file defining the feature ranges.

Dependencies:
    - numpy
    - scipy
    - matplotlib
    - sklearn
    - joblib
    - sqlite3

Typical usage:
    python analyze_consistency.py

Output:
    Generates multiple visualization files in the spotify_test_results directory:
    - clusters/: PCA and projection visualizations
    - consistency/: Statistical analysis plots
    - models/: Saved PCA models
"""

import os
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import scipy
import numpy as np
import joblib
import sqlite3
import random
from typing import Dict, List, Set, Tuple, Optional, Any, Union

def ensure_directories() -> None:
    """
    Create all necessary subdirectories for storing analysis results.

    Creates the following directory structure under 'results/':
        - clusters/: For cluster visualizations
        - consistency/: For consistency analysis results
        - models/: For saved models
    """
    base_dir = 'results'
    subdirs = [
        'clusters',
        'consistency',
        'models',
    ]
    
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)


def get_range_midpoint(range_str: str) -> float:
    """
    Calculate the midpoint of a numeric range specified as a string.

    Args:
        range_str (str): A string representing a numeric range (e.g., '0.0-0.36' or '90-110')

    Returns:
        float: The midpoint value of the range

    Example:
        >>> get_range_midpoint('0.0-0.36')
        0.18
    """
    start, end = map(float, range_str.split('-'))
    return (start + end) / 2


def generate_random_binned_point() -> np.ndarray:
    """
    Generate a random point in feature space using the binned feature ranges.

    Uses the global FEATURE_RANGES to select random bins for each feature
    and returns their midpoints as coordinates.

    Returns:
        numpy.ndarray: A point in feature space with randomly selected bin midpoints
    """
    point = []
    for feature_range in FEATURE_RANGES:
        # Randomly select a bin index
        bin_idx = random.randint(0, len(feature_range['ranges']) - 1)
        # Convert the selected bin index string (like '0.0-0.36') into a midpoint float
        point.append(get_range_midpoint(feature_range['ranges'][bin_idx]))
    return np.array(point)


def convert_location_to_values(location: List[int], 
                             feature_ranges: List[Dict[str, Any]]) -> List[float]:
    """
    Convert location indices to actual feature values using range midpoints.

    Args:
        location (list): List of indices, one per feature, indicating which bin was selected
        feature_ranges (list): List of feature range definitions from bin_edges.json

    Returns:
        list: The actual feature values corresponding to the midpoints of selected bins

    Raises:
        ValueError: If any location index is not an integer
    """
    values = []
    for idx, feature in zip(location, feature_ranges):
        # The location index must be an integer; if not, it's an error.
        if not float(idx).is_integer():
            raise ValueError(f"Location index {idx} is not an integer! All indices must be integers.")
        idx = int(idx)
        midpoint = get_range_midpoint(feature['ranges'][idx])
        values.append(midpoint)
    return values


###############################################
# LOAD FEATURE RANGES (bin_edges.json) GLOBAL #
###############################################
with open('data/bin_edges.json', 'r') as f:
    bin_edge_data = json.load(f)

# Normalize tempo using a more musically meaningful approach
tempo_edges = [float(edge) for edge in bin_edge_data["edges"]["tempo"]]
# Most music falls between 60-180 BPM, with outliers handled gracefully
min_meaningful_tempo = 60.0  # Typical slowest tempo (largo)
max_meaningful_tempo = 180.0  # Typical fastest tempo (presto)

bin_edge_data["edges"]["tempo"] = [
    max(0.0, min(1.0, (tempo - min_meaningful_tempo) / (max_meaningful_tempo - min_meaningful_tempo)))
    for tempo in tempo_edges
]

# We create a global variable for feature ranges
FEATURE_RANGES = [
    {
        'name': feature,
        'ranges': [
            f'{bin_edge_data["edges"][feature][i]}-{bin_edge_data["edges"][feature][i+1]}' 
            for i in range(len(bin_edge_data["edges"][feature]) - 1)
        ]
    } 
    for feature in bin_edge_data['feature_order']
]


##################################################
# PLOTTING FUNCTIONS FOR GENRE CLUSTERS (PCA)    #
##################################################

def plot_genre_clusters(significant_genres: Optional[Set[str]] = None) -> None:
    """
    Create and save visualizations of genre clusters using PCA.

    Generates three plots:
    1. Full cluster visualization with convex hulls
    2. Genre midpoints in original PCA space
    3. Genre midpoints in new PCA space

    Args:
        significant_genres (set, optional): Set of genre names to analyze.
            If None, analyzes all genres in the database.

    Saves:
        - PCA model to 'models/pca_model.joblib'
        - Plots to 'clusters/' directory:
            - genre_clusters.png
            - genre_midpoints_original_pca.png
            - genre_midpoints_new_pca.png
    """
    # Connect to the database of distributions
    search_conn = sqlite3.connect('spotify_search_distributions.db')
    search_c = search_conn.cursor()
    
    # Either filter genres or get all
    if significant_genres:
        placeholders = ','.join('?' * len(significant_genres))
        query = f'SELECT query_genre, search_results FROM genre_search_distributions WHERE query_genre IN ({placeholders})'
        results = search_c.execute(query, list(significant_genres)).fetchall()
    else:
        results = search_c.execute('SELECT query_genre, search_results FROM genre_search_distributions').fetchall()
    
    # Collect points by genre
    locations = {}
    for genre, search_results_json in results:
        search_results = json.loads(search_results_json)
        for search in search_results:
            if 'is_average' not in search:  # exclude average "summary" points
                if genre not in locations:
                    locations[genre] = []
                actual_values = convert_location_to_values(search['location'], FEATURE_RANGES)
                locations[genre].append(actual_values)
    
    # Prepare data for PCA
    X = []
    y = []
    for genre, locs in locations.items():
        X.extend(locs)
        y.extend([genre] * len(locs))
    
    X = np.array(X)

    # Handle the case: if no data
    if len(X) == 0:
        print("No data to plot in plot_genre_clusters.")
        return
    
    # Perform PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    
    # Save the PCA model
    joblib.dump(pca, 'results/models/pca_model.joblib')
    
    # Create a figure for the cluster scatter plot
    plt.figure(figsize=(20, 12))

    # Assign colors to genres
    unique_genres = list(set(y))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_genres)))
    
    # Plot each genre's points and hull
    for genre, color in zip(unique_genres, colors):
        mask = [label == genre for label in y]
        points = X_reduced[mask]
        
        # Scatter
        plt.scatter(points[:, 0], points[:, 1],
                    color=color, alpha=0.6, s=100, label=genre)

        # Attempt convex hull
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_points = np.vstack((hull_points, hull_points[0]))
                # Fill hull area
                plt.fill(hull_points[:, 0], hull_points[:, 1],
                         color=color, alpha=0.2)
                # Outline hull
                plt.plot(hull_points[:, 0], hull_points[:, 1],
                         color=color, alpha=0.5)
                # Place label near center
                center = points[hull.vertices].mean(axis=0)
                plt.annotate(genre, (center[0], center[1]),
                             fontsize=12, fontweight='bold',
                             ha='center', va='center', color=color)
            except scipy.spatial._qhull.QhullError:
                print(f"Couldn't create convex hull for {genre} - possible coplanar or insufficient points.")
    
    plt.title('Genre Clusters with Convex Hulls (PCA 2D)', fontsize=14, pad=20)
    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Second Principal Component', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    # Save this figure
    plt.savefig('results/clusters/genre_clusters.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Next: create separate plots for genre midpoints in original PCA space and in new PCA space
    
    # Compute average (midpoint) for each genre in the original dimensional space
    avg_locations = {}
    for genre, search_results_json in results:
        search_results = json.loads(search_results_json)
        real_values = []
        for search in search_results:
            if 'is_average' not in search:
                actual_values = convert_location_to_values(search['location'], FEATURE_RANGES)
                real_values.append(actual_values)
        
        if real_values:
            avg_locations[genre] = np.mean(real_values, axis=0)
    
    # We'll do the midpoints in the *same* PCA space
    # and then we'll do a *new* PCA just on midpoints alone.
    if len(avg_locations) == 0:
        print("No midpoints to plot in plot_genre_clusters.")
        return

    # 1) Midpoints in the original PCA space
    plt.figure(figsize=(10, 8))

    # Convert dictionary to arrays for transformation
    genres_for_avg = list(avg_locations.keys())
    avg_points = np.array([avg_locations[g] for g in genres_for_avg])

    # Project these average points using the PCA model we just fit
    avg_points_reduced = pca.transform(avg_points)

    # Plot them
    for g, color in zip(genres_for_avg, colors):
        idx = genres_for_avg.index(g)
        pt = avg_points_reduced[idx]
        plt.scatter(pt[0], pt[1], color=color, alpha=0.8, s=150)
        # Label
        plt.annotate(g, xy=(pt[0], pt[1]),
                     xytext=(10, 10),
                     textcoords='offset points',
                     fontsize=10, fontweight='bold',
                     color=color)

    plt.title('Genre Midpoints (in Original PCA Space)', fontsize=14)
    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Second Principal Component', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    plt.savefig('results/clusters/genre_midpoints_original_pca.png',
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2) Midpoints in a new PCA space, fitted only to the midpoints
    plt.figure(figsize=(10, 8))

    pca_avg = PCA(n_components=2)
    avg_points_reduced_new = pca_avg.fit_transform(avg_points)

    for g, color in zip(genres_for_avg, colors):
        idx = genres_for_avg.index(g)
        pt = avg_points_reduced_new[idx]
        plt.scatter(pt[0], pt[1], color=color, alpha=0.8, s=150)
        # Label
        plt.annotate(g, xy=(pt[0], pt[1]),
                     xytext=(10, 10),
                     textcoords='offset points',
                     fontsize=10, fontweight='bold',
                     color=color)

    plt.title('Genre Midpoints (New PCA Space)', fontsize=14)
    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Second Principal Component', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    plt.savefig('results/clusters/genre_midpoints_new_pca.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    # Close database connection
    search_conn.close()


###################################################################
# CONSISTENCY ANALYSIS: LOADING DATA, CALCULATING METRICS, PLOTTING
###################################################################

def analyze_genre_search_consistency(
    significant_genres: Optional[Set[str]] = None
) -> Tuple[Dict[str, Dict[str, float]], 
           Dict[str, float], 
           Dict[str, float]]:
    """
    Perform comprehensive analysis of genre search consistency.

    Analyzes multiple aspects of genre consistency:
    - Distance metrics (pairwise, centroid)
    - Volume coverage in feature space
    - Statistical significance vs random baseline
    - Visualization of results

    Args:
        significant_genres (set, optional): Set of genre names to analyze.
            If None, analyzes all genres in the database.

    Returns:
        tuple: (genre_metrics, random_baseline, improvements)
            - genre_metrics (dict): Metrics calculated for each genre
            - random_baseline (dict): Random baseline metrics
            - improvements (dict): Improvement percentages over random

    Saves multiple visualization files to the 'consistency/' directory.
    """
    # Connect to the DB
    search_conn = sqlite3.connect('spotify_search_distributions.db')
    search_c = search_conn.cursor()

    # Get the relevant data
    if significant_genres:
        placeholders = ','.join('?' * len(significant_genres))
        query = f'SELECT query_genre, search_results FROM genre_search_distributions WHERE query_genre IN ({placeholders})'
        results = search_c.execute(query, list(significant_genres)).fetchall()
    else:
        results = search_c.execute('SELECT query_genre, search_results FROM genre_search_distributions').fetchall()

    # Prepare dictionary to store metrics for each genre
    genre_metrics = {}

    # Keep track of attempts and failures to create hulls
    hull_failures = {
        'insufficient_points': 0,
        'coplanar_or_correlated': 0,
        'successful': 0,
        'effective_dimensions': []
    }
    
    # -------------------
    # METRICS PER GENRE
    # -------------------
    for genre, search_results_json in results:
        search_results = json.loads(search_results_json)
        
        points = []
        for search in search_results:
            if 'is_average' not in search:
                actual_values = convert_location_to_values(search['location'], FEATURE_RANGES)
                points.append(actual_values)
        
        if len(points) < 2:
            print(f"Skipping {genre} - insufficient points (need >=2).")
            continue
            
        points = np.array(points)
        
        # Calculate metrics including Manhattan distance
        metrics = {}

        # Euclidean distances (existing)
        pairwise_distances = scipy.spatial.distance.pdist(points)
        metrics['avg_pairwise_distance'] = np.mean(pairwise_distances)
        metrics['std_pairwise_distance'] = np.std(pairwise_distances)
        
        # Manhattan distances (new)
        manhattan_distances = scipy.spatial.distance.pdist(points, metric='cityblock')
        metrics['avg_pairwise_manhattan'] = np.mean(manhattan_distances)
        metrics['std_pairwise_manhattan'] = np.std(manhattan_distances)
        
        # Euclidean centroid distances (existing)
        centroid = np.mean(points, axis=0)
        distances_from_centroid = np.linalg.norm(points - centroid, axis=1)
        metrics['avg_centroid_distance'] = np.mean(distances_from_centroid)
        metrics['std_from_centroid'] = np.std(distances_from_centroid)
        metrics['max_from_centroid'] = np.max(distances_from_centroid)
        
        # Manhattan centroid distances (new)
        manhattan_from_centroid = np.sum(np.abs(points - centroid), axis=1)
        metrics['avg_centroid_manhattan'] = np.mean(manhattan_from_centroid)
        metrics['std_from_centroid_manhattan'] = np.std(manhattan_from_centroid)

        # 3) Convex hull volume (if we have enough points)
        if len(points) >= 8:
            # Attempt hull
            try:
                hull = ConvexHull(points)
                hull_failures['successful'] += 1
                metrics['hull_volume_per_point'] = hull.volume / len(points)
            except scipy.spatial._qhull.QhullError:
                # If hull fails, check dimensionality
                eigenvals = np.linalg.svd(points - np.mean(points, axis=0), compute_uv=False)
                effective_dims = sum(eigenvals > 1e-10)
                hull_failures['effective_dimensions'].append(effective_dims)
                hull_failures['coplanar_or_correlated'] += 1
                metrics['hull_volume_per_point'] = None
        else:
            hull_failures['insufficient_points'] += 1
            metrics['hull_volume_per_point'] = None
        
        genre_metrics[genre] = metrics

    # -------------------
    # OVERALL CONSISTENCY
    # -------------------
    all_metrics = {
        'avg_centroid_distances': [],
        'pairwise_distances': [],
        'hull_volumes': []
    }

    for g, m in genre_metrics.items():
        all_metrics['avg_centroid_distances'].append(m['avg_centroid_distance'])
        all_metrics['pairwise_distances'].append(m['avg_pairwise_distance'])
        if m['hull_volume_per_point'] is not None:
            all_metrics['hull_volumes'].append(m['hull_volume_per_point'])

    # If no data or all are empty
    if not all_metrics['avg_centroid_distances']:
        print("No valid data to analyze in analyze_genre_search_consistency().")
        return {}, {}, {}

    overall_metrics = {
        'mean_centroid_distance': np.mean(all_metrics['avg_centroid_distances']),
        'std_centroid_distance': np.std(all_metrics['avg_centroid_distances']),
        'mean_pairwise_distance': np.mean(all_metrics['pairwise_distances']),
        'std_pairwise_distance': np.std(all_metrics['pairwise_distances']),
        'coefficient_of_variation': np.std(all_metrics['avg_centroid_distances']) / 
                                     np.mean(all_metrics['avg_centroid_distances'])
    }


    # -----------------------------------------
    # COMPARE TO RANDOM BASELINE (centroid distance)
    # -----------------------------------------
    n_random_samples = 100
    n_points = 45 # typical number of points per genre
    random_metrics = []
    for _ in range(n_random_samples):
        sample_points = [generate_random_binned_point() for __ in range(n_points)]
        sample_points = np.array(sample_points)
        
        # Euclidean distances
        pairwise_dists = scipy.spatial.distance.pdist(sample_points)
        centroid = np.mean(sample_points, axis=0)
        dist_centroid = np.linalg.norm(sample_points - centroid, axis=1)
        
        # Manhattan distances
        manhattan_dists = scipy.spatial.distance.pdist(sample_points, metric='cityblock')
        manhattan_centroid = np.sum(np.abs(sample_points - centroid), axis=1)
        
        random_metrics.append({
            'avg_pairwise_distance': np.mean(pairwise_dists),
            'avg_centroid_distance': np.mean(dist_centroid),
            'avg_pairwise_manhattan': np.mean(manhattan_dists),
            'avg_centroid_manhattan': np.mean(manhattan_centroid)
        })

    random_baseline = {
        'avg_pairwise_distance': np.mean([rm['avg_pairwise_distance'] for rm in random_metrics]),
        'avg_centroid_distance': np.mean([rm['avg_centroid_distance'] for rm in random_metrics]),
        'avg_pairwise_manhattan': np.mean([rm['avg_pairwise_manhattan'] for rm in random_metrics]),
        'avg_centroid_manhattan': np.mean([rm['avg_centroid_manhattan'] for rm in random_metrics])
    }
    
    # Calculate improvements over random, genre by genre
    improvements = {}
    for genre, m in genre_metrics.items():
        actual_dist = m['avg_centroid_distance']
        improvements[genre] = (random_baseline['avg_centroid_distance'] - actual_dist) / \
                              random_baseline['avg_centroid_distance'] * 100

    avg_improvement = np.mean(list(improvements.values()))
    
    # --------------------------------
    # CHART #1: Improvement Over Random
    # --------------------------------
    plt.figure(figsize=(12, 6))
    genres_list = list(improvements.keys())
    improvement_vals = list(improvements.values())
    plt.bar(genres_list, improvement_vals, color='skyblue')
    plt.axhline(y=avg_improvement, color='r', linestyle='--',
                label=f'Mean Improvement: {avg_improvement:.1f}%')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Improvement over Random (%)')
    plt.title('Improvement over Random Baseline by Genre')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/consistency/improvement_over_random.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    # ------------------------------------------
    # More granular analysis (plots per metric)
    # ------------------------------------------
    sorted_genres = list(genre_metrics.keys())

    # After calculating random_baseline but before any plotting
    # Calculate statistical results first
    statistical_results = analyze_statistical_significance(genre_metrics, random_metrics, improvements)

    # Now start plotting
    # 1) Euclidean centroid distance plot
    plt.figure(figsize=(12, 6))
    genres_by_distance = sorted(sorted_genres, key=lambda g: genre_metrics[g]['avg_centroid_distance'])
    avg_distances = [genre_metrics[g]['avg_centroid_distance'] for g in genres_by_distance]
    std_distances = [genre_metrics[g]['std_from_centroid'] for g in genres_by_distance]
    mean_centroid_distance = np.mean(avg_distances)
    median_centroid_distance = np.median(avg_distances)

    # Add statistical significance info to title
    p_val = statistical_results['centroid_distances']['p_value']
    d_val = statistical_results['centroid_distances']['cohens_d']
    sig_stars = '*' * sum(p_val < threshold for threshold in [0.05, 0.01, 0.001])
    title = f'Centroid Distance Analysis {sig_stars}\n(p={p_val:.6f}, d={d_val:.2f}, effect: {interpret_cohens_d(d_val)})'

    plt.errorbar(range(len(genres_by_distance)), avg_distances, yerr=std_distances, 
                 fmt='o', capsize=5, capthick=1, elinewidth=1, label='Avg Dist (± std)')
    plt.axhline(y=mean_centroid_distance, color='r', linestyle='--', 
                label=f'Mean: {mean_centroid_distance:.3f}')
    plt.axhline(y=median_centroid_distance, color='orange', linestyle=':',
                label=f'Median: {median_centroid_distance:.3f}')
    plt.axhline(y=random_baseline['avg_centroid_distance'],
                color='g', linestyle='--',
                label=f'Random Baseline: {random_baseline["avg_centroid_distance"]:.3f}')

    plt.xticks(range(len(genres_by_distance)), genres_by_distance, rotation=45, ha='right')
    plt.ylabel('Average Distance from Centroid')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/consistency/genre_centroid_distance_euclidean.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 1) Manhattan Centroid Distance Plot
    plt.figure(figsize=(12, 6))
    genres_by_manhattan = sorted(sorted_genres, 
                               key=lambda g: genre_metrics[g]['avg_centroid_manhattan'])
    manhattan_distances = [genre_metrics[g]['avg_centroid_manhattan'] 
                         for g in genres_by_manhattan]
    manhattan_stds = [genre_metrics[g]['std_from_centroid_manhattan'] 
                     for g in genres_by_manhattan]
    mean_manhattan = np.mean(manhattan_distances)
    median_manhattan = np.median(manhattan_distances)

    # Calculate statistical significance for Manhattan metrics
    actual_manhattan = np.array([m['avg_centroid_manhattan'] for m in genre_metrics.values()])
    random_manhattan = np.array([r['avg_centroid_manhattan'] for r in random_metrics])
    
    u_stat_manhattan, p_val_manhattan = scipy.stats.mannwhitneyu(
        random_manhattan, actual_manhattan, alternative='greater'
    )
    
    pooled_var_manhattan = (np.var(random_manhattan) + np.var(actual_manhattan)) / 2
    cohens_d_manhattan = (np.mean(random_manhattan) - np.mean(actual_manhattan)) / \
                        np.sqrt(pooled_var_manhattan)

    sig_stars = '*' * sum(p_val_manhattan < threshold for threshold in [0.05, 0.01, 0.001])
    
    plt.errorbar(range(len(genres_by_manhattan)), manhattan_distances, 
                yerr=manhattan_stds, fmt='o', capsize=5, capthick=1, 
                elinewidth=1, label='Avg Manhattan Dist (± std)')
    plt.axhline(y=mean_manhattan, color='r', linestyle='--',
                label=f'Mean: {mean_manhattan:.3f}')
    plt.axhline(y=median_manhattan, color='orange', linestyle=':',
                label=f'Median: {median_manhattan:.3f}')
    plt.axhline(y=random_baseline['avg_centroid_manhattan'],
                color='g', linestyle='--',
                label=f'Random Baseline: {random_baseline["avg_centroid_manhattan"]:.3f}')

    plt.xticks(range(len(genres_by_manhattan)), genres_by_manhattan, 
               rotation=45, ha='right')
    plt.ylabel('Average Manhattan Distance from Centroid')
    plt.title(f'Manhattan Centroid Distance Analysis {sig_stars}\n'
              f'(p={p_val_manhattan:.6f}, d={cohens_d_manhattan:.2f}, '
              f'effect: {interpret_cohens_d(cohens_d_manhattan)})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/consistency/genre_centroid_distance_manhattan.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

    # 2) Euclidean Pairwise Distance Plot
    plt.figure(figsize=(12, 6))
    genres_by_pairwise = sorted(sorted_genres, key=lambda g: genre_metrics[g]['avg_pairwise_distance'])
    pairwise_vals = [genre_metrics[g]['avg_pairwise_distance'] for g in genres_by_pairwise]
    mean_pairwise = np.mean(pairwise_vals)
    median_pairwise = np.median(pairwise_vals)

    # Add statistical significance info to title
    p_val = statistical_results['pairwise_distances']['p_value']
    d_val = statistical_results['pairwise_distances']['cohens_d']
    sig_stars = '*' * sum(p_val < threshold for threshold in [0.05, 0.01, 0.001])
    title = f'Pairwise Distance Analysis {sig_stars}\n(p={p_val:.6f}, d={d_val:.2f}, effect: {interpret_cohens_d(d_val)})'

    plt.bar(range(len(genres_by_pairwise)), pairwise_vals, color='cadetblue')
    plt.axhline(y=mean_pairwise, color='r', linestyle='--',
                label=f'Mean: {mean_pairwise:.3f}')
    plt.axhline(y=median_pairwise, color='orange', linestyle=':',
                label=f'Median: {median_pairwise:.3f}')
    plt.axhline(y=random_baseline['avg_pairwise_distance'],
                color='g', linestyle='--',
                label=f'Random Baseline: {random_baseline["avg_pairwise_distance"]:.3f}')
    plt.xticks(range(len(genres_by_pairwise)), genres_by_pairwise, rotation=45, ha='right')
    plt.ylabel('Average Pairwise Distance')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/consistency/genre_pairwise_distance_euclidean.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 2) Manhattan Pairwise Distance Plot
    plt.figure(figsize=(12, 6))
    genres_by_pairwise_manhattan = sorted(sorted_genres, 
                                        key=lambda g: genre_metrics[g]['avg_pairwise_manhattan'])
    pairwise_manhattan = [genre_metrics[g]['avg_pairwise_manhattan'] 
                         for g in genres_by_pairwise_manhattan]
    mean_pairwise_manhattan = np.mean(pairwise_manhattan)
    median_pairwise_manhattan = np.median(pairwise_manhattan)

    # Statistical significance for pairwise Manhattan
    actual_pairwise_manhattan = np.array([m['avg_pairwise_manhattan'] 
                                        for m in genre_metrics.values()])
    random_pairwise_manhattan = np.array([r['avg_pairwise_manhattan'] 
                                        for r in random_metrics])
    
    u_stat_pairwise_manhattan, p_val_pairwise_manhattan = scipy.stats.mannwhitneyu(
        random_pairwise_manhattan, actual_pairwise_manhattan, alternative='greater'
    )
    
    pooled_var_pairwise_manhattan = (np.var(random_pairwise_manhattan) + 
                                    np.var(actual_pairwise_manhattan)) / 2
    cohens_d_pairwise_manhattan = (np.mean(random_pairwise_manhattan) - 
                                  np.mean(actual_pairwise_manhattan)) / \
                                 np.sqrt(pooled_var_pairwise_manhattan)

    sig_stars = '*' * sum(p_val_pairwise_manhattan < threshold 
                         for threshold in [0.05, 0.01, 0.001])

    plt.bar(range(len(genres_by_pairwise_manhattan)), pairwise_manhattan, 
            color='cadetblue')
    plt.axhline(y=mean_pairwise_manhattan, color='r', linestyle='--',
                label=f'Mean: {mean_pairwise_manhattan:.3f}')
    plt.axhline(y=median_pairwise_manhattan, color='orange', linestyle=':',
                label=f'Median: {median_pairwise_manhattan:.3f}')
    plt.axhline(y=random_baseline['avg_pairwise_manhattan'],
                color='g', linestyle='--',
                label=f'Random Baseline: {random_baseline["avg_pairwise_manhattan"]:.3f}')

    plt.xticks(range(len(genres_by_pairwise_manhattan)), 
               genres_by_pairwise_manhattan, rotation=45, ha='right')
    plt.ylabel('Average Pairwise Manhattan Distance')
    plt.title(f'Manhattan Pairwise Distance Analysis {sig_stars}\n'
              f'(p={p_val_pairwise_manhattan:.6f}, d={cohens_d_pairwise_manhattan:.2f}, '
              f'effect: {interpret_cohens_d(cohens_d_pairwise_manhattan)})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/consistency/genre_pairwise_distance_manhattan.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3) Hull volume analysis (only for genres that have it)
    hull_data = []
    for genre in sorted_genres:
        if genre_metrics[genre].get('hull_volume_per_point') is not None:
            hull_data.append((genre, genre_metrics[genre]['hull_volume_per_point']))

    if hull_data:
        plt.figure(figsize=(12, 6))
        hull_data_sorted = sorted(hull_data, key=lambda x: x[1])
        genres_by_hull = [x[0] for x in hull_data_sorted]
        hull_values = [x[1] for x in hull_data_sorted]
        mean_hull = np.mean(hull_values)
        median_hull = np.median(hull_values)

        plt.bar(range(len(genres_by_hull)), hull_values, color='lightseagreen')
        plt.axhline(y=mean_hull, color='r', linestyle='--', 
                    label=f'Mean: {mean_hull:.10f}')
        plt.axhline(y=median_hull, color='orange', linestyle=':',
                    label=f'Median: {median_hull:.10f}')
        plt.xticks(range(len(genres_by_hull)), genres_by_hull, rotation=45, ha='right')
        plt.ylabel('Hull Volume per Point')
        plt.title('Convex Hull Analysis (Sorted by Volume)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/consistency/genre_hull_analysis.png', bbox_inches='tight', dpi=300)
        plt.close()

    # 4) Search counts analysis: total vs unique
    plt.figure(figsize=(12, 6))
    search_count_data = []
    for genre in sorted_genres:
        search_c.execute('SELECT search_results FROM genre_search_distributions WHERE query_genre = ?', (genre,))
        row = search_c.fetchone()
        if not row:
            continue
        search_results = json.loads(row[0])
        total = len([s for s in search_results if 'is_average' not in s])
        unique_locations = set(tuple(s['location']) for s in search_results if 'is_average' not in s)
        unique_count = len(unique_locations)
        ratio = unique_count / total if total > 0 else 0
        search_count_data.append((genre, total, unique_count, ratio))

    search_count_data_sorted = sorted(search_count_data, key=lambda x: x[3])
    genres_by_ratio = [x[0] for x in search_count_data_sorted]
    search_counts = [x[1] for x in search_count_data_sorted]
    unique_counts = [x[2] for x in search_count_data_sorted]
    mean_unique_searches = np.mean(unique_counts)
    median_unique_searches = np.median(unique_counts)

    plt.bar(range(len(genres_by_ratio)), search_counts, label='Total Searches', color='cornflowerblue')
    plt.bar(range(len(genres_by_ratio)), unique_counts, label='Unique Searches', alpha=0.5, color='lightsteelblue')
    plt.axhline(y=mean_unique_searches, color='r', linestyle='--',
                label=f'Mean Unique: {mean_unique_searches:.1f}')
    plt.axhline(y=median_unique_searches, color='orange', linestyle=':',
                label=f'Median Unique: {median_unique_searches:.1f}')
    plt.xticks(range(len(genres_by_ratio)), genres_by_ratio, rotation=45, ha='right')
    plt.ylabel('Number of Searches')
    plt.title('Search Counts Analysis (Sorted by Unique/Total Ratio)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/consistency/genre_search_counts.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Save the genre metrics for further reference
    with open('results/consistency/genre_consistency_metrics.json', 'w') as f:
        json.dump(genre_metrics, f, indent=2)

   
    # ------------------------------------------------
    # VOLUME COVERAGE ANALYSIS (using spheres in N-D)
    # ------------------------------------------------
    n_dimensions = len(FEATURE_RANGES)  # e.g. 8
    total_space_volume = 1.0  # Because each dimension is [0, 1] if we assume bin-based approach

    def hypersphere_volume(radius, n_dim):
        """Calculate volume of an n-dimensional hypersphere given radius."""
        return (np.pi ** (n_dim / 2) * radius**n_dim) / scipy.special.gamma(n_dim / 2 + 1)
    
    genre_volumes = {}

    # Generate random volumes for baseline
    n_random_clusters = 100
    n_points_per_cluster = 47
    random_volumes = {
        'max': [],
        'mean': []
    }
    for _ in range(n_random_clusters):
        random_points = np.array([generate_random_binned_point() for __ in range(n_points_per_cluster)])
        random_centroid = np.mean(random_points, axis=0)
        random_distances = np.linalg.norm(random_points - random_centroid, axis=1)
        
        random_max_radius = np.max(random_distances)
        random_mean_radius = np.mean(random_distances)
        
        random_volumes['max'].append(hypersphere_volume(random_max_radius, n_dimensions))
        random_volumes['mean'].append(hypersphere_volume(random_mean_radius, n_dimensions))

    random_baseline_volumes = {
        'max_volume': np.mean(random_volumes['max']),
        'mean_volume': np.mean(random_volumes['mean']),
        'max_volume_std': np.std(random_volumes['max']),
        'mean_volume_std': np.std(random_volumes['mean'])
    }

    # Calculate coverage for each genre
    for g, m in genre_metrics.items():
        max_r = m['max_from_centroid']
        mean_r = m['avg_centroid_distance']

        sphere_vol_max = hypersphere_volume(max_r, n_dimensions)
        sphere_vol_mean = hypersphere_volume(mean_r, n_dimensions)

        genre_volumes[g] = {
            'max_radius': max_r,
            'mean_radius': mean_r,
            'volume_coverage_max': min(1.0, sphere_vol_max / total_space_volume) * 100,
            'volume_coverage_mean': min(1.0, sphere_vol_mean / total_space_volume) * 100,
            'volume_vs_random_max': sphere_vol_max / random_baseline_volumes['max_volume'],
            'volume_vs_random_mean': sphere_vol_mean / random_baseline_volumes['mean_volume']
        }

   

    # ----------------------------------
    # Split volume coverage into two plots
    # ----------------------------------
    # First define the genres list
    genres_vc = list(genre_volumes.keys())

    # Calculate significance stars for both metrics
    p_val_max = scipy.stats.ttest_1samp([genre_volumes[g]['volume_vs_random_max'] for g in genres_vc], 1.0)[1]
    p_val_mean = scipy.stats.ttest_1samp([genre_volumes[g]['volume_vs_random_mean'] for g in genres_vc], 1.0)[1]
    
    # Calculate Cohen's d for both metrics
    d_val_max = np.mean([genre_volumes[g]['volume_vs_random_max'] - 1.0 for g in genres_vc]) / \
                np.std([genre_volumes[g]['volume_vs_random_max'] for g in genres_vc])
    d_val_mean = np.mean([genre_volumes[g]['volume_vs_random_mean'] - 1.0 for g in genres_vc]) / \
                 np.std([genre_volumes[g]['volume_vs_random_mean'] for g in genres_vc])

    # Generate significance stars
    sig_stars_max = '*' * sum(p_val_max < threshold for threshold in [0.05, 0.01, 0.001])
    sig_stars_mean = '*' * sum(p_val_mean < threshold for threshold in [0.05, 0.01, 0.001])

    # 1. Max Radius Coverage Plot
    plt.figure(figsize=(12, 6))
    max_coverages = [genre_volumes[g]['volume_coverage_max'] for g in genres_vc]

    plt.yscale('log')
    plt.bar(genres_vc, max_coverages, color='skyblue', label='Max Radius Coverage')
    
    # Add statistics for max coverage
    actual_max_mean = np.mean(max_coverages)
    actual_max_median = np.median(max_coverages)
    
    plt.axhline(y=actual_max_mean, color='darkblue', linestyle='-.',
                label=f'Actual Mean: {actual_max_mean:.1e}%')
    plt.axhline(y=actual_max_median, color='blue', linestyle=':',
                label=f'Actual Median: {actual_max_median:.1e}%')
    
    # Random baseline for max
    plt.axhline(y=random_baseline_volumes['max_volume'] / total_space_volume * 100, color='red', linestyle='--',
                label=f'Random Coverage: {random_baseline_volumes["max_volume"] / total_space_volume * 100:.1e}%')
    plt.fill_between(range(len(genres_vc)), random_baseline_volumes['max_volume'] / total_space_volume * 100 - random_baseline_volumes['max_volume_std'] / total_space_volume * 100,
                     random_baseline_volumes['max_volume'] / total_space_volume * 100 + random_baseline_volumes['max_volume_std'] / total_space_volume * 100,
                     color='red', alpha=0.1)

    plt.xlabel('Genres')
    plt.ylabel('Volume Coverage (%) - Log Scale')
    plt.title(f'Maximum Radius Volume Coverage by Genre {sig_stars_max}\n'
              f'(p={p_val_max:.6f}, d={d_val_max:.2f}, effect: {interpret_cohens_d(d_val_max)})')
    plt.xticks(range(len(genres_vc)), genres_vc, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    
    # Set y-limits for max coverage plot
    min_coverage_max = min(min(max_coverages), random_baseline_volumes['max_volume'] / total_space_volume * 100 - random_baseline_volumes['max_volume_std'] / total_space_volume * 100)
    max_coverage_max = max(max(max_coverages), random_baseline_volumes['max_volume'] / total_space_volume * 100 + random_baseline_volumes['max_volume_std'] / total_space_volume * 100)
    plt.ylim([max(1e-6, min_coverage_max * 0.5), max_coverage_max * 2])

    plt.tight_layout()
    plt.savefig('results/consistency/volume_coverage_max.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Mean Radius Coverage Plot
    plt.figure(figsize=(12, 6))
    mean_coverages = [genre_volumes[g]['volume_coverage_mean'] for g in genres_vc]

    plt.yscale('log')
    plt.bar(genres_vc, mean_coverages, color='lightgreen', label='Mean Radius Coverage')
    
    # Add statistics for mean coverage
    actual_mean_mean = np.mean(mean_coverages)
    actual_mean_median = np.median(mean_coverages)
    
    plt.axhline(y=actual_mean_mean, color='darkgreen', linestyle='-.',
                label=f'Actual Mean: {actual_mean_mean:.1e}%')
    plt.axhline(y=actual_mean_median, color='green', linestyle=':',
                label=f'Actual Median: {actual_mean_median:.1e}%')
    
    # Random baseline for mean
    plt.axhline(y=random_baseline_volumes['mean_volume'] / total_space_volume * 100, color='red', linestyle='--',
                label=f'Random Coverage: {random_baseline_volumes["mean_volume"] / total_space_volume * 100:.1e}%')
    plt.fill_between(range(len(genres_vc)), random_baseline_volumes['mean_volume'] / total_space_volume * 100 - random_baseline_volumes['mean_volume_std'] / total_space_volume * 100,
                     random_baseline_volumes['mean_volume'] / total_space_volume * 100 + random_baseline_volumes['mean_volume_std'] / total_space_volume * 100,
                     color='red', alpha=0.1)

    plt.xlabel('Genres')
    plt.ylabel('Volume Coverage (%) - Log Scale')
    plt.title(f'Mean Radius Volume Coverage by Genre {sig_stars_mean}\n'
              f'(p={p_val_mean:.6f}, d={d_val_mean:.2f}, effect: {interpret_cohens_d(d_val_mean)})')
    plt.xticks(range(len(genres_vc)), genres_vc, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    
    # Set y-limits for mean coverage plot
    min_coverage_mean = min(min(mean_coverages), random_baseline_volumes['mean_volume'] / total_space_volume * 100 - random_baseline_volumes['mean_volume_std'] / total_space_volume * 100)
    max_coverage_mean = max(max(mean_coverages), random_baseline_volumes['mean_volume'] / total_space_volume * 100 + random_baseline_volumes['mean_volume_std'] / total_space_volume * 100)
    plt.ylim([max(1e-6, min_coverage_mean * 0.5), max_coverage_mean * 2])

    plt.tight_layout()
    plt.savefig('results/consistency/volume_coverage_mean.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --------------------------------------------
    # Second improvement plot with significance
    # --------------------------------------------
    # We'll plot the same improvement data, but add stats (p-value, effect size).
    plt.figure(figsize=(12, 6))
    improvement_vals = list(improvements.values())
    genres_improvements = list(improvements.keys())
    median_improvement = np.median(improvement_vals)
    mean_imp = np.mean(improvement_vals)

    bars = plt.bar(genres_improvements, improvement_vals, color='skyblue')
    plt.axhline(y=mean_imp, color='r', linestyle='--', 
                label=f'Mean: {mean_imp:.1f}%\n95% CI: '
                      f'[{statistical_results["improvements"]["ci_lower"]:.1f}%, {statistical_results["improvements"]["ci_upper"]:.1f}%]')
    plt.axhline(y=median_improvement, color='orange', linestyle=':', 
                label=f'Median: {median_improvement:.1f}%')

    # Add significance stars if p < 0.05
    p_val = statistical_results['improvements']['p_value']
    d_val = statistical_results['improvements']['cohens_d']
    if p_val < 0.05:
        # Count how many thresholds it beats
        star_count = sum(p_val < threshold for threshold in [0.05, 0.01, 0.001])
        stars = '*' * star_count
        plt.title(f'Improvement over Random Baseline by Genre {stars}\n'
                  f'(p={p_val:.6f}, d={d_val:.2f})')
    else:
        plt.title('Improvement over Random Baseline by Genre (not significant)')

    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Improvement over Random (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/consistency/improvement_over_random_stats.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    # Plot effective dimensions by genre
    if hull_failures['effective_dimensions']:
        plt.figure(figsize=(12, 6))
        
        # Collect effective dimensions by genre
        genre_dimensions = {}
        for genre, search_results_json in results:
            search_results = json.loads(search_results_json)
            points = []
            for search in search_results:
                if 'is_average' not in search:
                    actual_values = convert_location_to_values(search['location'], FEATURE_RANGES)
                    points.append(actual_values)
            
            if len(points) >= 8:  # Same threshold as used for hull calculation
                points = np.array(points)
                # Calculate effective dimensions using SVD
                eigenvals = np.linalg.svd(points - np.mean(points, axis=0), compute_uv=False)
                effective_dims = sum(eigenvals > 1e-10)
                genre_dimensions[genre] = effective_dims
        
        if genre_dimensions:
            # Sort genres by their effective dimensions
            sorted_genres = sorted(genre_dimensions.keys(), 
                                 key=lambda g: genre_dimensions[g])
            dimensions = [genre_dimensions[g] for g in sorted_genres]
            
            # Create bar plot
            plt.bar(range(len(sorted_genres)), dimensions, 
                   color='skyblue', edgecolor='black')
            
            # Add mean and median lines
            mean_dims = np.mean(dimensions)
            median_dims = np.median(dimensions)
            
            plt.axhline(y=mean_dims, color='r', linestyle='--', 
                       label=f'Mean: {mean_dims:.1f}')
            plt.axhline(y=median_dims, color='orange', linestyle=':', 
                       label=f'Median: {median_dims:.1f}')
            
            # Customize x-axis
            plt.xticks(range(len(sorted_genres)), sorted_genres, 
                      rotation=45, ha='right')
            
            # Add text box with statistics
            stats_text = (
                f"Total Genres: {len(genre_dimensions)}\n"
                f"Min Dimensions: {min(dimensions)}\n"
                f"Max Dimensions: {max(dimensions)}"
            )
            plt.text(0.95, 0.05, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.xlabel('Genre')
            plt.ylabel('Effective Dimensions')
            plt.title('Effective Dimensions by Genre')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='lower right')
            plt.tight_layout()
            
            plt.savefig('results/consistency/effective_dimensions.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

    search_conn.close()

    return genre_metrics, random_baseline, improvements


###############################################
# HELPER FUNCTION: STATISTICAL SIGNIFICANCE
###############################################

def analyze_statistical_significance(
    genre_metrics: Dict[str, Dict[str, float]],
    random_metrics: List[Dict[str, float]],
    improvements: Dict[str, float]
) -> Dict[str, Dict[str, Union[float, Tuple[float, float]]]]:
    """
    Perform statistical analysis on genre consistency metrics.

    Calculates:
    - T-tests for improvements over random
    - Mann-Whitney U tests for distances
    - Cohen's d effect sizes
    - Bootstrap confidence intervals

    Args:
        genre_metrics (dict): Metrics calculated for each genre
        random_metrics (list): List of metrics from random sampling
        improvements (dict): Dictionary of improvement percentages

    Returns:
        dict: Statistical test results including:
            - p-values
            - test statistics
            - effect sizes
            - confidence intervals
    """
    # Convert metrics to arrays
    actual_centroid_distances = np.array([m['avg_centroid_distance'] for m in genre_metrics.values()])
    actual_pairwise_distances = np.array([m['avg_pairwise_distance'] for m in genre_metrics.values()])
    random_centroid_distances = np.array([r['avg_centroid_distance'] for r in random_metrics])
    random_pairwise_distances = np.array([r['avg_pairwise_distance'] for r in random_metrics])

    # 1) One-sample t-test on improvements
    improvement_values = np.array(list(improvements.values()))
    t_stat_improvements, p_val_improvements = scipy.stats.ttest_1samp(improvement_values, 0)

    # 2) Mann-Whitney U test (non-parametric) comparing random vs actual
    u_stat_centroid, p_val_centroid = scipy.stats.mannwhitneyu(
        random_centroid_distances, actual_centroid_distances, alternative='greater'
    )
    u_stat_pairwise, p_val_pairwise = scipy.stats.mannwhitneyu(
        random_pairwise_distances, actual_pairwise_distances, alternative='greater'
    )

    # 3) Cohen's d for improvements (relative to 0)
    cohens_d_improvements = np.mean(improvement_values) / np.std(improvement_values)

    # Cohen's d for centroid distances
    # (pooled std dev approach)
    pooled_var_centroid = (np.var(random_centroid_distances) + np.var(actual_centroid_distances)) / 2
    cohens_d_centroid = (np.mean(random_centroid_distances) - np.mean(actual_centroid_distances)) / np.sqrt(pooled_var_centroid)

    # Cohen's d for pairwise distances
    pooled_var_pairwise = (np.var(random_pairwise_distances) + np.var(actual_pairwise_distances)) / 2
    cohens_d_pairwise = (np.mean(random_pairwise_distances) - np.mean(actual_pairwise_distances)) / np.sqrt(pooled_var_pairwise)

    # 4) Bootstrap confidence intervals for mean improvement
    n_bootstrap = 10000
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(improvement_values, size=len(improvement_values), replace=True)
        bootstrap_means.append(np.mean(sample))
    ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])

    # Compile results
    results = {
        'improvements': {
            't_statistic': t_stat_improvements,
            'p_value': p_val_improvements,
            'cohens_d': cohens_d_improvements,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        },
        'centroid_distances': {
            'u_statistic': u_stat_centroid,
            'p_value': p_val_centroid,
            'cohens_d': cohens_d_centroid
        },
        'pairwise_distances': {
            'u_statistic': u_stat_pairwise,
            'p_value': p_val_pairwise,
            'cohens_d': cohens_d_pairwise
        }
    }
    
    return results

def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size value.

    Args:
        d (float): Cohen's d value

    Returns:
        str: Interpretation as one of:
            - "negligible" (|d| < 0.2)
            - "small" (0.2 ≤ |d| < 0.5)
            - "medium" (0.5 ≤ |d| < 0.8)
            - "large" (|d| ≥ 0.8)
    """
    if abs(d) < 0.2: return "negligible"
    elif abs(d) < 0.5: return "small"
    elif abs(d) < 0.8: return "medium"
    else: return "large"


######################################################
# RANDOM 2D PROJECTIONS (instead of PCA) FOR VISUALS #
######################################################

def plot_random_genre_projections(
    significant_genres: Optional[Set[str]] = None,
    n_projections: int = 3
) -> None:
    """
    Create visualizations using random 2D projections of the feature space.

    Provides an alternative to PCA visualization by using random orthogonal
    projections, which can sometimes reveal different aspects of the data structure.

    Args:
        significant_genres (set, optional): Set of genre names to analyze.
            If None, analyzes all genres in the database.
        n_projections (int): Number of random projections to generate

    Saves:
        Multiple plot files to 'clusters/genre_clusters_random_projection_*.png'
        Each plot includes correlation with original distances.
    """
    search_conn = sqlite3.connect('spotify_search_distributions.db')
    search_c = search_conn.cursor()
    
    # Filter or get all data
    if significant_genres:
        placeholders = ','.join('?' * len(significant_genres))
        query = f'SELECT query_genre, search_results FROM genre_search_distributions WHERE query_genre IN ({placeholders})'
        results = search_c.execute(query, list(significant_genres)).fetchall()
    else:
        results = search_c.execute('SELECT query_genre, search_results FROM genre_search_distributions').fetchall()
    
    # Collect points
    locations = {}
    for genre, search_results_json in results:
        search_results = json.loads(search_results_json)
        for search in search_results:
            if 'is_average' not in search:
                if genre not in locations:
                    locations[genre] = []
                actual_values = convert_location_to_values(search['location'], FEATURE_RANGES)
                locations[genre].append(actual_values)
    
    X = []
    y = []
    for genre, locs in locations.items():
        X.extend(locs)
        y.extend([genre] * len(locs))
    
    X = np.array(X)
    if len(X) == 0:
        print("No data to project in plot_random_genre_projections.")
        return
    
    n_features = X.shape[1]
    unique_genres = list(set(y))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_genres)))

    # For each projection, create a separate figure
    for proj_idx in range(n_projections):
        plt.figure(figsize=(10, 8))
        
        # Random projection matrix
        random_matrix = np.random.randn(n_features, 2)
        # Normalize columns
        random_matrix = random_matrix / np.linalg.norm(random_matrix, axis=0)

        X_projected = X @ random_matrix
        
        # Plot each genre
        for genre, color in zip(unique_genres, colors):
            mask = [g == genre for g in y]
            points = X_projected[mask]
            plt.scatter(points[:, 0], points[:, 1], color=color, alpha=0.6, s=100)

            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    hull_points = np.vstack((hull_points, hull_points[0]))
                    plt.fill(hull_points[:, 0], hull_points[:, 1], color=color, alpha=0.2)
                    plt.plot(hull_points[:, 0], hull_points[:, 1], color=color, alpha=0.5)
                    center = points[hull.vertices].mean(axis=0)
                    plt.annotate(genre, (center[0], center[1]),
                                 fontsize=10, fontweight='bold',
                                 ha='center', va='center', color=color)
                except scipy.spatial._qhull.QhullError:
                    pass
        
        # Evaluate how well distances are preserved
        original_distances = scipy.spatial.distance.pdist(X)
        projected_distances = scipy.spatial.distance.pdist(X_projected)
        correlation = np.corrcoef(original_distances, projected_distances)[0, 1]

        plt.title(f'Random 2D Projection {proj_idx+1} (Distance Corr = {correlation:.3f})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        filename = f'results/clusters/genre_clusters_random_projection_{proj_idx+1}.png'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
    
    search_conn.close()


########################################################
# MAIN SCRIPT INVOCATION: RUNNING EVERYTHING SEQUENTIALLY
########################################################

if __name__ == '__main__':
    ensure_directories()
    
    # 1) Plot genre clusters (two PCA-based charts + midpoints)
    plot_genre_clusters()
    
    # 2) Analyze genre search consistency (multiple metrics) and produce charts
    analyze_genre_search_consistency()
    
    # 3) Plot random 2D projections for variety
    plot_random_genre_projections()
