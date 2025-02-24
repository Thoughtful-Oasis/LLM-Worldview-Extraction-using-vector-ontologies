"""
Accuracy Analysis Module (Quantitative)

This module analyzes the quantitative accuracy of Spotify genre searches by comparing
search results with actual genre distributions. It performs several types of analysis:

1. Genre Center Analysis:
   - Calculation of search result centers
   - Calculation of actual distribution centers
   - Comparison between search and distribution centers

2. Statistical Analysis:
   - Euclidean distance metrics
   - Cosine similarity metrics
   - Statistical significance testing
   - Effect size calculations

3. Visualization:
   - Three-panel comparison plots
   - Statistical analysis plots
   - Relative vs absolute distribution comparisons

Dependencies:
    - numpy
    - scipy
    - matplotlib
    - sqlite3
    - analyze_consistency (local module)

Typical usage:
    python analyze_accuracy_quantitative.py

Output:
    Generates multiple visualization files in the results/accuracy/ directory:
    - genre_center_comparisons_extended_relative.png
    - genre_center_comparisons_extended_absolute.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import sqlite3
from scipy.spatial import ConvexHull
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from analyze_consistency import (
    analyze_genre_search_consistency, 
    FEATURE_RANGES, 
    convert_location_to_values,
    generate_random_binned_point
)

def ensure_directories() -> None:
    """
    Create necessary subdirectories for storing analysis results.

    Creates the 'accuracy' subdirectory under 'results/' for storing
    quantitative accuracy analysis outputs.

    Example:
        >>> ensure_directories()
        >>> # Creates results/accuracy/ directory
    """
    base_dir = 'results'
    subdirs = [
        'accuracy',
    ]
    
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)


def normalize_coordinates(
    points: np.ndarray, 
    shift_to_center: bool = False
) -> np.ndarray:
    """
    Normalize coordinates, optionally shifting them to be centered around 0.
    
    Args:
        points: Array of points to normalize
        shift_to_center: If True, shift range from [0,1] to [-0.5,0.5]
    
    Returns:
        np.ndarray: Normalized points

    Example:
        >>> pts = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> normalized = normalize_coordinates(pts, shift_to_center=True)
        >>> print(normalized)  # Shows points shifted by -0.5
    """
    if shift_to_center:
        return points - 0.5
    return points

def calculate_cosine_similarity(
    vec1: np.ndarray, 
    vec2: np.ndarray
) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        float: Cosine similarity value between -1 and 1

    Example:
        >>> v1 = np.array([1, 0])
        >>> v2 = np.array([0, 1])
        >>> sim = calculate_cosine_similarity(v1, v2)
        >>> print(sim)  # Shows 0.0 (perpendicular vectors)
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_statistical_center_metrics_baseline(
    search_centers: Dict[str, np.ndarray],
    dist_centers: Dict[str, np.ndarray],
    n_samples: int = 1000,
    shift_to_center: bool = False
) -> Tuple[List[float], List[float]]:
    """
    Calculate baseline using statistical properties of genre and distribution centers.
    
    Args:
        search_centers: Dictionary of search centers by genre
        dist_centers: Dictionary of distribution centers by genre
        n_samples: Number of center pairs to generate
        shift_to_center: Whether to shift coordinates to [-0.5, 0.5] range

    Returns:
        tuple: (random_distances, random_similarities)
            - random_distances: List of Euclidean distances between random centers
            - random_similarities: List of cosine similarities between random centers

    Example:
        >>> distances, similarities = calculate_statistical_center_metrics_baseline(
        ...     search_centers, dist_centers)
        >>> print(f"Mean distance: {np.mean(distances)}")
    """
    # Calculate statistical properties
    search_points = np.array(list(search_centers.values()))
    dist_points = np.array(list(dist_centers.values()))
    
    search_mean = np.mean(search_points, axis=0)
    dist_mean = np.mean(dist_points, axis=0)
    search_std = np.std(search_points, axis=0)
    dist_std = np.std(dist_points, axis=0)
    
    random_distances = []
    random_similarities = []
    
    for _ in range(n_samples):
        random_search = np.random.normal(search_mean, search_std)
        random_dist = np.random.normal(dist_mean, dist_std)
        
        random_search = np.clip(random_search, 0, 1)
        random_dist = np.clip(random_dist, 0, 1)
        
        if shift_to_center:
            random_search = normalize_coordinates(random_search, True)
            random_dist = normalize_coordinates(random_dist, True)
        
        distance = np.linalg.norm(random_search - random_dist)
        similarity = calculate_cosine_similarity(random_search, random_dist)
        
        random_distances.append(distance)
        random_similarities.append(similarity)
    
    return random_distances, random_similarities

def calculate_genre_presence(
    distribution: Union[str, Dict[str, Dict[str, float]]], 
    target_genre: str
) -> float:
    """
    Calculate relative presence of a genre at a location.

    Computes the normalized percentage of a genre's presence relative to
    total percentage of all genres at that location.

    Args:
        distribution: Genre distribution data (JSON string or dict)
        target_genre: Genre to calculate presence for

    Returns:
        float: Normalized presence value between 0 and 1

    Example:
        >>> dist = {'rock': {'percentage': 60}, 'pop': {'percentage': 40}}
        >>> presence = calculate_genre_presence(dist, 'rock')
        >>> print(presence)  # Shows 0.6
    """
    if isinstance(distribution, str):
        distribution = json.loads(distribution)

    total_percentage = 0
    genre_percentage = 0
    for genre, data in distribution.items():
        total_percentage += data['percentage'] / 100
        if target_genre.lower() in genre.lower():
            genre_percentage += data['percentage'] / 100
    if total_percentage > 0:
        return genre_percentage/total_percentage
    else:
        return 0

def calculate_genre_rankings(
    search_points: Dict[str, List[np.ndarray]],
    dist_centers: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, List[float]]]:
    """
    Calculate rankings of true genre centers for each search point.
    
    Args:
        search_points: Dictionary mapping genres to lists of search result points
        dist_centers: Dictionary mapping genres to their distribution centers
    
    Returns:
        Dict mapping genres to their ranking statistics for both distance metrics
    """
    rankings = {}
    
    # Only process genres that exist in both search points and distribution centers
    common_genres = set(search_points.keys()) & set(dist_centers.keys())
    
    for query_genre in common_genres:
        rankings[query_genre] = {
            'euclidean': [],
            'manhattan': []
        }
        
        points = search_points[query_genre]
        
        # For each search point
        for point in points:
            # Calculate distances to all true centers
            euclidean_distances = {
                genre: np.linalg.norm(point - center)
                for genre, center in dist_centers.items()
            }
            manhattan_distances = {
                genre: np.sum(np.abs(point - center))
                for genre, center in dist_centers.items()
            }
            
            # Sort genres by distance and find rank of target genre
            euclidean_ranking = sorted(euclidean_distances.items(), key=lambda x: x[1])
            manhattan_ranking = sorted(manhattan_distances.items(), key=lambda x: x[1])
            
            # Find ranks (with safety check)
            euc_ranks = [i for i, (g, _) in enumerate(euclidean_ranking) if g == query_genre]
            man_ranks = [i for i, (g, _) in enumerate(manhattan_ranking) if g == query_genre]
            
            if euc_ranks and man_ranks:  # Only add ranks if genre was found
                rankings[query_genre]['euclidean'].append(euc_ranks[0] + 1)
                rankings[query_genre]['manhattan'].append(man_ranks[0] + 1)
    
    return rankings

def plot_ranking_analysis(
    rankings: Dict[str, Dict[str, List[float]]],
    n_genres: int,
    ax: plt.Axes,
    metric: str
) -> None:
    """
    Create ranking analysis plot for a specific distance metric.
    """
    # Calculate mean and std for each genre
    genres = []
    means = []
    stds = []
    
    for genre, data in rankings.items():
        genres.append(genre)
        means.append(np.mean(data[metric]))
        stds.append(np.std(data[metric]))
    
    # Calculate random baseline (mean rank would be n_genres/2)
    random_mean = n_genres / 2
    random_std = np.sqrt((n_genres ** 2 - 1) / 12)  # Standard deviation for uniform distribution
    
    # Sort by mean ranking
    sorted_idx = np.argsort(means)
    genres = np.array(genres)[sorted_idx]
    means = np.array(means)[sorted_idx]
    stds = np.array(stds)[sorted_idx]
    
    # Create the plot
    x = np.arange(len(genres))
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.axhline(y=random_mean, color='r', linestyle='--', 
               label=f'Random Mean: {random_mean:.1f} ± {random_std:.1f}')
    
    # Calculate statistics
    all_ranks = [rank for genre_data in rankings.values() for rank in genre_data[metric]]
    random_ranks = np.random.uniform(1, n_genres, size=len(all_ranks))
    
    t_stat, p_val = scipy.stats.ttest_ind(all_ranks, random_ranks)
    cohens_d = (np.mean(all_ranks) - random_mean) / np.sqrt(
        (np.var(all_ranks) + np.var(random_ranks)) / 2)
    
    ax.set_title(f'{metric.capitalize()} Distance Rankings\n'
                 f'p={p_val:.2e}, d={cohens_d:.2f}')
    ax.set_xticks(x)
    ax.set_xticklabels(genres, rotation=45, ha='right')
    ax.set_ylabel('Mean Rank')
    ax.legend()

def analyze_distribution_center_precision(
    search_points: Dict[str, List[np.ndarray]],
    dist_centers: Dict[str, np.ndarray],
    k: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Analyze precision@k for each distribution center by finding closest search points.
    
    Args:
        search_points: Dictionary mapping genres to lists of search result points
        dist_centers: Dictionary mapping genres to their distribution centers
        k: Number of nearest neighbors to consider
    
    Returns:
        Dict mapping genres to their precision metrics for both distance metrics
    """
    results = {}
    
    # Flatten all search points into a single list with their genres
    all_points = []
    for genre, points in search_points.items():
        for point in points:
            all_points.append((point, genre))
    
    for target_genre, dist_center in dist_centers.items():
        results[target_genre] = {
            'euclidean': 0.0,
            'manhattan': 0.0
        }
        
        if not all_points:  # Skip if no search points available
            continue
        
        # Calculate distances using both metrics
        euclidean_distances = [
            (np.linalg.norm(point - dist_center), genre)
            for point, genre in all_points
        ]
        manhattan_distances = [
            (np.sum(np.abs(point - dist_center)), genre)
            for point, genre in all_points
        ]
        
        # Sort by distance and get top k
        euclidean_nearest = sorted(euclidean_distances, key=lambda x: x[0])[:k]
        manhattan_nearest = sorted(manhattan_distances, key=lambda x: x[0])[:k]
        
        # Calculate precision for each metric
        results[target_genre]['euclidean'] = sum(
            1 for _, genre in euclidean_nearest if genre == target_genre
        ) / k
        
        results[target_genre]['manhattan'] = sum(
            1 for _, genre in manhattan_nearest if genre == target_genre
        ) / k
    
    return results

def plot_precision_analysis(
    precision_results: Dict[str, Dict[str, float]],
    ax: plt.Axes,
    metric: str
) -> None:
    """
    Create precision analysis plot for a specific distance metric.
    """
    genres = []
    precisions = []
    
    # Extract data
    for genre, metrics in precision_results.items():
        genres.append(genre)
        precisions.append(metrics[metric])
    
    # Sort by precision
    sorted_idx = np.argsort(precisions)[::-1]
    genres = np.array(genres)[sorted_idx]
    precisions = np.array(precisions)[sorted_idx]
    
    # Calculate random baseline (1/n_genres)
    random_precision = 1.0 / len(genres)
    
    # Create the plot
    x = np.arange(len(genres))
    ax.bar(x, precisions)
    ax.axhline(y=random_precision, color='r', linestyle='--',
               label=f'Random Baseline: {random_precision:.3f}')
    
    # Calculate statistics
    t_stat, p_val = scipy.stats.ttest_1samp(precisions, random_precision)
    cohens_d = (np.mean(precisions) - random_precision) / np.std(precisions)
    
    ax.set_title(f'{metric.capitalize()} Distance Precision@10\n'
                 f'p={p_val:.2e}, d={cohens_d:.2f}')
    ax.set_xticks(x)
    ax.set_xticklabels(genres, rotation=45, ha='right')
    ax.set_ylabel('Precision@10')
    ax.legend()

def analyze_genre_distribution_vs_consistency(
    significant_genres: Optional[Set[str]] = None
) -> None:
    """
    Analyze correlation between genre search consistency and actual genre distribution.

    Creates extended genre center comparison visualizations comparing search results
    with actual genre distributions, including statistical analysis.

    Args:
        significant_genres: Optional set of genres to analyze. If None, analyzes all genres.

    Outputs:
        Saves visualization files to results/accuracy/:
        - genre_center_comparisons_extended_relative.png
        - genre_center_comparisons_extended_absolute.png

    Example:
        >>> analyze_genre_distribution_vs_consistency({'rock', 'jazz'})
        >>> # Creates comparison visualizations for rock and jazz genres
    """
    # Ensure output directory exists
    os.makedirs('results/accuracy', exist_ok=True)
    
    # Connect to databases
    search_conn = sqlite3.connect('spotify_search_distributions.db')
    dist_conn = sqlite3.connect('spotify_genre_distributions.db')
    search_c = search_conn.cursor()
    dist_c = dist_conn.cursor()
    
    # Get all genres from search database
    search_c.execute('SELECT DISTINCT query_genre FROM genre_search_distributions')
    genres = {row[0] for row in search_c.fetchall()}
    
    # Filter genres if significant_genres is provided
    if significant_genres:
        genres = genres & significant_genres
    print(f"Processing {len(genres)} genres")
    
    # Get all location distributions from distribution database
    print("Fetching location distributions...")
    dist_c.execute('SELECT location, songs_found, distribution FROM location_distributions')
    location_results = dist_c.fetchall()
    print(f"Found {len(location_results)} location distributions")
    
    # Get search results and calculate centers
    search_centers = {}
    search_points = {}  # Add this dictionary to store all points
    for row in search_conn.execute('SELECT query_genre, search_results FROM genre_search_distributions'):
        genre, search_results_json = row
        if significant_genres and genre not in significant_genres:
            continue
            
        search_results = json.loads(search_results_json)
        points = []
        for search in search_results:
            if 'is_average' not in search:
                points.append(convert_location_to_values(search['location'], FEATURE_RANGES))
        
        if len(points) >= 2:
            search_points[genre] = points  # Store all points
            search_centers[genre] = np.mean(points, axis=0)
    
    # Get distribution data and calculate centers
    distribution_centers = {}
    distribution_centers_absolute = {}
    
    for location, songs_found, distribution_json in location_results:
        actual_values = convert_location_to_values(location, FEATURE_RANGES)
        
        for genre in genres:
            presence = calculate_genre_presence(distribution_json, genre)
            if presence > 0:
                if genre not in distribution_centers:
                    distribution_centers[genre] = []
                    distribution_centers_absolute[genre] = []
                
                distribution_centers[genre].append((actual_values, presence))
                distribution_centers_absolute[genre].append((actual_values, presence * songs_found))
    
    # Calculate weighted centers
    for genre in list(distribution_centers.keys()):
        if len(distribution_centers[genre]) >= 2:
            points, weights = zip(*distribution_centers[genre])
            distribution_centers[genre] = np.average(points, weights=weights, axis=0)
            
            points, weights = zip(*distribution_centers_absolute[genre])
            distribution_centers_absolute[genre] = np.average(points, weights=weights, axis=0)
        else:
            del distribution_centers[genre]
            del distribution_centers_absolute[genre]
    
    # Generate visualizations for both relative and absolute distributions
    for distribution_type, centers in [("Relative", distribution_centers), 
                                     ("Absolute", distribution_centers_absolute)]:
        common_genres = set(search_centers.keys()) & set(centers.keys())
        if not common_genres:
            continue
            
        # Calculate metrics and create visualization
        center_distances = []
        cosine_similarities = []
        cosine_similarities_shifted = []
        genres_list = []
        
        for genre in common_genres:
            search_center = search_centers[genre]
            dist_center = centers[genre]
            
            distance = np.linalg.norm(search_center - dist_center)
            cosine_sim = calculate_cosine_similarity(search_center, dist_center)
            
            search_center_shifted = normalize_coordinates(search_center, True)
            dist_center_shifted = normalize_coordinates(dist_center, True)
            cosine_sim_shifted = calculate_cosine_similarity(search_center_shifted, dist_center_shifted)
            
            center_distances.append(distance)
            cosine_similarities.append(cosine_sim)
            cosine_similarities_shifted.append(cosine_sim_shifted)
            genres_list.append(genre)
        
    
        # Calculate random baselines and create visualization
        random_distances, random_similarities = calculate_statistical_center_metrics_baseline(
            search_centers, centers, n_samples=1000, shift_to_center=False)
        _, random_similarities_shifted = calculate_statistical_center_metrics_baseline(
            search_centers, centers, n_samples=1000, shift_to_center=True)
        
        # Create the three-panel visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))
        
        # Plot distances with statistical analysis
        sorted_idx = np.argsort(center_distances)
        ax1.bar(range(len(genres_list)), np.array(center_distances)[sorted_idx])
        ax1.axhline(y=np.mean(center_distances), color='g', linestyle='-', 
                    label=f'Mean: {np.mean(center_distances):.3f}')
        ax1.axhline(y=np.mean(random_distances), color='r', linestyle='--',
                    label=f'Random: {np.mean(random_distances):.3f}')
        
        # Calculate statistics for distances
        t_stat_dist, p_val_dist = scipy.stats.ttest_ind(center_distances, random_distances)
        cohens_d_dist = (np.mean(center_distances) - np.mean(random_distances)) / np.sqrt(
            (np.var(center_distances) + np.var(random_distances)) / 2)
        
        ax1.set_title(f'Euclidean Distances\np={p_val_dist:.2e}, d={cohens_d_dist:.2f}')
        ax1.set_xticks(range(len(genres_list)))
        ax1.set_xticklabels(np.array(genres_list)[sorted_idx], rotation=45, ha='right')
        ax1.legend()
        
        # Plot cosine similarities with statistical analysis
        for ax, similarities, random_sims, title in [
            (ax2, cosine_similarities, random_similarities, 'Cosine Similarities [0,1]'),
            (ax3, cosine_similarities_shifted, random_similarities_shifted, 'Cosine Similarities [-0.5,0.5]')
        ]:
            sorted_idx = np.argsort(similarities)[::-1]
            ax.bar(range(len(genres_list)), np.array(similarities)[sorted_idx])
            ax.axhline(y=np.mean(similarities), color='g', linestyle='-',
                      label=f'Mean: {np.mean(similarities):.3f}')
            ax.axhline(y=np.mean(random_sims), color='r', linestyle='--',
                      label=f'Random: {np.mean(random_sims):.3f}')
            
            # Calculate statistics
            t_stat, p_val = scipy.stats.ttest_ind(similarities, random_sims)
            cohens_d = (np.mean(similarities) - np.mean(random_sims)) / np.sqrt(
                (np.var(similarities) + np.var(random_sims)) / 2)
            
            ax.set_xticks(range(len(genres_list)))
            ax.set_xticklabels(np.array(genres_list)[sorted_idx], rotation=45, ha='right')
            ax.set_title(f'{title}\np={p_val:.2e}, d={cohens_d:.2f}')
            ax.legend()
        
        plt.suptitle(f'{distribution_type} Distribution Analysis', y=1.05, fontsize=16)
        plt.tight_layout()
        plt.savefig(
            f'results/accuracy/genre_center_comparisons_extended_{distribution_type.lower()}.png',
            bbox_inches='tight', dpi=300
        )
        plt.close()
    
    # Add ranking analysis
    rankings = calculate_genre_rankings(search_points, distribution_centers)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    plot_ranking_analysis(rankings, len(distribution_centers), ax1, 'euclidean')
    plot_ranking_analysis(rankings, len(distribution_centers), ax2, 'manhattan')
    
    plt.suptitle('Genre Ranking Analysis', y=1.05, fontsize=16)
    plt.tight_layout()
    plt.savefig('results/accuracy/genre_rankings.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # After the ranking analysis, add precision analysis
    precision_results = analyze_distribution_center_precision(search_points, distribution_centers)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    plot_precision_analysis(precision_results, ax1, 'euclidean')
    plot_precision_analysis(precision_results, ax2, 'manhattan')
    
    plt.suptitle('Genre Precision@10 Analysis', y=1.05, fontsize=16)
    plt.tight_layout()
    plt.savefig('results/accuracy/genre_precision.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    search_conn.close()
    dist_conn.close()

if __name__ == '__main__':
    """
    Main execution function for quantitative accuracy analysis.

    Performs the complete analysis pipeline:
    1. Creates necessary directories
    2. Analyzes genre distribution vs consistency
    3. Generates statistical comparisons and visualizations

    Example:
        >>> python analyze_accuracy_quantitative.py
    """
    analyze_genre_distribution_vs_consistency()
