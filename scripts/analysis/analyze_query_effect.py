"""
Query Effect Analysis Module

This module analyzes the effect of different query patterns on search results in the Spotify dataset.
It examines how different query formulations affect the positioning of results in the feature space
relative to genre centroids.

The analysis includes:
- Loading and organizing query data by pattern
- Calculating and analyzing vectors between genre centroids and query results
- Computing similarities between query vectors
- Generating visualizations of query effects
- Statistical comparison against random baselines

Key visualizations produced:
- Distribution of vector similarities for each query pattern
- Nearest neighbor similarity distributions
- Mean similarities compared to random baseline
- Statistical significance analysis
- PCA visualizations of query effects as vectors in feature space

Dependencies:
    - numpy
    - scipy
    - matplotlib
    - sklearn
    - sqlite3
    - seaborn
    - analyze_consistency (local module)

Typical usage:
    python query_effect_analysis.py

Output:
    Generates multiple visualization files in the results/query_analysis directory:
    - query_similarity_distribution.png
    - query_similarity_distribution_nearest_neighbors.png
    - query_similarity_means.png
    - query_nearest_neighbor_means.png
    - query_template_*.png (multiple files, one per query pattern)
"""

import sqlite3
import json
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import random
from analyze_consistency import convert_location_to_values, FEATURE_RANGES
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
import os
import scipy.stats

def ensure_directories() -> None:
    """
    Create all necessary subdirectories for storing analysis results.

    Creates the following directory structure under 'results/':
        - query_analysis/ for query effect analysis results
    """
    base_dir = 'results'
    subdirs = [
        'query_analysis',
    ]
    
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)


def load_query_data() -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, np.ndarray]]:
    """
    Load and organize query data from the SQLite database by query pattern.

    This function:
    1. Connects to the Spotify search distributions database
    2. Loads all search results
    3. Calculates genre centroids in feature space
    4. Organizes results by query pattern template

    Returns:
        tuple: (query_data, genre_centroids)
            - query_data: dict mapping query patterns to lists of (genre, location) pairs
            - genre_centroids: dict mapping genres to their centroid coordinates in feature space

    Example:
        >>> query_data, genre_centroids = load_query_data()
        >>> print(len(query_data))  # Number of unique query patterns
        >>> print(len(genre_centroids))  # Number of genres
    """
    conn = sqlite3.connect('spotify_search_distributions.db')
    c = conn.cursor()
    
    # Get all search results
    results = c.execute('SELECT query_genre, search_results FROM genre_search_distributions').fetchall()
    
    # Organize data by query pattern
    query_data = {}
    genre_centroids = {}
    
    for genre, search_results_json in results:
        searches = json.loads(search_results_json)
        
        # Calculate genre centroid
        points = []
        for search in searches:
            if 'is_average' not in search:
                points.append(convert_location_to_values(search['location'], FEATURE_RANGES))
        genre_centroids[genre] = np.mean(points, axis=0)
        
        # Organize by query pattern
        for search in searches:
            if 'is_average' not in search:
                # Extract query template by replacing the genre with a placeholder
                query = search['query']
                query_template = query.replace(genre, "{genre}")
                
                if query_template not in query_data:
                    query_data[query_template] = []
                query_data[query_template].append({
                    'genre': genre,
                    'location': convert_location_to_values(search['location'], FEATURE_RANGES),
                    'centroid': genre_centroids[genre]
                })
    
    conn.close()
    return query_data, genre_centroids

def calculate_query_vectors(
    query_data: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Calculate vectors from genre centroids to query result points.

    For each query pattern and result, computes the vector from the genre's
    centroid to the query result location in feature space. This helps analyze
    how different query patterns systematically shift results.

    Args:
        query_data: Mapping of query patterns to lists of points with genre and location data

    Returns:
        dict: Mapping of query patterns to lists of vector data containing:
            - genre: The genre label
            - vector: The displacement vector from centroid to result
            - start: The centroid coordinates
            - end: The result coordinates

    Example:
        >>> query_vectors = calculate_query_vectors(query_data)
        >>> for pattern, vectors in query_vectors.items():
        >>>     print(f"{pattern}: {len(vectors)} vectors")
    """
    query_vectors = {}
    
    for query, points in query_data.items():
        vectors = []
        for point in points:
            # Vector from centroid to query point
            vector = np.array(point['location']) - np.array(point['centroid'])
            vectors.append({
                'genre': point['genre'],
                'vector': vector,
                'start': point['centroid'],
                'end': point['location']
            })
        query_vectors[query] = vectors
    
    return query_vectors

def calculate_vector_similarities(
    query_vectors: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, Union[np.ndarray, float, List[float]]]]:
    """
    Calculate similarity metrics between vectors for each query pattern.

    Computes pairwise cosine similarities between vectors for each query pattern
    to measure consistency of query effects across different genres.

    Args:
        query_vectors: Mapping of query patterns to vector data

    Returns:
        dict: Mapping of query patterns to similarity statistics including:
            - matrix: Pairwise similarity matrix
            - mean: Mean similarity
            - std: Standard deviation of similarities
            - nearest_neighbors: List of highest similarities for each vector
            - nearest_mean: Mean of nearest neighbor similarities
            - nearest_median: Median of nearest neighbor similarities
            - nearest_std: Standard deviation of nearest neighbor similarities

    Example:
        >>> similarities = calculate_vector_similarities(query_vectors)
        >>> print(f"Mean similarity: {similarities['pattern']['mean']}")
    """
    similarities = {}
    for query, vectors in query_vectors.items():
        if len(vectors) < 2:
            continue
            
        # Extract vectors
        vector_array = np.array([v['end'] - v['start'] for v in vectors])
        
        # Calculate similarity matrix
        sim_matrix = np.zeros((len(vector_array), len(vector_array)))
        nearest_neighbors = []
        
        for i in range(len(vector_array)):
            sims = []
            for j in range(len(vector_array)):
                if i != j:
                    sim = 1 - cosine(vector_array[i], vector_array[j])
                    sim_matrix[i, j] = sim
                    sims.append(sim)
            if sims:  # Get the highest similarity (nearest neighbor)
                nearest_neighbors.append(max(sims))
        
        # Store results
        similarities[query] = {
            'matrix': sim_matrix,
            'mean': np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)]),
            'std': np.std(sim_matrix[np.triu_indices_from(sim_matrix, k=1)]),
            'nearest_neighbors': nearest_neighbors,
            'nearest_mean': np.mean(nearest_neighbors),
            'nearest_median': np.median(nearest_neighbors),
            'nearest_std': np.std(nearest_neighbors)
        }
    
    return similarities

def plot_similarity_distribution(
    similarities: Dict[str, Dict[str, Union[np.ndarray, float, List[float]]]]
) -> None:
    """
    Create violin plots showing the distribution of vector similarities for each query pattern.

    Visualizes the full distribution of pairwise similarities between query effect vectors,
    helping identify which query patterns have consistent effects across genres.

    Args:
        similarities: Query pattern similarity data from calculate_vector_similarities()

    Outputs:
        Saves plot to results/query_analysis/query_similarity_distribution.png

    Example:
        >>> plot_similarity_distribution(similarities)
    """
    output_dir = 'results/query_analysis'
    plt.figure(figsize=(20, 10))
    
    # Prepare data
    data = []
    labels = []
    
    for query, sim_data in similarities.items():
        # Get upper triangle values
        sim_values = sim_data['matrix'][np.triu_indices_from(sim_data['matrix'], k=1)]
        data.append(sim_values)
        labels.append(query[:50] + '...' if len(query) > 50 else query)
    
    # Create violin plot
    plt.violinplot(data, showmeans=True)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
    plt.ylabel('Cosine Similarity')
    plt.title('Distribution of Vector Similarities by Query Pattern')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/query_similarity_distribution.png',
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_nearest_neighbor_similarities(
    similarities: Dict[str, Dict[str, Union[np.ndarray, float, List[float]]]]
) -> None:
    """
    Create a violin plot showing the distribution of similarities for each query,
    considering only the 5 nearest neighbors for each vector.

    Args:
        similarities: Query pattern similarity data from calculate_vector_similarities()

    Outputs:
        Saves plot to results/query_analysis/query_similarity_distribution_nearest_neighbors.png

    Example:
        >>> plot_nearest_neighbor_similarities(similarities)
    """
    output_dir = 'results/query_analysis'
    plt.figure(figsize=(20, 10))
    
    # Prepare data
    data = []
    labels = []
    
    for query, sim_data in similarities.items():
        sim_matrix = sim_data['matrix']
        n_vectors = len(sim_matrix)
        
        if n_vectors < 2:
            continue
            
        # For each vector, get its 5 highest similarities (excluding self)
        nearest_neighbor_sims = []
        for i in range(n_vectors):
            # Get similarities for this vector (excluding self-similarity)
            vector_sims = np.concatenate([sim_matrix[i, :i], sim_matrix[i, i+1:]])
            # Take top 5 (or less if there aren't 5 neighbors)
            k = min(5, len(vector_sims))
            top_k_sims = np.partition(vector_sims, -k)[-k:]
            nearest_neighbor_sims.extend(top_k_sims)
        
        data.append(nearest_neighbor_sims)
        labels.append(query[:50] + '...' if len(query) > 50 else query)
    
    # Create violin plot
    plt.violinplot(data, showmeans=True)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
    plt.ylabel('Cosine Similarity (5 Nearest Neighbors)')
    plt.title('Distribution of Vector Similarities by Query Pattern (5 Nearest Neighbors)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/query_similarity_distribution_nearest_neighbors.png',
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_similarity_means(
    similarities: Dict[str, Dict[str, Union[np.ndarray, float, List[float]]]],
    random_baseline: np.ndarray
) -> Tuple[float, float]:
    """
    Create a bar plot showing mean and std of similarities for each query pattern,
    with overall statistics, random baseline comparison, and statistical significance.

    Args:
        similarities: Query pattern similarity data
        random_baseline: Array of similarity values from random vector pairs

    Returns:
        tuple: (overall_mean, overall_median) of similarities across all patterns

    Outputs:
        Saves plot to results/query_analysis/query_similarity_means.png

    Example:
        >>> mean, median = plot_similarity_means(similarities, random_baseline)
        >>> print(f"Overall mean: {mean}, median: {median}")
    """
    output_dir = 'results/query_analysis'
    plt.figure(figsize=(20, 10))
    
    # Prepare data
    queries = list(similarities.keys())
    means = [sim_data['mean'] for sim_data in similarities.values()]
    stds = [sim_data['std'] for sim_data in similarities.values()]
    
    # Calculate overall statistics
    overall_mean = np.mean(means)
    overall_median = np.median(means)
    random_mean = np.mean(random_baseline)
    random_std = np.std(random_baseline)
    
    # Calculate statistical tests
    p_value, cohens_d = calculate_statistical_tests(similarities, random_baseline)
    
    # Create bar plot
    x = range(len(queries))
    plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7, label='Query Patterns')
    
    # Add overall statistics lines
    plt.axhline(y=overall_mean, color='r', linestyle='--', 
                label=f'Overall Mean: {overall_mean:.3f}')
    plt.axhline(y=overall_median, color='g', linestyle='--', 
                label=f'Overall Median: {overall_median:.3f}')
    plt.axhline(y=random_mean, color='gray', linestyle=':', 
                label=f'Random Baseline: {random_mean:.3f} ± {random_std:.3f}')
    
    # Add shaded area for random baseline standard deviation
    plt.axhspan(random_mean - random_std, random_mean + random_std, 
                color='gray', alpha=0.1)
    
    # Add statistical information to the plot
    stat_text = f'p-value: {p_value:.2e}\nCohen\'s d: {cohens_d:.2f}'
    plt.text(0.02, 0.98, stat_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xticks(x, [q[:50] + '...' if len(q) > 50 else q for q in queries], rotation=45, ha='right')
    plt.ylabel('Mean Cosine Similarity')
    plt.title('Mean Vector Similarities by Query Pattern\nwith Random Baseline Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/query_similarity_means.png',
                bbox_inches='tight', dpi=300)
    plt.close()
    
    return overall_mean, overall_median

def generate_random_baseline(
    query_vectors: Dict[str, List[Dict[str, Any]]],
    n_iterations: int = 1000
) -> np.ndarray:
    """
    Generate random baseline similarities for statistical comparison.

    Creates random vectors within the same feature space bounds as the actual query vectors
    to establish a null hypothesis distribution for similarity values.

    Args:
        query_vectors: The actual query vector data
        n_iterations: Number of random iterations to perform

    Returns:
        np.array: Array of similarity values from random vector pairs

    Example:
        >>> random_baseline = generate_random_baseline(query_vectors, n_iterations=1000)
        >>> print(f"Mean random similarity: {np.mean(random_baseline)}")
    """
    # Calculate average number of vectors per query
    avg_vectors = np.mean([len(vectors) for vectors in query_vectors.values()])
    n_vectors = int(avg_vectors)
    
    # Get feature space bounds from existing vectors
    all_starts = []
    all_ends = []
    for vectors in query_vectors.values():
        for v in vectors:
            all_starts.append(v['start'])
            all_ends.append(v['end'])
    
    start_mins = np.min(all_starts, axis=0)
    start_maxs = np.max(all_starts, axis=0)
    end_mins = np.min(all_ends, axis=0)
    end_maxs = np.max(all_ends, axis=0)
    
    # Generate random similarities
    random_sims = []
    for _ in range(n_iterations):
        # Generate random start and end points
        random_starts = np.random.uniform(start_mins, start_maxs, (n_vectors, len(start_mins)))
        random_ends = np.random.uniform(end_mins, end_maxs, (n_vectors, len(end_mins)))
        
        # Calculate vectors
        random_vectors = random_ends - random_starts
        
        # Calculate pairwise similarities
        sim_matrix = np.zeros((n_vectors, n_vectors))
        for i in range(n_vectors):
            for j in range(i + 1, n_vectors):
                sim = 1 - cosine(random_vectors[i], random_vectors[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        
        # Get upper triangle values
        sim_values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        random_sims.extend(sim_values)
    
    return np.array(random_sims)

def calculate_statistical_tests(
    similarities: Dict[str, Dict[str, Union[np.ndarray, float, List[float]]]],
    random_baseline: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate statistical significance of query vector similarities.

    Performs t-test and calculates Cohen's d effect size comparing actual
    similarities to random baseline distribution.

    Args:
        similarities: Query pattern similarity data
        random_baseline: Random baseline similarities

    Returns:
        tuple: (p_value, cohens_d) statistical test results

    Example:
        >>> p_value, cohens_d = calculate_statistical_tests(similarities, random_baseline)
        >>> print(f"p-value: {p_value}, Cohen's d: {cohens_d}")
    """
    # Collect all actual similarities
    actual_sims = []
    for sim_data in similarities.values():
        sim_matrix = sim_data['matrix']
        actual_sims.extend(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
    actual_sims = np.array(actual_sims)
    
    # Calculate p-value (using t-test)
    t_stat, p_value = scipy.stats.ttest_ind(actual_sims, random_baseline)
    
    # Calculate Cohen's d
    pooled_std = np.sqrt((np.var(actual_sims) + np.var(random_baseline)) / 2)
    cohens_d = (np.mean(actual_sims) - np.mean(random_baseline)) / pooled_std
    
    return p_value, cohens_d

def plot_nearest_neighbor_means(
    similarities: Dict[str, Dict[str, Union[np.ndarray, float, List[float]]]],
    random_baseline: np.ndarray
) -> Tuple[float, float]:
    """
    Create a bar plot showing mean and std of nearest neighbor similarities for each query pattern,
    with overall statistics, random baseline comparison, and statistical significance.

    Args:
        similarities: Query pattern similarity data
        random_baseline: Random baseline similarities

    Returns:
        tuple: (overall_mean, overall_median) of nearest neighbor similarities

    Outputs:
        Saves plot to results/query_analysis/query_nearest_neighbor_means.png

    Example:
        >>> mean, median = plot_nearest_neighbor_means(similarities, random_baseline)
        >>> print(f"Overall mean: {mean}, median: {median}")
    """
    output_dir = 'results/query_analysis'
    plt.figure(figsize=(20, 10))
    
    # Prepare data
    queries = list(similarities.keys())
    means = [sim_data['nearest_mean'] for sim_data in similarities.values()]
    stds = [sim_data['nearest_std'] for sim_data in similarities.values()]
    medians = [sim_data['nearest_median'] for sim_data in similarities.values()]
    
    # Calculate overall statistics
    overall_mean = np.mean(means)
    overall_median = np.median(means)
    random_mean = np.mean(random_baseline)
    random_std = np.std(random_baseline)
    
    # Calculate statistical tests for nearest neighbor similarities
    nearest_sims = []
    for sim_data in similarities.values():
        nearest_sims.extend(sim_data['nearest_neighbors'])
    nearest_sims = np.array(nearest_sims)
    
    t_stat, p_value = scipy.stats.ttest_ind(nearest_sims, random_baseline)
    pooled_std = np.sqrt((np.var(nearest_sims) + np.var(random_baseline)) / 2)
    cohens_d = (np.mean(nearest_sims) - np.mean(random_baseline)) / pooled_std
    
    # Create bar plot
    x = range(len(queries))
    plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7, label='Query Patterns (Mean)')
    plt.plot(x, medians, 'ro', label='Medians', markersize=8)
    
    # Add overall statistics lines
    plt.axhline(y=overall_mean, color='r', linestyle='--', 
                label=f'Overall Mean: {overall_mean:.3f}')
    plt.axhline(y=overall_median, color='g', linestyle='--', 
                label=f'Overall Median: {overall_median:.3f}')
    plt.axhline(y=random_mean, color='gray', linestyle=':', 
                label=f'Random Baseline: {random_mean:.3f} ± {random_std:.3f}')
    
    # Add shaded area for random baseline standard deviation
    plt.axhspan(random_mean - random_std, random_mean + random_std, 
                color='gray', alpha=0.1)
    
    # Add statistical information to the plot
    stat_text = f'p-value: {p_value:.2e}\nCohen\'s d: {cohens_d:.2f}'
    plt.text(0.02, 0.98, stat_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xticks(x, [q[:50] + '...' if len(q) > 50 else q for q in queries], rotation=45, ha='right')
    plt.ylabel('Nearest Neighbor Cosine Similarity')
    plt.title('Nearest Neighbor Similarities by Query Pattern\nwith Random Baseline Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/query_nearest_neighbor_means.png',
                bbox_inches='tight', dpi=300)
    plt.close()
    
    return overall_mean, overall_median

def plot_query_vectors_pca(
    query_vectors: Dict[str, List[Dict[str, Any]]],
    genre_centroids: Dict[str, np.ndarray]
) -> None:
    """
    Create one plot per query pattern showing how that query shifts results across genres.
    Shows vectors from genre centroids to their respective query points in PCA space.

    Args:
        query_vectors: Mapping of query patterns to vector data
        genre_centroids: Mapping of genres to their centroid coordinates

    Outputs:
        Saves multiple plots to results/query_analysis/query_template_*.png

    Example:
        >>> plot_query_vectors_pca(query_vectors, genre_centroids)
    """
    # Create output directory if it doesn't exist
    output_dir = 'results/query_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for PCA
    all_points = []
    for genre, centroid in genre_centroids.items():
        all_points.append(centroid)
    
    # Add query points
    for vectors in query_vectors.values():
        for v in vectors:
            all_points.append(v['end'])
    
    # Fit PCA
    pca = PCA(n_components=2)
    all_points_2d = pca.fit_transform(all_points)
    
    # Create centroid lookup
    centroid_2d = {genre: all_points_2d[i] for i, genre in enumerate(genre_centroids.keys())}
    
    # Create one plot per query template
    for query_template, vectors in query_vectors.items():
        if len(vectors) < 5:  # Skip patterns with too few examples
            continue
            
        plt.figure(figsize=(15, 10))
        
        # Plot all centroids in black
        centroid_points = np.array(list(centroid_2d.values()))
        plt.scatter(centroid_points[:, 0], centroid_points[:, 1], 
                   c='black', s=50, label='Genre Centroids')
        
        # Add small labels for centroids
        for genre, point in centroid_2d.items():
            plt.annotate(genre, (point[0], point[1]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
        
        # Plot vectors for this query pattern
        for v in vectors:
            start = centroid_2d[v['genre']]
            end = pca.transform([v['end']])[0]
            
            # Draw arrow
            arrow = FancyArrowPatch(
                start, end,
                arrowstyle='-|>',
                mutation_scale=10,
                linewidth=1.5,
                color='red',
                alpha=0.6
            )
            plt.gca().add_patch(arrow)
        
        # Example query with placeholder
        example_query = query_template.format(genre="<genre>")
        plt.title(f'Query Template: "{example_query}"')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.grid(True, alpha=0.3)
        
        # Add legend with vector count
        plt.plot([], [], color='red', label=f'Query vectors (n={len(vectors)})')
        plt.legend()
        
        plt.tight_layout()
        
        # Create safe filename from query template
        safe_filename = "".join(x for x in query_template if x.isalnum() or x in (' ', '-', '_', '{', '}'))[:50]
        plt.savefig(f'{output_dir}/query_template_{safe_filename}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

def main() -> None:
    """
    Main execution function for query effect analysis.

    Orchestrates the full analysis pipeline:
    1. Loads query data from database
    2. Calculates query effect vectors
    3. Analyzes vector similarities
    4. Generates visualizations
    5. Performs statistical analysis
    6. Outputs summary statistics

    The results help understand how different query formulations systematically
    affect search results across different genres.

    Example:
        >>> main()
    """
    # Ensure directories exist
    ensure_directories()

    # Load and process data
    print("Loading query data...")
    query_data, genre_centroids = load_query_data()
    
    print("Calculating query vectors...")
    query_vectors = calculate_query_vectors(query_data)
    
    print("Calculating vector similarities...")
    similarities = calculate_vector_similarities(query_vectors)

    print("Generating similarity distribution plots...")
    plot_similarity_distribution(similarities)
    plot_nearest_neighbor_similarities(similarities)
    
    print("Generating random baseline...")
    random_baseline = generate_random_baseline(query_vectors)
    
    print("Generating mean similarity plots...")
    overall_mean, overall_median = plot_similarity_means(similarities, random_baseline)
    nn_mean, nn_median = plot_nearest_neighbor_means(similarities, random_baseline)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"\nOverall Statistics:")
    print(f"Overall Mean: {overall_mean:.3f}")
    print(f"Overall Median: {overall_median:.3f}")

    print("Generating query vectors PCA plots...")
    plot_query_vectors_pca(query_vectors, genre_centroids)

if __name__ == '__main__':
    main()
