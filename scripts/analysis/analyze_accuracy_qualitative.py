"""
Accuracy Analysis Module (Qualitative)

This module provides tools for analyzing and visualizing the accuracy of Spotify genre searches
through various qualitative metrics and visualizations. It performs several types of analysis:

1. Genre Distribution Analysis:
   - Distribution of genres across search results
   - Genre group pattern analysis
   - Visualization of genre distributions

2. Location-based Analysis:
   - PCA-based heatmaps of genre distributions
   - Dimensional analysis across feature space
   - Search point overlay visualization

3. Feature Space Analysis:
   - Pairwise dimension plots
   - Absolute and relative distribution analysis
   - Search point clustering visualization

Dependencies:
    - numpy
    - scipy
    - matplotlib
    - sklearn
    - sqlite3
    - joblib
    - analyze_consistency (local module)

Typical usage:
    python analyze_accuracy_qualitative.py

Output:
    Generates multiple visualization files in the spotify_test_results directory:
    - distributions/: Genre distribution plots
    - heatmaps/: PCA and dimensional analysis plots
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
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from analyze_consistency import FEATURE_RANGES, convert_location_to_values, get_range_midpoint

def ensure_directories() -> None:
    """
    Create all necessary subdirectories for different types of analysis results.
    
    Creates the following directory structure under 'results/':
        - distributions/: For genre distribution analysis
        - heatmaps/pca/: For PCA-based heatmaps
        - heatmaps/dimensions/: For dimensional analysis
        - models/: For saved models

    Example:
        >>> ensure_directories()
        >>> # All directories are created if they don't exist
    """
    base_dir = 'results'
    subdirs = [
        'distributions',
        'heatmaps/pca',
        'heatmaps/dimensions',
        'models'
    ]
    
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)

def get_genre_group_distribution(
    distribution: Dict[str, Dict[str, Any]], 
    total_songs: int,
    min_len: int = 3
) -> Dict[str, int]:
    """
    Create distribution of genre substrings that represent meaningful patterns.

    Analyzes genre names to find common substrings and patterns, helping identify
    related genre groups and their relative frequencies.

    Args:
        distribution: Dictionary mapping genres to their occurrence data
        total_songs: Total number of songs analyzed
        min_len: Minimum length for substring consideration (default: 3)

    Returns:
        dict: Mapping of genre substrings to their occurrence counts

    Example:
        >>> dist = {'rock metal': {'count': 10}, 'metal core': {'count': 5}}
        >>> groups = get_genre_group_distribution(dist, 15, min_len=3)
        >>> print(groups)  # Shows counts for 'rock', 'metal', 'core', etc.
    """
    genre_group_distribution = {}
    
    for genre, data in distribution.items():
        count = data['count']
        words = genre.lower().split()
        
        # Generate meaningful substrings
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                substring = ' '.join(words[i:j])
                if len(substring) >= min_len:
                    if substring in genre_group_distribution:
                        genre_group_distribution[substring] += count
                    else:
                        genre_group_distribution[substring] = count
                        
        # Also add individual words that meet minimum length
        for word in words:
            if len(word) >= min_len:
                if word in genre_group_distribution:
                    genre_group_distribution[word] += count
                else:
                    genre_group_distribution[word] = count
    
    return genre_group_distribution

def build_search_graphs(significant_genres: Optional[Set[str]] = None) -> None:
    """
    Build search graphs for analyzing genre distributions.

    Creates visualizations showing how genres are distributed across search results,
    including both full genre distributions and genre group patterns.

    Args:
        significant_genres: Optional set of genres to analyze. If None, analyzes all genres.

    Outputs:
        Saves multiple visualization files to results/distributions/:
        - {genre}_distribution.png: Full genre distribution plots
        - {genre}_group_distribution.png: Genre group pattern plots

    Example:
        >>> build_search_graphs({'rock', 'jazz'})
        >>> # Creates distribution plots for rock and jazz genres
    """
    search_conn = sqlite3.connect('spotify_search_distributions.db')
    dist_conn = sqlite3.connect('spotify_genre_distributions.db')
    search_c = search_conn.cursor()
    
    if significant_genres:
        placeholders = ','.join('?' * len(significant_genres))
        query = f'SELECT query_genre, distribution, search_results FROM genre_search_distributions WHERE query_genre IN ({placeholders})'
        results = search_c.execute(query, list(significant_genres)).fetchall()
    else:
        results = search_c.execute('SELECT query_genre, distribution, search_results FROM genre_search_distributions').fetchall()

    locations = {}
    
    for genre, distribution_json, search_results_json in results:
        distribution = json.loads(distribution_json)
        search_results = json.loads(search_results_json)
        
        for search in search_results:
            location = search['location']
            if genre not in locations:
                locations[genre] = []
            locations[genre].append(location)
        
        # Create distribution plots
        genre_counts = {g: data['count'] for g, data in distribution.items()}
        filtered_counts = {k: v for k, v in genre_counts.items() if v > 1}
        top_15_counts = dict(sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:15])

        plt.figure(figsize=(12, 6))
        plt.bar(list(top_15_counts.keys()), list(top_15_counts.values()))
        plt.title(f'{genre} - Genre Distribution (Top 15)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'results/distributions/{genre}_distribution.png')
        plt.close()

        # Genre group distribution
        genre_group_distribution = get_genre_group_distribution(distribution, None)
        filtered_groups = {k: v for k, v in genre_group_distribution.items() if v > 1}
        top_15_groups = dict(sorted(filtered_groups.items(), key=lambda x: x[1], reverse=True)[:15])

        plt.figure(figsize=(12, 6))
        plt.bar(list(top_15_groups.keys()), list(top_15_groups.values()))
        plt.title(f'{genre} - Genre Group Distribution (Top 15)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'results/distributions/{genre}_group_distribution.png')
        plt.close()
    
    search_conn.close()
    dist_conn.close()

def plot_genre_distributions_by_location(
    significant_genres: Optional[Set[str]] = None
) -> None:
    """
    Create heatmaps showing genre distribution across locations, with optional search point overlays.
    
    Args:
        significant_genres: Optional set of genres to process. If None, process all genres.

    Outputs:
        Saves multiple visualization files to spotify_test_results/heatmaps/pca/:
        - {genre}_distribution_relative_heatmap[_with_overlay].png
        - {genre}_distribution_absolute_heatmap[_with_overlay].png

    Example:
        >>> plot_genre_distributions_by_location({'rock', 'jazz'})
        >>> # Creates heatmaps showing distribution of rock and jazz genres
    """
    # Connect to both databases
    search_conn = sqlite3.connect('spotify_search_distributions.db')
    dist_conn = sqlite3.connect('spotify_genre_distributions.db')
    search_c = search_conn.cursor()
    dist_c = dist_conn.cursor()
    
    # Get all genres from search database
    if significant_genres:
        placeholders = ','.join('?' * len(significant_genres))
        query = f'SELECT DISTINCT query_genre FROM genre_search_distributions WHERE query_genre IN ({placeholders})'
        search_c.execute(query, list(significant_genres))
    else:
        search_c.execute('SELECT DISTINCT query_genre FROM genre_search_distributions')
    genres = {row[0] for row in search_c.fetchall()}
    
    print(f"Processing {len(genres)} genres")
    
    # Get all location distributions from distribution database
    dist_c.execute('SELECT location, songs_found, distribution FROM location_distributions')
    location_results = dist_c.fetchall()
    
    original_points = []
    projected_points = []
    genre_counts = {genre: {} for genre in genres}
    song_counts = {}
    
    print("Processing location distributions...")
    for location_str, songs_found, distribution_json in location_results:
        location = [int(x) for x in location_str]
        original_points.append(location)
        location_tuple = tuple(location)
        song_counts[location_tuple] = songs_found
        
        # Calculate genre presence for all genres
        distribution = json.loads(distribution_json)
        for genre in genres:
            genre_counts[genre][location_tuple] = calculate_genre_presence(
                distribution, genre)
    
    # Get search locations from database
    search_locations = {genre: [] for genre in genres}
    if significant_genres:
        placeholders = ','.join('?' * len(significant_genres))
        query = f'SELECT query_genre, search_results FROM genre_search_distributions WHERE query_genre IN ({placeholders})'
        search_results = search_c.execute(query, list(significant_genres)).fetchall()
    else:
        search_c.execute('SELECT query_genre, search_results FROM genre_search_distributions')
        search_results = search_c.fetchall()
    
    print("Collecting search points for overlay...")
    for genre, search_results_json in search_results:
        search_results_data = json.loads(search_results_json)
        for search in search_results_data:
            if 'location' in search and ('is_average' not in search or search['is_average'] == False):
                search_locations[genre].append(search['location'])
    
    # Project points using saved PCA
    projected_points = transform_new_points(np.array([convert_location_to_values(loc, FEATURE_RANGES) for loc in original_points]))
    
    # Create grid for heatmap
    x_min, x_max = projected_points[:, 0].min(), projected_points[:, 0].max()
    y_min, y_max = projected_points[:, 1].min(), projected_points[:, 1].max()
    grid_size = 50
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Project search points
    search_projected = {
        genre: transform_new_points(np.array([convert_location_to_values(loc, FEATURE_RANGES) for loc in locs])) 
        for genre, locs in search_locations.items()
        if locs
    }

    print("Generating heatmaps...")
    # Generate both types of plots for each genre
    for genre in genres:
        for include_overlay in [False, True]:
            # Calculate heatmap data for relative presence
            grid_counts = np.zeros((grid_size, grid_size))
            grid_weights = np.zeros((grid_size, grid_size))
            
            # Calculate heatmap data for absolute counts
            grid_absolute = np.zeros((grid_size, grid_size))
            
            for i in range(len(original_points)):
                proj_point = projected_points[i]
                orig_point = tuple(original_points[i])
                
                x_idx = np.digitize(proj_point[0], x_grid) - 1
                y_idx = np.digitize(proj_point[1], y_grid) - 1
                
                if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                    # Relative presence calculation
                    count = genre_counts[genre][orig_point]
                    grid_counts[y_idx, x_idx] += count
                    grid_weights[y_idx, x_idx] += 1
                    
                    # Absolute count calculation
                    if orig_point in genre_counts[genre]:
                        absolute_count = genre_counts[genre][orig_point] * song_counts[orig_point]
                        grid_absolute[y_idx, x_idx] += absolute_count
            
            # Average the relative counts
            mask = grid_weights > 0
            grid_counts[mask] = grid_counts[mask] / grid_weights[mask]
            
            # Plot relative presence heatmap
            plt.figure(figsize=(12, 8))
            plt.pcolormesh(xx, yy, grid_counts, shading='auto', cmap='YlOrRd')
            plt.colorbar(label=f'{genre.capitalize()} Genre Relative Presence')
            
            if include_overlay and genre in search_projected:
                points = search_projected[genre]
                plt.scatter(points[:, 0], points[:, 1], color='blue', alpha=0.6, s=50)
                if len(points) >= 3:
                    try:
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        hull_points = np.vstack((hull_points, hull_points[0]))
                        plt.fill(hull_points[:, 0], hull_points[:, 1], color='blue', alpha=0.2)
                        plt.plot(hull_points[:, 0], hull_points[:, 1], color='blue', alpha=0.5)
                    except scipy.spatial._qhull.QhullError:
                        print(f"Couldn't create convex hull for {genre}")
            
            suffix = "_with_overlay" if include_overlay else ""
            plt.title(f'Relative Distribution of {genre.capitalize()} Genre in PCA Space')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.tight_layout()
            plt.savefig(
                f'results/heatmaps/pca/{genre}_distribution_relative_heatmap{suffix}.png',
                bbox_inches='tight', dpi=300
            )
            plt.close()
            
            # Plot absolute count heatmap
            plt.figure(figsize=(12, 8))
            plt.pcolormesh(xx, yy, grid_absolute, shading='auto', cmap='YlOrRd')
            plt.colorbar(label=f'{genre.capitalize()} Genre Absolute Count')
            
            if include_overlay and genre in search_projected:
                points = search_projected[genre]
                plt.scatter(points[:, 0], points[:, 1], color='blue', alpha=0.6, s=50)
                if len(points) >= 3:
                    try:
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        hull_points = np.vstack((hull_points, hull_points[0]))
                        plt.fill(hull_points[:, 0], hull_points[:, 1], color='blue', alpha=0.2)
                        plt.plot(hull_points[:, 0], hull_points[:, 1], color='blue', alpha=0.5)
                    except scipy.spatial._qhull.QhullError:
                        print(f"Couldn't create convex hull for {genre}")
            
            suffix = "_with_overlay" if include_overlay else ""
            plt.title(f'Absolute Distribution of {genre.capitalize()} Genre in PCA Space')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.tight_layout()
            plt.savefig(
                f'results/heatmaps/pca/{genre}_distribution_absolute_heatmap{suffix}.png',
                bbox_inches='tight', dpi=300
            )
            plt.close()
            print(f'Created {genre}_distribution_heatmaps{suffix}.png')

    search_conn.close()
    dist_conn.close()

def calculate_genre_presence(
    distribution: Dict[str, Dict[str, Any]], 
    target_genre: str
) -> float:
    """
    Calculate the relative presence of a genre at a location,
    normalized by total percentage of all genres at that location.

    Args:
        distribution: Dictionary mapping genres to their occurrence data
        target_genre: Genre to calculate presence for

    Returns:
        float: Normalized presence value between 0 and 1

    Example:
        >>> dist = {'rock': {'percentage': 60}, 'pop': {'percentage': 40}}
        >>> presence = calculate_genre_presence(dist, 'rock')
        >>> print(presence)  # Returns 0.6
    """
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

def transform_new_points(new_points: np.ndarray) -> np.ndarray:
    """
    Transform new points using the saved PCA model.
    
    Args:
        new_points: numpy array of shape (n_samples, n_features)
                   Must have same number of features as training data
    
    Returns:
        np.ndarray: Transformed points of shape (n_samples, 2)

    Example:
        >>> points = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        >>> transformed = transform_new_points(points)
        >>> print(transformed.shape)  # Shows (2, 2)
    """
    pca = joblib.load('results/models/pca_model.joblib')
    return pca.transform(new_points)

def plot_dimensional_distributions(
    significant_genres: Optional[Set[str]] = None
) -> None:
    """
    Create pairwise dimension plots for each genre showing absolute distribution.

    Generates visualizations showing how genres are distributed across different
    feature dimensions, with optional search point overlays.

    Args:
        significant_genres: Optional set of genres to analyze. If None, analyzes all genres.

    Outputs:
        Saves visualization files to spotify_test_results/heatmaps/dimensions/:
        - {genre}_{dimension}_distribution.png for each genre and feature dimension

    Example:
        >>> plot_dimensional_distributions({'rock', 'jazz'})
        >>> # Creates dimensional distribution plots for rock and jazz
    """
    search_conn = sqlite3.connect('spotify_search_distributions.db')
    dist_conn = sqlite3.connect('spotify_genre_distributions.db')
    dist_c = dist_conn.cursor()
    search_c = search_conn.cursor()
    
    # Get number of dimensions from FEATURE_RANGES
    num_dimensions = len(FEATURE_RANGES)
    
    # Get genres and their search points
    search_locations = {}
    if significant_genres:
        placeholders = ','.join('?' * len(significant_genres))
        query = f'SELECT query_genre, search_results FROM genre_search_distributions WHERE query_genre IN ({placeholders})'
        search_results = search_c.execute(query, list(significant_genres)).fetchall()
        genres = significant_genres
    else:
        search_results = search_c.execute('SELECT query_genre, search_results FROM genre_search_distributions').fetchall()
        genres = {row[0] for row in search_results}
    
    # Collect search points for each genre
    for genre, search_results_json in search_results:
        search_results_data = json.loads(search_results_json)
        if genre not in search_locations:
            search_locations[genre] = []
        for search in search_results_data:
            if 'location' in search and ('is_average' not in search or not search['is_average']):
                search_locations[genre].append(search['location'])
    
    # Get location distributions
    dist_c.execute('SELECT location, songs_found, distribution FROM location_distributions')
    location_results = dist_c.fetchall()
    
    # Feature names for axis labels
    feature_names = [feature['name'] for feature in FEATURE_RANGES]
    
    for genre in genres:
        # Process location data
        locations = []
        genre_counts = []
        for location_str, songs_found, distribution_json in location_results:
            location = [int(x) for x in location_str]
            distribution = json.loads(distribution_json)
            
            genre_presence = calculate_genre_presence(distribution, genre)
            absolute_count = genre_presence * songs_found
            
            locations.append(location)
            genre_counts.append(absolute_count)
        
        locations = np.array(locations)
        genre_counts = np.array(genre_counts)
        search_points = np.array(search_locations.get(genre, []))
        
        # Create a plot for each dimension
        for dim1 in range(num_dimensions):
            # Create figure with subplots for each other dimension
            n_plots = num_dimensions - 1
            n_rows = (n_plots + 3) // 4  # Ceiling division to determine number of rows
            fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
            fig.suptitle(f'{genre.capitalize()} Distribution - {feature_names[dim1]}', fontsize=16)
            axes = axes.flatten()
            
            plot_idx = 0
            for dim2 in range(num_dimensions):
                if dim1 != dim2:
                    # Convert indices to actual feature values for binning
                    dim1_ranges = FEATURE_RANGES[dim1]['ranges']
                    dim2_ranges = FEATURE_RANGES[dim2]['ranges']
                    
                    # Create bin edges using the actual feature values
                    xedges = [float(edge.split('-')[0]) for edge in dim1_ranges] + [float(dim1_ranges[-1].split('-')[1])]
                    yedges = [float(edge.split('-')[0]) for edge in dim2_ranges] + [float(dim2_ranges[-1].split('-')[1])]
                    
                    # Convert locations to actual feature values for histogram
                    x_values = np.array([convert_location_to_values(loc, FEATURE_RANGES)[dim1] for loc in locations])
                    y_values = np.array([convert_location_to_values(loc, FEATURE_RANGES)[dim2] for loc in locations])
                    
                    # Create 2D histogram with actual feature values
                    hist, xedges, yedges = np.histogram2d(
                        x_values,
                        y_values,
                        weights=genre_counts,
                        bins=[xedges, yedges]
                    )
                    
                    # Create meshgrid from edges
                    X, Y = np.meshgrid(xedges, yedges)
                    im = axes[plot_idx].pcolormesh(
                        X, Y, hist.T,
                        cmap='YlOrRd',
                        shading='flat'
                    )
                    
                    # Set limits to match the edges exactly
                    axes[plot_idx].set_xlim(xedges[0], xedges[-1])
                    axes[plot_idx].set_ylim(yedges[0], yedges[-1])
                    
                    # Remove white space around plot
                    axes[plot_idx].set_position([
                        axes[plot_idx].get_position().x0,
                        axes[plot_idx].get_position().y0,
                        axes[plot_idx].get_position().width,
                        axes[plot_idx].get_position().height
                    ])
                    
                    # Plot search points if they exist
                    if len(search_points) > 0:
                        # Convert search points to actual feature values
                        search_x_values = np.array([convert_location_to_values(loc, FEATURE_RANGES)[dim1] for loc in search_points])
                        search_y_values = np.array([convert_location_to_values(loc, FEATURE_RANGES)[dim2] for loc in search_points])
                        
                        # Plot points at their actual feature values
                        axes[plot_idx].scatter(search_x_values, search_y_values, 
                                             color='blue', alpha=0.6, s=50)
                    
                    axes[plot_idx].set_xlabel(feature_names[dim1])
                    axes[plot_idx].set_ylabel(feature_names[dim2])
                    
                    # Add colorbar with smaller size
                    plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
                    plot_idx += 1
            
            # Hide empty subplots if any
            for idx in range(plot_idx, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            # Use dimension name instead of index in filename
            safe_dim_name = feature_names[dim1].lower().replace(' ', '_')
            plt.savefig(
                f'results/heatmaps/dimensions/{genre}_{safe_dim_name}_distribution.png',
                bbox_inches='tight',
                dpi=500,
                pad_inches=0  # Remove padding around the saved figure
            )
            plt.close()
            print(f'Created dimensional distribution plot for {genre} - {feature_names[dim1]}')
    
    search_conn.close()
    dist_conn.close()

if __name__ == '__main__':
    """
    Main execution function for accuracy analysis.

    Performs the complete qualitative accuracy analysis pipeline:
    1. Creates necessary directories
    2. Builds search distribution graphs
    3. Generates location-based distribution plots
    4. Creates dimensional distribution visualizations

    Example:
        >>> python analyze_accuracy_qualitative.py
    """
    ensure_directories()
    build_search_graphs()
    plot_genre_distributions_by_location()
    plot_dimensional_distributions()
