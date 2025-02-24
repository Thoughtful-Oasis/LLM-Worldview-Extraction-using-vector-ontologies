"""
Audio Feature Binning Module

This module preprocesses audio feature data from Spotify into a multi-dimensional binned array
structure optimized for efficient song retrieval based on audio characteristics. It handles
large datasets efficiently through chunked processing and implements adaptive binning strategies.

The module creates a dense N-dimensional array where each dimension corresponds to an audio
feature (e.g., danceability, energy, etc.). Songs are assigned to bins based on their feature
values, allowing for quick retrieval of songs with similar characteristics.

Features:
    - Efficient processing of large CSV files through chunking
    - Multiple binning strategies (quantile or uniform)
    - Automatic optimization of bin counts to balance granularity and sparsity
    - Memory-efficient processing of large datasets
    - Progress tracking with tqdm

Dependencies:
    - pandas: For data processing
    - numpy: For array operations
    - pickle: For serialization
    - json: For configuration storage
    - tqdm: For progress tracking

Output Files:
    - binned_array.pkl: Serialized N-dimensional array containing binned song IDs
    - bin_edges.json: Configuration file containing bin edges and feature order
"""

# This script is used to read in the csv file containing song ids and audio feature ratings and create a database of songs optimised for searching songs based on audio feature bins.

# it reads in a csv and creates a dense 8 dimensonal array of songs binned by their audio features. it saves the array as a pickle file.

import pandas as pd
import numpy as np
import pickle
import json
from tqdm import tqdm


def calculate_empty_bins_percentage(binned_array: np.ndarray) -> float:
    """
    Calculate the percentage of empty bins in a multi-dimensional array.
    
    Args:
        binned_array (np.ndarray): N-dimensional numpy array where each cell contains
            a list of song IDs or None

    Returns:
        float: Percentage of empty bins (0.0 to 1.0)
        
    Example:
        >>> array = create_binned_array(df, features, 6)
        >>> empty_percent = calculate_empty_bins_percentage(array)
        >>> print(f"{empty_percent:.2%} bins are empty")
    """
    total_bins = np.prod(binned_array.shape)
    empty_bins = sum(1 for x in binned_array.flat if len(x) == 0)  # Count bins with empty lists
    return empty_bins / total_bins



def create_binned_array(df: pd.DataFrame, audio_features: list, n_bins: int, 
                       max_bins_empty: float = 0.5, 
                       binning_strategy: str = 'quantile') -> tuple[np.ndarray, dict]:
    """
    Create a multi-dimensional array with songs binned by audio features.
    
    Args:
        df (pd.DataFrame): DataFrame containing song data with audio feature columns
        audio_features (list): List of audio feature column names to use for binning
        n_bins (int): Number of bins per dimension
        max_bins_empty (float, optional): Maximum allowed percentage of empty bins. 
            Defaults to 0.5.
        binning_strategy (str, optional): Either 'quantile' for equal-sized bins or 
            'uniform' for uniform bins. Defaults to 'quantile'.

    Returns:
        tuple[np.ndarray, dict]: Returns either:
            - (False, None) if too many empty bins
            - (binned_array, bin_edges) where:
                - binned_array: N-dimensional numpy array containing song IDs
                - bin_edges: Dictionary mapping features to their bin edges

    Example:
        >>> features = ['danceability', 'energy', 'tempo']
        >>> array, edges = create_binned_array(df, features, 6)
        >>> if array is False:
        >>>     print("Too many empty bins")
    """
    print("Creating binned array...")
    
    # Initialize n-dimensional array with empty lists
    array_shape = tuple([n_bins] * len(audio_features))
    binned_array = np.empty(array_shape, dtype=object)
    for idx in np.ndindex(array_shape):
        binned_array[idx] = []
    
    # Calculate bin edges for each feature
    print("Calculating bin edges...")
    bin_edges = {}
    for feature in tqdm(audio_features, desc="Processing features"):
        if binning_strategy == 'quantile':
            # Remove NaN values before calculating percentiles
            feature_data = df[feature].dropna()
            if len(feature_data) == 0:
                print(f"Warning: Feature '{feature}' contains all NaN values!")
                bin_edges[feature] = np.linspace(0, 1, n_bins + 1)  # fallback to uniform bins
            else:
                bin_edges[feature] = np.percentile(feature_data, np.linspace(0, 100, n_bins + 1))

        else:
            bin_edges[feature] = np.linspace(df[feature].min(), df[feature].max(), n_bins + 1)
    
    # Process songs in chunks with progress bar
    chunk_size = 100_000
    with tqdm(total=len(df), desc="Binning songs") as pbar:
        for chunk_start in range(0, len(df), chunk_size):
            chunk = df.iloc[chunk_start:chunk_start + chunk_size]
            for _, song in chunk.iterrows():
                bin_indices = []
                for feature in audio_features:
                    bin_idx = np.digitize(song[feature], bin_edges[feature]) - 1
                    bin_idx = min(max(bin_idx, 0), n_bins - 1)
                    bin_indices.append(bin_idx)
                binned_array[tuple(bin_indices)].append(song['id'])
                pbar.update(1)
    
    # Check if too many empty bins
    empty_percentage = calculate_empty_bins_percentage(binned_array)
    print(f"Empty bins: {empty_percentage:.2%}")
    
    if empty_percentage > max_bins_empty:
        return False, None
    
    return binned_array, bin_edges

def find_optimal_bins(df: pd.DataFrame, audio_features: list, max_bins_empty: float,
                     binning_strategy: str = 'quantile', 
                     min_bins: int = 3, 
                     max_bins: int = 10) -> tuple[int, np.ndarray, dict]:
    """
    Find the optimal number of bins that keeps empty bins percentage under threshold.
    
    Args:
        df (pd.DataFrame): DataFrame containing song data
        audio_features (list): List of audio feature columns to use
        max_bins_empty (float): Maximum allowed percentage of empty bins
        binning_strategy (str, optional): Binning strategy to use. Defaults to 'quantile'.
        min_bins (int, optional): Minimum number of bins to try. Defaults to 6.
        max_bins (int, optional): Maximum number of bins to try. Defaults to 6.

    Returns:
        tuple[int, np.ndarray, dict]: Contains:
            - optimal_bins: Best number of bins found
            - optimal_result: The binned array using optimal bins
            - optimal_edges: The bin edges for optimal configuration

    Raises:
        ValueError: If no valid binning solution is found

    Example:
        >>> features = ['danceability', 'energy', 'tempo']
        >>> n_bins, array, edges = find_optimal_bins(df, features, 0.5)
        >>> print(f"Optimal bins: {n_bins}")
    """
    print("\nFinding optimal number of bins...")
    left, right = min_bins, max_bins
    optimal_bins = min_bins
    optimal_result = None
    optimal_edges = None
    
    for n_bins in tqdm(range(left, right + 1), desc="Testing bin sizes"):
        print(f"\nTrying {n_bins} bins per dimension...")
        result = create_binned_array(df, audio_features, n_bins, max_bins_empty, binning_strategy)
        
        if result[0] is False:
            print(f"{n_bins} bins: Too many empty bins")
            break
        else:
            binned_array, bin_edges = result
            optimal_bins = n_bins
            optimal_result = binned_array
            optimal_edges = bin_edges
            print(f"{n_bins} bins: Valid solution found")
    
    if optimal_result is None or optimal_edges is None:
        raise ValueError("Could not find a valid binning solution")
        
    return optimal_bins, optimal_result, optimal_edges

def save_binned_data(binned_array: np.ndarray, bin_edges: dict, 
                    audio_features: list, output_dir: str = 'data') -> None:
    """
    Save binned array and bin edges while preserving feature order.
    Ensures bin edges are saved in standard decimal format without scientific notation.
    
    Args:
        binned_array (np.ndarray): N-dimensional array containing binned song IDs
        bin_edges (dict): Dictionary mapping features to their bin edges
        audio_features (list): List of audio features in the correct order
        output_dir (str, optional): Directory to save the files. Defaults to 'data'.
    """
    # Save the binned array
    with open(f'{output_dir}/binned_array.pkl', 'wb') as f:
        pickle.dump(binned_array, f)

    # Create ordered dictionary of bin edges with formatted numbers
    bin_edges_ordered = {
        'feature_order': audio_features,
        'edges': {
            feature: [f"{edge:.10f}" for edge in bin_edges[feature]]
            for feature in audio_features
        }
    }
    
    # Save the bin edges
    with open(f'{output_dir}/bin_edges.json', 'w') as f:
        json.dump(bin_edges_ordered, f)

if __name__ == "__main__":
    """
    Main execution block for preprocessing audio feature data.
    
    Process:
    1. Reads song audio features from CSV in chunks
    2. Finds optimal binning configuration
    3. Creates binned array structure
    4. Saves results to disk
    
    The script handles large datasets efficiently through chunked processing
    and provides progress feedback throughout the operation.
    """
    print("Starting data preprocessing...")


    # change the content column to id in the same file


    # Read CSV in chunks with progress bar
    print("\nReading CSV file...")
    chunk_size = 100_000
    chunks = []
    with pd.read_csv('data/song_audio_features.csv', chunksize=chunk_size) as reader:
        for chunk in tqdm(reader, desc="Reading chunks"):
            chunks.append(chunk)
    df = pd.concat(chunks)

    print(f'\nDataset loaded:')
    print(f'Number of songs: {len(df):,}')
    print(f'Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB')

    # Define audio features to include
    audio_features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                     'instrumentalness', 'liveness', 'valence', 'tempo']

    # Process and save data
    max_bins_empty = 0.5
    binning_strategy = 'quantile'
    n_bins, binned_array, bin_edges = find_optimal_bins(
        df, audio_features, max_bins_empty, binning_strategy
    )
    print(f"Optimal number of bins: {n_bins}")
    save_binned_data(binned_array, bin_edges, audio_features)

