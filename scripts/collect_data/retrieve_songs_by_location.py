"""
Song Retrieval Module

This module provides functionality to retrieve songs from either a local binned array
or through an API endpoint based on location indices. The binned array is a pre-processed
data structure that organizes songs into bins based on their characteristics or metadata.

The module supports both local and API-based retrieval methods, allowing for flexible
deployment scenarios and easy switching between data sources.

Dependencies:
    - pickle: For loading local binned array data
    - json: For API response handling
    - random: For random sampling of songs
    - requests: For API communication
"""

import pickle
import random

#load the binned array
with open('data/binned_array.pkl', 'rb') as f:
    binned_array = pickle.load(f)

BINNED_ARRAY = binned_array

def retrieve_songs_by_location_local(bin_indices: list, max_results: int = 100) -> list:
    """
    Retrieve songs from a local binned array using specified indices.

    This function loads a pre-processed binned array from disk and returns songs
    from the specified bin location. If the number of songs in the bin exceeds
    max_results, a random sample is returned.

    Args:
        bin_indices (list): List of integers representing the indices to access
            in the multi-dimensional binned array
        max_results (int, optional): Maximum number of songs to return. 
            Defaults to 100.

    Returns:
        list: List of songs from the specified bin location. If the bin contains
            more songs than max_results, returns a random sample of max_results songs.

    Example:
        >>> songs = retrieve_songs_by_location_local([1, 2, 3], max_results=50)
        >>> print(f"Retrieved {len(songs)} songs")
    """


    #get the songs in the bin by using the indices to index into the binned array
    songs_in_bin = BINNED_ARRAY[tuple(bin_indices)]  # Convert list of indices to tuple for proper array indexing
    
    if len(songs_in_bin) < max_results:
        return songs_in_bin
    #return random max_results songs in the bin
    return random.sample(songs_in_bin, max_results)
