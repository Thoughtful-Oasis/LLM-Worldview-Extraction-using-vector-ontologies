"""
Spotify Genre Distribution Analysis Module

This module builds a database mapping the distribution of music genres across a
multi-dimensional feature space of audio characteristics. It uses random sampling
to efficiently cover the space slowly in case the dataset is very large, however we recommend to run it until it is fully popullated for best results.

The module implements a resumable process that can be safely interrupted and restarted,
maintaining its progress through persistent SQLite storage. It randomly samples locations
in the feature space to quickly achieve sparse coverage of the entire space.

The dimensionality of the feature space is determined by the bin_edges.json configuration
file, which specifies the edges for each feature dimension. This allows for flexible
analysis across any number of audio characteristics.

Features analyzed include:
    - Audio characteristics from Spotify's API (configurable dimensions)
    - Genre distributions at each point in the feature space
    - Song presence and density in different regions

Dependencies:
    - sqlite3: Database storage
    - collections: For Counter objects
    - json: Configuration and data serialization
    - random: For sampling locations
    - numpy: For array operations
    - get_genres (local module): Get genres for a batch of songs
    - retrieve_songs_by_location (local module): Song retrieval by location
    - threading: For thread-safe database operations

Database Schema:
    Table: location_distributions
        - location (TEXT PRIMARY KEY): String representation of N-dimensional coordinates
        - songs_found (INTEGER): Number of songs found at this location
        - distribution (TEXT): JSON string of genre distribution data

Configuration:
    The bin_edges.json file should contain:
        - edges: Dictionary mapping feature names to their bin edges
        - feature_order: List defining the order of features in the coordinate system
"""

import sqlite3
from collections import Counter
import json
import random
from get_genres import get_genres_for_songs_batch
from retrieve_songs_by_location import retrieve_songs_by_location_local
from contextlib import contextmanager
from threading import Lock
import numpy as np

db_lock = Lock()

@contextmanager
def get_db_connection():
    """
    Context manager for thread-safe database connections.
    
    Provides a connection with a 30-second timeout for concurrent access handling.
    Automatically commits changes and closes the connection when done.
    
    Yields:
        sqlite3.Connection: Database connection object
        
    Example:
        >>> with get_db_connection() as conn:
        >>>     cursor = conn.cursor()
        >>>     cursor.execute("SELECT * FROM location_distributions")
    """
    conn = sqlite3.connect('spotify_genre_distributions.db', timeout=30.0)
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()

def init_database():
    """
    Initialize the SQLite database with required schema.
    
    Creates the location_distributions table if it doesn't exist.
    The table stores genre distribution data for each analyzed location
    in the N-dimensional feature space.
    """
    conn = sqlite3.connect('spotify_genre_distributions.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS location_distributions (
            location TEXT PRIMARY KEY,
            songs_found INTEGER,
            distribution TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def check_genre_distribution(location: list, 
                           number_of_songs: int = 10) -> dict:
    """
    Analyze genre distribution for songs at a specific location in feature space.
    
    Retrieves songs from the specified location and analyzes their genres using
    the get_genres_for_songs_batch function. Calculates distribution percentages for each genre found.
    
    Args:
        location (list): Coordinates in the N-dimensional feature space
        number_of_songs (int, optional): Maximum songs to analyze. Defaults to 10.
    
    Returns:
        dict: Contains the following keys:
            - overall_distribution: Dict mapping genres to count and percentage
            - total_songs_analyzed: Number of songs successfully analyzed
            - location: Input location coordinates
            - songs_found: Total number of songs found at location
            
    Example:
        >>> results = check_genre_distribution([1,2,0,0,0,0,0,1], 50)
        >>> print(f"Found {results['songs_found']} songs at location")
    """
    all_genre_counts = Counter()
    total_songs_analyzed = 0

    song_ids = retrieve_songs_by_location_local(location)
    
    if len(song_ids) == 0:
        return {
            'overall_distribution': {},
            'total_songs_analyzed': 0,
            'location': location,
            'songs_found': len(song_ids)
        }
    
    random.shuffle(song_ids)
    selected_songs = song_ids[:number_of_songs]
    
    song_genres = get_genres_for_songs_batch(selected_songs)
    
    for song_id, genres in song_genres.items():
        if genres:
            all_genre_counts.update(genres)
            total_songs_analyzed += 1

    overall_distribution = {
        genre: {
            'count': count,
            'percentage': (count / total_songs_analyzed) * 100
        }
        for genre, count in all_genre_counts.items()
    } if total_songs_analyzed > 0 else {}

    return {
        'overall_distribution': overall_distribution,
        'total_songs_analyzed': total_songs_analyzed,
        'location': location,
        'songs_found': len(song_ids)
    }

if __name__ == '__main__':
    """
    Main execution block for building the genre distribution database.
    
    Process:
    1. Initializes database
    2. Loads feature space configuration from bin_edges.json
    3. Generates all possible locations in N-dimensional feature space
    4. Identifies unprocessed locations
    5. Randomly samples locations to process
    6. Analyzes genre distributions and stores results
    
    The script can be safely interrupted and restarted, as it tracks
    progress in the SQLite database.
    """
    init_database()

    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT location FROM location_distributions')
        existing_locations = {row[0] for row in c.fetchall()}

    with open('data/bin_edges.json', 'r') as f:
        edge_data = json.load(f)
        shape = [len(edge_data['edges'][feature])-1 for feature in edge_data['feature_order']]
        print(f"Found shape: {shape}")

    # Create all possible combinations for an N-dimensional array with indices 0-4
    all_locations = np.array(np.meshgrid(*[range(len(edge_data['edges'][feature])-1) for feature in edge_data['feature_order']])).T.reshape(-1, len(edge_data['feature_order']))

    remaining_locations = [
        loc for loc in all_locations 
        if ''.join(str(x) for x in loc) not in existing_locations 
    ]

    print(f"Total possible locations: {len(all_locations)}")
    print(f"Already processed: {len(existing_locations)}")
    print(f"Remaining to process: {len(remaining_locations)}")

    locations_to_process = random.sample(remaining_locations, min(1000000, len(remaining_locations)))
    print(f"Selected {len(locations_to_process)} locations to process")

    with get_db_connection() as conn:
        c = conn.cursor()

        for i, location in enumerate(locations_to_process):
            print(f'Testing location: {location} ({i+1}/{len(locations_to_process)}) {(i+1)/len(locations_to_process)*100:.2f}%')
            results = check_genre_distribution(location, number_of_songs=50)
            
            location_str = ''.join(str(x) for x in location)
            
            with db_lock:
                c.execute('''
                    INSERT INTO location_distributions (location, songs_found, distribution)
                    VALUES (?, ?, ?)
                ''', (
                    location_str,
                    results['songs_found'],
                    json.dumps(results['overall_distribution'])
                ))

                if i % 100 == 0:
                    conn.commit()
