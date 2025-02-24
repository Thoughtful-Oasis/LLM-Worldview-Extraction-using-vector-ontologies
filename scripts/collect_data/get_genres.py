import pandas as pd
from typing import List, Tuple

# Global DataFrame to cache the data
_GENRES_DF = None

def _load_genres_df():
    """Load the genres DataFrame if not already loaded"""
    global _GENRES_DF
    if _GENRES_DF is None:
        _GENRES_DF = pd.read_csv('data/song_audio_features.csv')

def get_genres_for_songs_batch(song_ids: List[str]) -> Tuple[dict, dict, dict]:
    """
    Get genres for a batch of songs from the csv file containing the song ids and genres

    Args:
        song_ids (List[str]): List of song ids to get genres for

    Returns:
        Tuple[dict, dict, dict]: Three dictionaries:
            - song_id to genre mapping
            - genre to count mapping
            - genre to song_ids mapping
    """
    # Load DataFrame if not already loaded
    _load_genres_df()
    
    # Initialize return dictionaries
    song_to_genre = {}
    
    # Filter DataFrame to only include requested song_ids
    df_filtered = _GENRES_DF[_GENRES_DF['id'].isin(song_ids)]
    
    # Iterate through filtered rows
    for _, row in df_filtered.iterrows():
        song_id = row['id']
        genre = str(row['genres']) if pd.notna(row['genres']) else ''
        if genre:
            genres = genre.split(';')
            song_to_genre[song_id] = genres
        else:
            song_to_genre[song_id] = []
    
    return song_to_genre    