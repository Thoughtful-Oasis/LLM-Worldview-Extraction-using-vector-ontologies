"""
Genre Search Distribution Analysis Module

This module analyzes how search results map to different genres by performing
multiple searches with various query patterns and tracking the genre distributions of
the returned songs. Results are stored in an SQLite database for further analysis.

The module performs searches using natural language queries, processes them through an LLM
to extract location context, and then analyzes the genre distribution of the returned songs.

Dependencies:
    - sqlite3
    - collections
    - json
    - llm_response (local module)
    - retrieve_songs_by_location (local module)
    - threading
    - contextlib
    - random
Database Schema:
    Table: genre_search_distributions
        - query_genre (TEXT): The input genre used for searching
        - search_results (TEXT): JSON string containing detailed search results
        - distribution (TEXT): JSON string containing overall genre distribution
        - total_songs_analyzed (INTEGER): Total number of songs analyzed for this genre
"""

import sqlite3
from collections import Counter
import json
from get_genres import get_genres_for_songs_batch
from llm_response import get_location_from_llm
from retrieve_songs_by_location import retrieve_songs_by_location_local
from contextlib import contextmanager
from threading import Lock
import random

db_lock = Lock()

@contextmanager
def get_db_connection():
    """
    Context manager for handling SQLite database connections safely.
    
    Yields:
        sqlite3.Connection: An active database connection
        
    Example:
        >>> with get_db_connection() as conn:
        >>>     cursor = conn.cursor()
        >>>     cursor.execute("SELECT * FROM genre_search_distributions")
    """
    conn = sqlite3.connect('spotify_search_distributions.db', timeout=30.0)
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()

def init_database():
    """
    Initialize SQLite database with the required schema.
    Creates the genre_search_distributions table if it doesn't exist.
    """
    conn = sqlite3.connect('spotify_search_distributions.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS genre_search_distributions (
            query_genre TEXT PRIMARY KEY,
            search_results TEXT,
            distribution TEXT,
            total_songs_analyzed INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

def check_genre_distribution(genre: str, 
                           number_of_searches: int, number_of_songs: int = 10) -> dict:
    """
    Analyze genre distribution across multiple search variations for a given genre.

    This function performs multiple searches using different natural language patterns
    and analyzes the genre distribution of the returned songs.

    Args:
        genre (str): The genre to analyze
        number_of_searches (int): Number of different search patterns to try
        number_of_songs (int, optional): Number of songs to analyze per search. Defaults to 10.

    Returns:
        dict: Contains the following keys:
            - overall_distribution: Dictionary mapping genres to their frequencies
            - total_songs_analyzed: Total number of songs processed
            - individual_searches: List of detailed results for each search

    Example:
        >>> results = check_genre_distribution("jazz", 5, 10)
        >>> print(f"Analyzed {results['total_songs_analyzed']} songs")
    """
    all_genre_counts = Counter()
    total_songs_analyzed = 0
    search_results = []

    query_variations= [
        f"{genre}",
        f"I'm feeling {genre}",
        f"I'm in the mood for {genre}",
        f"I'm in the mood for {genre} music",
        f"I'm in the mood for {genre} songs",
        f"Find me {genre}",
        f"Find me {genre} music",
        f"Find me {genre} songs",
        f"Play me {genre}",
        f"Play me {genre} music",
        f"Play me {genre} songs",
        f"Some {genre} music",
        f"Some {genre} songs",
        f"I want {genre}",
        f"I want {genre} music",
        f"I want {genre} songs",
        f"Can you play {genre}?",
        f"Can you play {genre} music?",
        f"Can you play {genre} songs?",
        f"I'd love some {genre}",
        f"I'd love some {genre} music",
        f"I'd love some {genre} songs",
        f"Could you find me something {genre}?",
        f"What about some {genre} music?",
        f"How about some {genre} songs?",
        f"I need something {genre} right now.",
        f"Do you have any {genre} suggestions?",
        f"Let's hear some {genre}.",
        f"Let's go with {genre} music.",
        f"Any good {genre} tracks?",
        f"Can you recommend some {genre} music?",
        f"Let's vibe with {genre}.",
        f"I'm craving some {genre} tunes.",
        f"Bring on the {genre} sounds.",
        f"Queue up some {genre}.",
        f"Hit me with some {genre} music.",
        f"Can we get into some {genre}?",
        f"Set the mood with {genre}.",
        f"Let's explore some {genre} vibes.",
        f"Give me your best {genre} music.",
        f"Turn on some {genre} tunes.",
        f"Show me what you've got for {genre}.",
        f"Kick off some {genre} songs.",
        f"I'm looking for some {genre} tracks.",
        f"Help me find some {genre} music.",
        f"Let's chill with some {genre} beats.",
        f"Do you know any good {genre} tunes?"
    ]
    random.shuffle(query_variations)
    
    for search_num, query in enumerate(query_variations[:number_of_searches]):
        search_genre_counts = Counter()
        songs_in_search = 0
        
        song_ids, location = get_song_ids_from_llm_query(query)
        if len(song_ids) == 0:
            search_results.append({
                'search_number': search_num + 1,
                'query': query,
                'location': location,
                'genre_distribution': {},
                'songs_analyzed': 0,
                'tracks_data': [],
                'artists_data': [],
                'error': f"No songs found for location: {location}"
            })
            continue

        selected_songs = song_ids[:number_of_songs]
        song_genres = get_genres_for_songs_batch(selected_songs)
        print(f"Retrieved genres for {len(song_genres)} songs")  # Debug log
        
        for song_id, genres in song_genres.items():
            if genres:
                search_genre_counts.update(genres)
                all_genre_counts.update(genres)
                songs_in_search += 1
                total_songs_analyzed += 1

        if songs_in_search > 0:
            search_distribution = {
                genre: {
                    'count': count,
                    'percentage': (count / songs_in_search) * 100
                }
                for genre, count in search_genre_counts.items()
            }
            
            search_results.append({
                'search_number': search_num + 1,
                'query': query,
                'location': location,
                'genre_distribution': search_distribution,
                'songs_analyzed': songs_in_search,
            })

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
        'individual_searches': search_results
    }


def get_song_ids_from_llm_query(query: str) -> tuple[list[str], str]:
    """
    Process a natural language query through LLM to get relevant song IDs.

    Args:
        query (str): Natural language query for music search

    Returns:
        tuple[list[str], str]: A tuple containing:
            - List of Spotify track IDs
            - Extracted location context from the query
    """
    location = get_location_from_llm(query)
    try:
        song_ids = retrieve_songs_by_location_local(location)
    except Exception as e:
        print(f"Error retrieving songs by location: {e}")
        return [], location
    return song_ids, location


def build_search_db(genres: list[str], number_of_searches: int = 20, 
                   number_of_songs: int = 50) -> None:
    """
    Build a database of genre search distributions for multiple input genres.

    This function processes each genre through multiple search variations and
    stores the results in an SQLite database. It includes thread-safe database
    operations and skips already processed genres.

    Args:
        genres (list[str]): List of genres to analyze
        number_of_searches (int, optional): Number of search variations per genre. 
            Defaults to 20.
        number_of_songs (int, optional): Number of songs to analyze per search. 
            Defaults to 50.

    Example:
        >>> genres = ["jazz", "rock", "classical"]
        >>> build_search_db(genres, number_of_searches=10, number_of_songs=25)
    """
    init_database()

    with get_db_connection() as conn:
        c = conn.cursor()

        for query_genre in genres:
            c.execute('SELECT query_genre FROM genre_search_distributions WHERE query_genre = ?', (query_genre,))
            if c.fetchone():
                print(f"Skipping {query_genre} - already processed")
                continue

            print(f"\nAnalyzing genre distribution for query: {query_genre}")
            results = check_genre_distribution(query_genre, number_of_searches, number_of_songs)
            
            # Add debug logging
            print(f"Results for {query_genre}:")
            print(f"- Total songs analyzed: {results['total_songs_analyzed']}")
            print(f"- Number of searches: {len(results['individual_searches'])}")
            print(f"- Distribution: {list(results['overall_distribution'].keys())}")

            with db_lock:
                c.execute('''
                    INSERT INTO genre_search_distributions 
                    (query_genre, search_results, distribution, total_songs_analyzed)
                    VALUES (?, ?, ?, ?)
                ''', (
                    query_genre,
                    json.dumps(results['individual_searches']),
                    json.dumps(results['overall_distribution']),
                    results['total_songs_analyzed']
                ))
                print(f"Saved results for {query_genre} to database")  # Debug log

if __name__ == '__main__':
    # Example usage of the module

    genres = [
        "pop",
        "rock",
        "jazz",
        "classical",
        "hip-hop",
        "r&b",
        "electronic",
        "country",
        "reggae",
        "blues",
        "funk",
        "soul",
        "gospel",
    ]

    genres_large = [  
        "pop",
        "rock",
        "jazz",
        "classical",
        "hip-hop",
        "r&b",
        "electronic",
        "country",
        "reggae",
        "blues",
        "funk",
        "soul",
        "gospel",
        "punk",
        "metal",
        "ska",
        "dance",
        "house",
        "techno",
        "trance",
        "dubstep",
        "drum and bass",
        "folk",
        "indie",
        "alternative",
        "opera",
        "disco",
        "ambient",
        "world",
        "latin",
        "k-pop",
        "j-pop",
        "afrobeat",
        "edm",
        "grime",
        "trap",
        "garage",
        "swing",
        "bossa nova",
        "samba",
        "tango",
        "chillout",
        "progressive",
        "hardcore",
        "post-rock",
        "shoegaze",
        "synthwave",
        "new age",
        "experimental",
        "industrial",
        "lofi",
        "emo",
        "post-punk",
        "psychedelic",
    ]
    build_search_db(genres, number_of_searches=47, number_of_songs=50)
