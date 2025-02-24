"""
LLM Query Processing Module

This module handles the processing of natural language queries through a Large Language Model
(LLM) to extract location coordinates in a multi-dimensional feature space. It uses the OpenAI
API to interpret queries and map them to specific coordinates based on predefined bin edges.

The module reads feature space configuration from a JSON file that defines both the feature
dimensions and their corresponding bin edges. This allows for flexible dimensionality in the
feature space while maintaining consistent interpretation of queries.

Dependencies:
    - json: For reading configuration data
    - openai: For LLM API access
    - os: For environment variable access

Configuration:
    The bin_edges.json file should contain:
        - edges: Dictionary mapping feature names to their bin edges
        - feature_order: List defining the order of features in the coordinate system

Environment Variables:
    - OPENAI_API_KEY: API key for OpenAI services
"""

import json
from openai import OpenAI
import os

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_location_from_llm(query: str) -> list[int]:
    """
    Process a natural language query to extract feature space coordinates.

    This function sends the query to an LLM with specific instructions to interpret
    the query in the context of the feature space defined in bin_edges.json. The LLM
    returns coordinates that best match the musical characteristics described in the query.

    Args:
        query (str): Natural language query describing desired musical characteristics
            Example: "I want upbeat electronic music for a workout session"

    Returns:
        list[int]: List of indices corresponding to the location in feature space
            Each index corresponds to a bin in the respective dimension, ordered
            according to the feature_order in the configuration.

    Example:
        >>> location = get_location_from_llm("I want calm classical music for studying")
        >>> print(f"Mapped to coordinates: {location}")

    Note:
        The function uses a zero-temperature setting to ensure deterministic responses
        and enforces a strict JSON schema to guarantee valid coordinate outputs.
    """
    # Load feature space configuration
    with open('data/bin_edges.json', 'r') as f:
        bin_edge_data = json.load(f)
        bin_edges = bin_edge_data['edges']
        feature_order = bin_edge_data['feature_order']
    # Add the feature ranges as a constant
    FEATURE_RANGES = [{'name': feature, 'ranges': [f'{bin_edge_data['edges'][feature][i]}-{bin_edge_data['edges'][feature][i+1]}' for i in range(len(bin_edge_data['edges'][feature])-1)]} for feature in bin_edge_data['feature_order']]

    
    # Construct system prompt with feature space information
    system_prompt = f"""Answer in a single JSON object with an entry called 'location', which is a list
of variables formatted as dimension names as keys and indices as values. These describe the
location in the {len(feature_order)}D vibe space, with the dimensions {feature_order} taken by a vibe corresponding
to the music you would recommend based on the chat history. Return variables as indices corresponding
to the bucket values in this pattern (zero-indexed) ensuring that the values are within the range of the bins (0-{len(bin_edges[feature_order[0]])-1}): {FEATURE_RANGES}"""
    
    user_prompt = query

    # Make API call with structured response format
    completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "location",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": { key: {"type": "integer", "min": 0, "max": len(bin_edges[key])-2} for key in bin_edges.keys()},
                            "required": [key for key in bin_edges.keys()],
                            "additionalProperties": False
                        }
                    }
                },
                temperature=0.0
        )   
    
    # Parse and order the response according to feature_order
    response_dict = json.loads(completion.choices[0].message.content)
    indices = [response_dict[key] for key in feature_order]
    print(indices)
    return indices