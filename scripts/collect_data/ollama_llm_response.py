"""
LLM Query Processing Module (Ollama Version)
"""

import json
import ollama
from pydantic import BaseModel
from typing import Dict

# Define the response schema using Pydantic
class LocationResponse(BaseModel):
    location: Dict[str, int]

class LocationResponseWithReasoning(LocationResponse):
    location: Dict[str, int]
    reasoning: str


def get_location_from_llm(query: str, model: str = "mistral-large", reasoning: bool = False) -> list[int]:
    """
    Process a natural language query to extract feature space coordinates using Ollama.
    """
    # Load feature space configuration
    with open('data/bin_edges.json', 'r') as f:
        bin_edge_data = json.load(f)
        bin_edges = bin_edge_data['edges']
        feature_order = bin_edge_data['feature_order']
        #feature_order = feature_order[::-1]
        print(feature_order)
    
    # Add the feature ranges as a constant
    FEATURE_RANGES = [{'name': feature, 'ranges': [f'{bin_edge_data['edges'][feature][i]}-{bin_edge_data['edges'][feature][i+1]}' for i in range(len(bin_edge_data['edges'][feature])-1)]} for feature in feature_order]

    # Construct system prompt
    system_prompt = f"""Answer in a single JSON object with an entry called 'location', which is a list
of variables formatted as dimension names as keys and indices as values. These describe the
location in the {len(feature_order)}D vibe space, with the dimensions {feature_order} taken by a vibe corresponding
to the music you would recommend based on the chat history. Return variables as indices corresponding
to the bucket values in this pattern (zero-indexed) ensuring that the values are within the range of the bins (0-{len(bin_edges[feature_order[0]])-1}): {FEATURE_RANGES}"""

    # Create schema for structured output
    schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "object",
                "properties": {
                    key: {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": len(bin_edges[key])-2
                    } for key in feature_order
                },
                "required": list(feature_order),
                "additionalProperties": False
            }
        },
        "required": ["location"]
    }

    if reasoning:
        schema["properties"]["reasoning"] = {
            "type": "string",
            "description": "A short reasoning which location in the vibe space is the best fit for the query broken down by dimension"
        }
        schema["required"].append("reasoning")
    # Make API call with structured response format
    response = ollama.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        model=model,  # or your preferred model
        format=schema,
        options={"temperature": 0.0}
    )
    print(response.message.content)
    # Parse response using Pydantic
    if reasoning:
        location_data = LocationResponseWithReasoning.model_validate_json(response.message.content)
        reasoning = location_data.reasoning
        print(reasoning)
    else:
        location_data = LocationResponse.model_validate_json(response.message.content)
    # Order the indices according to feature_order
    indices = [location_data.location[key] for key in feature_order]
    print(indices)
    return indices