# Vector-Ontologies-as-an-interpretable-LLM-worldview-extraction-method

This repository contains the code for all the experiments and charts in the paper "Vector Ontologies as an Interpretable LLM Worldview Extraction Method"

## Data

The data is expected to be in the 'Data' folder in the following format:
song_audio_features.csv - a csv file with song ideas and their audio features as well as their genres. 
This repository contains a sample dataset file, but in order to do the full analysis you will need a much larger dataset. The dataset used in the paper contains roughly 20 Million songs. Sadly sharing this dataset is not possible due to spotifies terms of service. A smaller dataset is available on Kaggle (https://www.kaggle.com/datasets/kasparrothenfusser/spotify-songs-with-audio-features-and-genres/data) (1 Million songs), which does replicate the results.

## Data Preprocessing

In order to use the data we need to make it available by its location in the Vector ontology. To do so, we run the preprocess_data.py script. This script will take the raw data and preprocess it into a dense high dimension array representing the vector space. (the reason for this is explained in the paper)

to run the preprocess_data.py script, run the following command:
```
python scripts/preprocessing/preprocess_data.py
```

## Building the search database

REQUIRED: set environment variable for "OPENAI_API_KEY"

The Search database holds records of query locations and their corresponding songs and genre distribution, which will be required for the analysis. These records are created by running the build_search_db.py script and setting the appropriate api keys.

you can adjust the following parameters:
- number_of_searches: the number of different queries to run for each genre (make sure to have enough query variations in the query_variations list)
- number_of_songs: the number of songs to retrieve for each query.
- list of genres, we provided a default list and a large list, which was used in the paper.


### LLM setting
You can change the import function in the build_search_db.py to import from ollama_llm_response.py instead of llm_response.py this will run the experiment with a locally hosted model using Ollama.

to run the build_search_db.py script, run the following command:
```
python scripts/preprocessing/build_search_db.py
```
## Building the distribution database

The distribution database holds records of locations their songs, and their genre distributions, which will be required for the accuracy analysis. These records are created by running the build_distribution_db.py script and setting the appropriate api keys.

to run the build_distribution_db.py script, run the following command:
```
python scripts/preprocessing/build_distribution_db.py
```

## Analysis

Severa, analysis files are provided in the analysis folder. 

1. Consistency Analysis (Paper Figures 1, 3, 5)
2. Accuracy Qualitative Analysis (Paper Figures 2, 4)
3. Accuracy Quantitative Analysis (Paper Figures 6, 7)
4. Query Effect Analysis (Paper Figures 8, 9)


To run the analysis run the following command:
```
python scripts/analysis/<analysis_file>.py
```



# Quick Start:

1. Run the preprocess_data.py script to create the search and distribution databases. Be aware this step has a complexity of O(N^2) where N is the number of songs in the dataset. you might find tuning the search boundaries might be a wise investment in time.
2. Run the build_search_db.py script to create the search database (this requires api key for openai to be set in the .env file. as well as a spotify api key list in a spotify_api_keys.json file in the root directory)
3. Run the build_distribution_db.py script to create the distribution database .
4. Run the analysis files to create the analysis results.


```
pipenv install
pipenv run python scripts/data_preperation/preprocess_data.py
pipenv run python scripts/collect_data/build_search_db.py 
pipenv run python scripts/collect_data/build_location_db.py
pipenv run python scripts/analysis/analyze_consistency.py
pipenv run python scripts/analysis/analyze_accuracy_qualitative.py
pipenv run python scripts/analysis/analyze_accuracy_quantitative.py
pipenv run python scripts/analysis/analyze_query_effect.py
```


# Notes

- We are dealing with high dimensional dataspaces. This means that any operation in this space is going to be computationally expensive, and in our case also api expensive. Especially when building the distribution database, we reuqire a representative sampling of the entire dataspace. with high dimensionality, this can become very expensive. We hence recommend aiming for around 65 thousand to 1.6 million possible locations which translates to 4 to 6 bins per dimension if you have 8 dimensions. (4 bins are reccommended for datasets below 3 Million, 5 bins for datasets below 10 Million, and 6 bins for datasets below 40 Million songs)

- when building the distribution database/location database, you should aim for at least 30 thousand locations, or 25 percent of the space, whichever is larger.
