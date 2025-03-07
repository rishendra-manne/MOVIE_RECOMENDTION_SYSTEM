# Movie Recommendation System

A content-based movie recommendation system built using the Bag of Words algorithm. This system suggests similar movies based on the content and features of a selected movie.

## Overview

This project implements a movie recommendation engine that:
- Uses textual movie data from TMDB datasets
- Processes and transforms movie metadata into feature vectors
- Calculates similarities between movies using cosine similarity
- Presents recommendations through an interactive Streamlit web interface
- Displays movie posters fetched from TMDB API

## Dataset

The system uses two TMDB (The Movie Database) datasets:
1. `tmdb_5000_credits.csv` - Contains cast and crew information
2. `tmdb_5000_movies.csv` - Contains movie metadata like overview, genres, keywords, etc.

The datasets are merged, cleaned, and processed to extract relevant features for the recommendation algorithm.

## Features

- **Interactive UI**: Select any movie from a dropdown menu to get recommendations
- **Visual Results**: View movie posters along with titles
- **Content-Based Filtering**: Recommendations based on movie content, not user ratings
- **Real-time API Integration**: Fetches up-to-date movie posters from TMDB API

## Installation

### Prerequisites
- Python 3.7+
- Required libraries: pandas, numpy, scikit-learn, streamlit, requests, pickle

### Setup

1. Clone the repository:
```bash
git clone https://github.com/rishendra-manne/MOVIE_RECOMMENDATION_SYSTEM.git
cd MOVIE_RECOMMENDATION_SYSTEM
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the datasets:
   - `tmdb_5000_credits.csv`
   - `tmdb_5000_movies.csv`

4. Run the data processing script to generate the pickle files by running the jupyter nootebook cells.

5. Launch the Streamlit app:
```bash
streamlit run app.py
```

## Technical Approach

### Data Processing
1. Load and merge the two TMDB datasets
2. Clean the data by handling missing values and removing duplicates
3. Extract relevant features from JSON-formatted columns (cast, crew, genres, keywords)
4. Create a "tags" column that combines various textual features (overview, genres, keywords, cast, crew)
5. Convert the textual data to vector form using Count Vectorization
6. Calculate cosine similarity between all movie vectors

### Recommendation Algorithm
1. Find the index of the selected movie
2. Get similarity scores between the selected movie and all other movies
3. Sort movies by similarity score (descending)
4. Return the top 5 most similar movies

### Web Interface
The Streamlit app provides a user-friendly interface to interact with the recommendation system:
1. Select a movie from the dropdown
2. Click the "Show Recommendation" button
3. View 5 recommended movies with their posters and titles

## Data Processing Code Example

Here's a snippet of how the data is processed:

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import ast

# Load data
credits = pd.read_csv("datasets/tmdb_5000_credits.csv")
movies = pd.read_csv("datasets/tmdb_5000_movies.csv")

# Rename columns for merging
credits.columns = ['id', 'title', 'cast', 'crew']

# Merge datasets
movies = movies.merge(credits, on='id')

# Extract relevant features
# [Feature extraction code for cast, crew, genres, etc.]

# Create tags
movies['tags'] = movies['overview'] + movies['genres_list'] + movies['keywords_list'] + \
                 movies['cast_list'] + movies['crew_list']

# Create count vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Calculate similarity
similarity = cosine_similarity(vectors)

# Save processed data
pickle.dump(movies.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and go to `http://localhost:8501`

3. Select a movie from the dropdown menu

4. Click the "Show Recommendation" button to see similar movies

## API Integration

The application uses TMDB API to fetch movie poster images. An API key is required for this functionality. The key is already included in the code but can be replaced with your own.

## Future Improvements

- Add user-based collaborative filtering
- Implement hybrid recommendation system
- Add more movie details and streaming availability
- Include search functionality for movies
- Improve UI/UX design

## Credits

- Developed by RISHENDRA MANNE
- Data provided by TMDB (The Movie Database)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
