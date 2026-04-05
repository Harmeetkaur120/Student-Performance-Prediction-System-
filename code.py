import pandas as pd
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# 1. Load and Merge Data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
df = movies.merge(credits, on='title')

# 2. Select Relevant Columns
df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
df.dropna(inplace=True)

# 3. Helper function to parse JSON-like strings (Genres, Keywords)
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

df['genres'] = df['genres'].apply(convert)
df['keywords'] = df['keywords'].apply(convert)

# 4. Extract Top 3 Actors and Director
def convert_cast(obj):
    L = []
    for i in ast.literal_eval(obj)[:3]:
        L.append(i['name'])
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

df['cast'] = df['cast'].apply(convert_cast)
df['crew'] = df['crew'].apply(fetch_director)

# 5. Remove Spaces (Transform "Sam Worthington" to "SamWorthington")
def collapse(L):
    return [i.replace(" ", "") for i in L]

df['cast'] = df['cast'].apply(collapse)
df['crew'] = df['crew'].apply(collapse)
df['genres'] = df['genres'].apply(collapse)
df['keywords'] = df['keywords'].apply(collapse)

# 6. Create the "Tags" Column
df['overview'] = df['overview'].apply(lambda x: x.split())
df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
new_df = df[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# 7. Apply NLTK Stemming (e.g., 'action-packed' -> 'action')
ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

new_df['tags'] = new_df['tags'].apply(stem)

# 8. Vectorization (CountVectorizer)
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# 9. Initialize KNN with Cosine Similarity
# Metric='cosine' makes the model find the angle between movie vectors
model = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
model.fit(vectors)

# 10. The Recommendation Function
def recommend(movie_name):
    # Find the index of the movie
    movie_index = new_df[new_df['title'] == movie_name].index[0]
    
    # Calculate distances and find nearest neighbors
    distances, indices = model.kneighbors(vectors[movie_index].reshape(1, -1))
    
    print(f"Movies similar to '{movie_name}':")
    for i in indices[0][1:]: # Skip index 0 as it's the movie itself
        print(new_df.iloc[i].title)

# TEST: Get recommendations for 'Avatar'
recommend('Avatar')
