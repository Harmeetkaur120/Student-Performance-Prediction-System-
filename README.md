# 🎬 Movie Successor: A Content-Based Recommendation Engine

**Movie Successor** is a Machine Learning-based system that suggests the top 5 movies most similar to a user's favorite film. By analyzing metadata such as genres, keywords, cast, and crew, the engine maps movies into a high-dimensional vector space and identifies "neighbors" using mathematical similarity.

## 🚀 Key Features
* **Content-Based Filtering:** Recommends movies based on their inherent characteristics rather than user ratings.
* **Natural Language Processing (NLP):** Uses **NLTK** for text normalization (Stemming) and **CountVectorizer** for text-to-numerical transformation.
* **Efficient Similarity Search:** Implements the **K-Nearest Neighbors (KNN)** algorithm with **Cosine Similarity** for high-accuracy matching.
* **Scalable Architecture:** Handles a dataset of nearly 5,000 movies with optimized preprocessing.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (KNeighborsClassifier, CountVectorizer)
* **NLP:** NLTK (PorterStemmer)

## 📊 Dataset
This project utilizes the **TMDB 5000 Movie Dataset**, which contains extensive metadata for over 4,800 films.
* **Source:** [TMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
* **Files used:** `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`.

## ⚙️ How it Works
1.  **Data Merging:** Combines movie details with cast/crew information.
2.  **Cleaning:** Extracts Directors and top 3 Actors; handles missing values.
3.  **Tag Creation:** Concatenates overview, genres, keywords, cast, and crew into a single "tags" column.
4.  **Vectorization:** Converts the tags into 5,000-dimensional vectors using **Bag-of-Words**.
5.  **Distance Calculation:** Uses **Cosine Similarity** to find the closest vectors in the feature space.

## 💻 Installation & Usage
1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/movie-successor.git
    ```
2.  Install dependencies:
    ```bash
    pip install pandas scikit-learn nltk
    ```
3.  Run the recommender:
    ```python
    python recommender.py
    ```

---

### 💡 Why I built this
As a final-year student at **GNDU**, I wanted to explore the intersection of **Linear Algebra** and **Information Retrieval**. This project serves as a practical application of how high-dimensional data can be manipulated to solve real-world problems like personalized content discovery.

