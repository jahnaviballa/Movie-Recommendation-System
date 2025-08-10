
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Reader, Dataset
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# Page configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

# Cache data loading for performance
@st.cache_data
def load_data():
    data_path = '../data/'
    movies = pd.read_csv(os.path.join(data_path, 'movie.csv'))
    ratings = pd.read_csv(os.path.join(data_path, 'rating.csv'), nrows=1000000)
    tags = pd.read_csv(os.path.join(data_path, 'tag.csv'))
    return movies, ratings, tags

# Cache preprocessing
@st.cache_data
def preprocess_data(movies, ratings, tags):
    movies = movies[movies['movieId'].isin(ratings['movieId'].unique())]
    tags = tags[tags['movieId'].isin(ratings['movieId'].unique())]
    movies['genres'] = movies['genres'].replace('|', ' ', regex=True)
    movies['genres'] = movies['genres'].replace('(no genres listed)', '')
    movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    movies = movies.merge(movie_tags, on='movieId', how='left')
    movies['tag'] = movies['tag'].fillna('')
    movies['content'] = movies['genres'] + ' ' + movies['tag']
    ratings = ratings.merge(movies[['movieId', 'title', 'content']], on='movieId', how='left')
    return movies, ratings

# Cache model training
@st.cache_resource
def train_models(movies, ratings):
    # Collaborative filtering
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD(n_factors=100, random_state=42)
    svd.fit(trainset)
    
    # Content-based filtering
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return svd, tfidf, cosine_sim

# Cache sentiment analysis
@st.cache_data
def compute_sentiment(tags):
    sia = SentimentIntensityAnalyzer()
    tags['sentiment'] = tags['tag'].apply(lambda x: sia.polarity_scores(str(x))['compound'] if isinstance(x, str) else 0.0)
    return tags.groupby('movieId')['sentiment'].mean().reset_index()

# Recommendation functions
def get_collaborative_recommendations(user_id, svd, movies, n=5):
    movie_ids = movies['movieId'].unique()
    predictions = [svd.predict(user_id, movie_id) for movie_id in movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_movie_ids = [pred.iid for pred in predictions[:n]]
    return movies[movies['movieId'].isin(top_movie_ids)][['movieId', 'title', 'content']]

def get_content_recommendations(title, movies, cosine_sim, n=5):
    idx = movies[movies['title'] == title].index
    if len(idx) == 0:
        return pd.DataFrame()
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['movieId', 'title', 'content']]

def filter_by_sentiment(recommendations, movie_sentiment, min_sentiment=0.1):
    recs_with_sentiment = recommendations.merge(movie_sentiment, on='movieId', how='left')
    recs_with_sentiment['sentiment'] = recs_with_sentiment['sentiment'].fillna(0)
    return recs_with_sentiment[recs_with_sentiment['sentiment'] >= min_sentiment]

def get_hybrid_recommendations(user_id, movie_title, svd, movies, cosine_sim, movie_sentiment, n=5, use_sentiment=True):
    collab_recs = get_collaborative_recommendations(user_id, svd, movies, n=10)
    content_recs = get_content_recommendations(movie_title, movies, cosine_sim, n=10)
    combined = pd.concat([collab_recs, content_recs]).drop_duplicates(subset=['movieId'])
    if use_sentiment:
        combined = filter_by_sentiment(combined, movie_sentiment)
    return combined.head(n)

# Load and preprocess data
try:
    movies, ratings, tags = load_data()
    movies, ratings = preprocess_data(movies, ratings, tags)
    svd, tfidf, cosine_sim = train_models(movies, ratings)
    movie_sentiment = compute_sentiment(tags)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.markdown("""
Discover your next favorite movie! Enter your User ID and select a movie to get personalized recommendations 
based on collaborative filtering, content similarity, and sentiment analysis.
""")

# Sidebar for settings
st.sidebar.header("Recommendation Settings")
n_recs = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=5, step=1)
use_sentiment = st.sidebar.checkbox("Filter by Positive Sentiment", value=True, help="Only include movies with positive tag sentiment.")
min_sentiment = st.sidebar.slider("Minimum Sentiment Score", min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                                 help="Filter movies with sentiment above this threshold.")

# User input
col1, col2 = st.columns([1, 2])
with col1:
    user_id = st.number_input("Enter User ID", min_value=1, max_value=int(ratings['userId'].max()), value=1, step=1)
with col2:
    movie_title = st.selectbox("Search for a Movie", [""] + list(movies['title'].sort_values()),
                               help="Select or type to search for a movie.")

# Tabs for recommendation results
tab1, tab2, tab3 = st.tabs(["Hybrid Recommendations", "Collaborative Filtering", "Content-Based Filtering"])

# Get recommendations
if st.button("🎥 Get Recommendations", key="recommend"):
    if not movie_title:
        st.error("Please select a movie.")
    else:
        with st.spinner("Generating recommendations..."):
            # Hybrid Recommendations
            with tab1:
                st.subheader(f"Top {n_recs} Hybrid Recommendations")
                recs = get_hybrid_recommendations(user_id, movie_title, svd, movies, cosine_sim, movie_sentiment,
                                                 n=n_recs, use_sentiment=use_sentiment)
                if recs.empty:
                    st.warning("No recommendations found. Try a different movie or adjust settings.")
                else:
                    for i, row in recs.iterrows():
                        with st.expander(f"{i+1}. {row['title']}", expanded=True):
                            st.write(f"**Genres & Tags**: {row['content']}")
                            sentiment = movie_sentiment[movie_sentiment['movieId'] == row['movieId']]['sentiment'].iloc[0] if row['movieId'] in movie_sentiment['movieId'].values else 0
                            st.write(f"**Sentiment Score**: {sentiment:.2f}")
                            st.markdown(f"{'🟢 Positive' if sentiment >= 0.1 else '🟡 Neutral' if sentiment > -0.1 else '🔴 Negative'}")

            # Collaborative Filtering
            with tab2:
                st.subheader(f"Top {n_recs} Collaborative Recommendations")
                collab_recs = get_collaborative_recommendations(user_id, svd, movies, n=n_recs)
                if collab_recs.empty:
                    st.warning("No collaborative recommendations found.")
                else:
                    for i, row in collab_recs.iterrows():
                        with st.expander(f"{i+1}. {row['title']}", expanded=True):
                            st.write(f"**Genres & Tags**: {row['content']}")
                            sentiment = movie_sentiment[movie_sentiment['movieId'] == row['movieId']]['sentiment'].iloc[0] if row['movieId'] in movie_sentiment['movieId'].values else 0
                            st.write(f"**Sentiment Score**: {sentiment:.2f}")

            # Content-Based Filtering
            with tab3:
                st.subheader(f"Top {n_recs} Content-Based Recommendations")
                content_recs = get_content_recommendations(movie_title, movies, cosine_sim, n=n_recs)
                if content_recs.empty:
                    st.warning("No content-based recommendations found.")
                else:
                    for i, row in content_recs.iterrows():
                        with st.expander(f"{i+1}. {row['title']}", expanded=True):
                            st.write(f"**Genres & Tags**: {row['content']}")
                            sentiment = movie_sentiment[movie_sentiment['movieId'] == row['movieId']]['sentiment'].iloc[0] if row['movieId'] in movie_sentiment['movieId'].values else 0
                            st.write(f"**Sentiment Score**: {sentiment:.2f}")

