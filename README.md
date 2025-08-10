# 🎬 Movie Recommendation System

A **content-based movie recommender system** built with the **MovieLens 20M Dataset**, utilizing **TF-IDF vectorization** on genres and taglines to suggest similar films. The project features an interactive **Streamlit web interface** for a seamless user experience.

---

## 🚀 Features

- 🎯 Content-based recommendations using genres and tags  
- 🧼 Clean, user-friendly Streamlit interface  
- ⚡ Fast and efficient — no dependency on user ratings  
- 📈 Scalable to large datasets (e.g., MovieLens 20M)

---

## 📁 Dataset

**Source:** [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/) by GroupLens

**Required Files (place in project directory):**  
- `movie.csv`  
- `review.csv`

---

## 🛠 Installation

1. **Clone the repository** or download the project files.

2. **Install dependencies:**

```bash
pip install streamlit pandas scikit-learn
```
3.Download the dataset:

- Visit MovieLens 20M Dataset

- Extract the downloaded ZIP

- Place movies.csv and tags.csv in the same directory as app.py

## ▶️ Run the App
To launch the Streamlit app, run:

```bash
Copy
Edit
streamlit run app.py
```
Then open the provided localhost URL in your browser.

## 🧠 How It Works

1. Loads and merges metadata from movies.csv and tags.csv

2. Aggregates tags and genres per movie into a unified text field

3. Transforms text into numerical vectors using TF-IDF

4. Computes cosine similarity between all movie vectors

5. Recommends top N similar movies based on the selected title

## 📦 Project Structure
```bash
Copy
Edit
├── app.py           # Streamlit application
├── movies.csv       # Movie metadata from MovieLens
├── tags.csv         # Movie tags from MovieLens
├── README.md        # Project documentation
```
