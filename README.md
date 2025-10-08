# CineMatch – Movie Recommendation System

## Project Overview

CineMatch is a Movie Recommendation Engine that suggests films to users based on their viewing preferences and ratings.

Recommendation systems are at the heart of platforms like Netflix, Amazon Prime, and YouTube. They improve user experience by surfacing relevant content and boosting engagement.

##  This project demonstrates:

- **Collaborative Filtering** → recommends based on similar users
- **Content-Based Filtering** → recommends based on movie features (genres, actors, etc.)
- **Hybrid Approach** → combines both for improved accuracy

## Tech Stack

- **Python** 
- **Pandas, NumPy** → Data preprocessing
- **Scikit-learn / Surprise / LightFM** → ML algorithms for recommendations
- **Matplotlib, Seaborn** → Data visualization
- **Streamlit** → Interactive web demo

## Dataset

This project uses the MovieLens 100k dataset for training and evaluation.

 Download here: [MovieLens Dataset](https://grouplens.org/datasets/movielens/100k/)

- **Users**: 943
- **Movies**: 1,682
- **Ratings**: 100,000

##Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/CineMatch.git
cd CineMatch
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download the MovieLens 100k dataset and extract it to the `data/` folder.

### 4. Run the Project

**Jupyter Notebook:**
```bash
jupyter notebook CineMatch.ipynb
```

**Python Script:**
```bash
python main.py
```

**Streamlit App:**
```bash
streamlit run app.py
```

## Project Workflow

1. **Data Preprocessing**
   - Clean missing values
   - Encode user & movie IDs

2. **Exploratory Data Analysis (EDA)**
   - Most popular movies
   - Ratings distribution

3. **Model Building**
   - Collaborative Filtering (User-User / Item-Item Similarity)
   - Content-Based Filtering (Genre-based)
   - Hybrid Model

4. **Evaluation**
   - RMSE, MAE for regression
   - Precision@K, Recall@K for recommendation relevance

5. **Deployment**
   - Streamlit app where user selects a movie → CineMatch recommends Top 5 similar movies 

## Results

- **Collaborative Filtering RMSE**: 0.94
- **Content-Based Accuracy**: 82%
- **Hybrid System** improved recommendations by combining strengths of both approaches

## Future Improvements

- Incorporate deep learning (Neural Collaborative Filtering, Autoencoders)
- Add real-time recommendation API
- Include user-based feedback loop for continuous improvement
- Extend to multi-modal data (text reviews, posters, trailers)

## Author

**Asiyatu Mbongwa** – Aspiring AI/ML Engineer
- Email: mbongwaasiyatu@gmail.com
- Portfolio: [your-portfolio-link]

With CineMatch, I've built a project that combines machine learning, data science, and user personalization — core skills for AI/ML roles in industry.

