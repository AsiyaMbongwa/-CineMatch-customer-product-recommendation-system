"""
CineMatch - Interactive Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import re
import requests
warnings.filterwarnings('ignore')



# Import our recommendation systems
from data_loader import MovieLensDataLoader
from collaborative_filtering import CollaborativeFiltering
from content_based_filtering import ContentBasedFiltering
from hybrid_recommender import HybridRecommender

# Page configuration
st.set_page_config(
    page_title="CineMatch - Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Neon cyberpunk theme */
    .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, rgba(176,38,255,0.25), transparent),
                    radial-gradient(1200px 800px at 100% 0%, rgba(0,255,255,0.08), transparent),
                    #090812;
        color: #EDE7FF;
    }
    .main-header {
        font-size: 3rem;
        color: #B026FF;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 10px rgba(176,38,255,0.8), 0 0 20px rgba(176,38,255,0.5);
    }
    .sub-header {
        font-size: 1.4rem;
        color: #CBA6FF;
        margin-top: 2rem;
        margin-bottom: 1rem;
        text-shadow: 0 0 6px rgba(203,166,255,0.6);
    }
    .metric-card {
        background: rgba(23, 15, 37, 0.7);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(176,38,255,0.35);
        box-shadow: 0 0 12px rgba(176,38,255,0.2), inset 0 0 20px rgba(176,38,255,0.06);
    }
    .recommendation-card {
        background: linear-gradient(180deg, rgba(19,14,29,0.9) 0%, rgba(12,10,20,0.9) 100%);
        padding: 0.8rem;
        border-radius: 12px;
        border: 1px solid rgba(176,38,255,0.35);
        box-shadow: 0 0 16px rgba(176,38,255,0.25);
        margin-bottom: 12px;
    }
    .reco-title { color:#EDE7FF; font-weight:600; }
    .reco-sub { color:#A18CD1; font-size:0.9rem; }
    .stButton>button {
        background: linear-gradient(90deg,#B026FF,#6C2BD9);
        color: white; border: 0; border-radius: 10px;
        box-shadow: 0 0 12px rgba(176,38,255,0.5);
    }
    .stSlider>div>div>div>div { background: linear-gradient(90deg,#B026FF,#6C2BD9) !important; }
    img.poster { border-radius: 8px; border: 1px solid rgba(176,38,255,0.35); box-shadow: 0 0 12px rgba(176,38,255,0.25); }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_movie_poster(title: str, year: int | None = None) -> str:
    """Return a poster URL for the given movie title using OMDb if available, else a placeholder."""
    # Priority: session_state -> secrets -> env var
    api_key = None
    try:
        api_key = st.session_state.get("OMDB_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        api_key = None
    if not api_key:
        try:
            api_key = st.secrets.get("OMDB_API_KEY")  # type: ignore[attr-defined]
        except Exception:
            api_key = None
    if not api_key:
        api_key = os.environ.get("OMDB_API_KEY")
    
    clean_title = re.sub(r"\s*\(\d{4}\)$", "", str(title or "")).strip()
    if not clean_title:
        return "https://dummyimage.com/200x300/0f0d1a/b026ff.png&text=No+Image"
    if api_key:
        try:
            params = {"t": clean_title, "apikey": api_key}
            if year:
                params["y"] = str(year)
            r = requests.get("https://www.omdbapi.com/", params=params, timeout=8)
            if r.ok:
                data = r.json()
                poster = data.get("Poster")
                if poster and poster != "N/A":
                    return poster
        except Exception:
            pass
    from requests.utils import quote
    return f"https://dummyimage.com/240x360/140f22/b026ff.png&text={quote(clean_title[:24])}"

@st.cache_data
def load_data():
    """Load and cache the MovieLens dataset"""
    try:
        data_loader = MovieLensDataLoader("data")
        ratings_df, movies_df, users_df = data_loader.load_data()
        
        if ratings_df is None:
            return None, None, None
        
        # Preprocess data
        ratings_df, movies_df, users_df = data_loader.preprocess_data()
        
        return ratings_df, movies_df, users_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_resource
def initialize_systems(ratings_df, movies_df):
    """Initialize recommendation systems"""
    try:
        # Collaborative Filtering
        cf_system = CollaborativeFiltering(ratings_df, movies_df)
        cf_system.create_user_movie_matrix()
        
        # Content-Based Filtering
        cb_system = ContentBasedFiltering(ratings_df, movies_df)
        cb_system.extract_movie_features()
        cb_system.calculate_movie_similarity()
        
        # Hybrid System
        hybrid_system = HybridRecommender(ratings_df, movies_df)
        
        return cf_system, cb_system, hybrid_system
    except Exception as e:
        st.error(f"Error initializing systems: {e}")
        return None, None, None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">CineMatch</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Movie Recommendation System</h2>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading MovieLens dataset..."):
        ratings_df, movies_df, users_df = load_data()
    
    if ratings_df is None:
        st.error("""
        **Dataset not found!**
        
        Please download the MovieLens 100k dataset and extract it to the `data/` folder.
        
        Download from: https://grouplens.org/datasets/movielens/100k/
        """)
        return
    
    # Initialize systems
    with st.spinner("Initializing recommendation systems..."):
        cf_system, cb_system, hybrid_system = initialize_systems(ratings_df, movies_df)
    
    if cf_system is None:
        st.error("Failed to initialize recommendation systems")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Get Recommendations", "Data Analysis", "Movie Explorer", "System Settings"]
    )
    
    # Main content based on selected page
    if page == "Home":
        show_home_page(ratings_df, movies_df, users_df)
    elif page == "Get Recommendations":
        show_recommendations_page(cf_system, cb_system, hybrid_system, movies_df)
    elif page == "Data Analysis":
        show_analysis_page(ratings_df, movies_df, users_df)
    elif page == "Movie Explorer":
        show_movie_explorer_page(cb_system, movies_df)
    elif page == "System Settings":
        show_settings_page(hybrid_system)

def show_home_page(ratings_df, movies_df, users_df):
    """Display the home page"""
    
    # Featured covers (hero grid)
    st.markdown('<div class="sub-header">Featured Covers</div>', unsafe_allow_html=True)
    pop = ratings_df.groupby('item_id').size().reset_index(name='rating_count')
    pop = pop.merge(movies_df[['movie_id','title','year']], left_on='item_id', right_on='movie_id')
    pop = pop.sort_values('rating_count', ascending=False).head(15)
    cols = st.columns(5)
    for idx, (_, row) in enumerate(pop.iterrows()):
        col = cols[idx % 5]
        with col:
            poster = get_movie_poster(str(row['title']), int(row['year']) if pd.notna(row.get('year')) else None)
            st.image(poster, caption=row['title'], width=200)
    
    st.markdown('<div class="sub-header">Dataset Overview</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", f"{len(users_df):,}")
    
    with col2:
        st.metric("Total Movies", f"{len(movies_df):,}")
    
    with col3:
        st.metric("Total Ratings", f"{len(ratings_df):,}")
    
    with col4:
        avg_rating = ratings_df['rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f}")
    
    # Rating distribution
    st.markdown('<div class="sub-header">Rating Distribution</div>', unsafe_allow_html=True)
    
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    fig = px.bar(
        x=rating_counts.index, 
        y=rating_counts.values,
        title="Distribution of Movie Ratings",
        labels={'x': 'Rating', 'y': 'Count'},
        color=rating_counts.values,
        color_continuous_scale='viridis'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Popular movies
    st.markdown('<div class="sub-header">Most Popular Movies</div>', unsafe_allow_html=True)
    
    popular_movies = ratings_df.groupby('item_id').size().reset_index(name='rating_count')
    popular_movies = popular_movies.merge(movies_df[['movie_id', 'title']], left_on='item_id', right_on='movie_id')
    popular_movies = popular_movies.sort_values('rating_count', ascending=False).head(10)
    
    for i, (_, row) in enumerate(popular_movies.iterrows(), 1):
        with st.container():
            st.markdown(f"""
            <div class="recommendation-card">
                <strong>{i}. {row['title']}</strong><br>
                <span style="color: #666;">{row['rating_count']} ratings</span>
            </div>
            """, unsafe_allow_html=True)
    
    # System information
    st.markdown('<div class="sub-header">Recommendation Systems</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Collaborative Filtering</h4>
            <p>Recommends movies based on similar users' preferences</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>Content-Based Filtering</h4>
            <p>Recommends movies based on genre and content similarity</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>Hybrid System</h4>
            <p>Combines both approaches for improved accuracy</p>
        </div>
        """, unsafe_allow_html=True)

def show_recommendations_page(cf_system, cb_system, hybrid_system, movies_df):
    """Display the recommendations page"""
    
    st.markdown('<div class="sub-header">Get Movie Recommendations</div>', unsafe_allow_html=True)
    
    # User selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_id = st.selectbox(
            "Select a user:",
            options=cf_system.ratings_df['user_id'].unique(),
            format_func=lambda x: f"User {x}"
        )
    
    with col2:
        n_recommendations = st.slider("Number of recommendations:", 3, 10, 5)
    
    # Show user's current ratings
    user_ratings = cf_system.ratings_df[cf_system.ratings_df['user_id'] == user_id]
    
    st.markdown(f"**User {user_id} has rated {len(user_ratings)} movies**")
    
    if len(user_ratings) > 0:
        # Display user's top rated movies
        top_rated = user_ratings.nlargest(5, 'rating')
        st.markdown("**User's Top Rated Movies:**")
        
        for _, rating in top_rated.iterrows():
            movie_title = cf_system.get_movie_title(rating['item_id'])
            st.write(f"{movie_title} - {rating['rating']} stars")
    
    # Recommendation methods
    st.markdown('<div class="sub-header">Recommendations</div>', unsafe_allow_html=True)
    
    # Tabs for different recommendation methods
    tab1, tab2, tab3, tab4 = st.tabs(["Collaborative", "Content-Based", "Hybrid", "Compare All"])
    
    with tab1:
        st.markdown("**User-Based Collaborative Filtering**")
        try:
            cf_recommendations = cf_system.user_based_collaborative_filtering(user_id, n_recommendations)
            display_recommendations(cf_recommendations, cf_system)
        except Exception as e:
            st.error(f"Error generating collaborative filtering recommendations: {e}")
    
    with tab2:
        st.markdown("**Content-Based Filtering**")
        try:
            cb_recommendations = cb_system.get_content_based_recommendations(user_id, n_recommendations)
            display_recommendations(cb_recommendations, cb_system)
        except Exception as e:
            st.error(f"Error generating content-based recommendations: {e}")
    
    with tab3:
        st.markdown("**Hybrid Recommendations**")
        
        # Hybrid method selection
        hybrid_method = st.selectbox(
            "Hybrid Method:",
            ["weighted", "switching", "mixed"],
            format_func=lambda x: x.capitalize()
        )
        
        try:
            hybrid_recommendations = hybrid_system.get_hybrid_recommendations(user_id, n_recommendations, hybrid_method)
            display_recommendations(hybrid_recommendations, hybrid_system)
        except Exception as e:
            st.error(f"Error generating hybrid recommendations: {e}")
    
    with tab4:
        st.markdown("**Compare All Methods**")
        
        try:
            # Get recommendations from all methods
            cf_recs = cf_system.user_based_collaborative_filtering(user_id, n_recommendations)
            cb_recs = cb_system.get_content_based_recommendations(user_id, n_recommendations)
            hybrid_recs = hybrid_system.get_hybrid_recommendations(user_id, n_recommendations, 'weighted')
            
            # Create comparison table
            comparison_data = []
            
            for i in range(n_recommendations):
                row = {"Rank": i + 1}
                
                if i < len(cf_recs):
                    row["Collaborative"] = cf_system.get_movie_title(cf_recs[i][0])
                else:
                    row["Collaborative"] = "-"
                
                if i < len(cb_recs):
                    row["Content-Based"] = cb_system.get_movie_title(cb_recs[i][0])
                else:
                    row["Content-Based"] = "-"
                
                if i < len(hybrid_recs):
                    row["Hybrid"] = hybrid_system.get_movie_title(hybrid_recs[i][0])
                else:
                    row["Hybrid"] = "-"
                
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error comparing recommendations: {e}")

def show_analysis_page(ratings_df, movies_df, users_df):
    """Display the data analysis page"""
    
    st.markdown('<div class="sub-header">Data Analysis</div>', unsafe_allow_html=True)
    
    # User analysis
    st.markdown("**User Analysis**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Ratings per user distribution
        user_rating_counts = ratings_df['user_id'].value_counts()
        fig1 = px.histogram(
            x=user_rating_counts.values,
            title="Distribution of Ratings per User",
            labels={'x': 'Number of Ratings', 'y': 'Number of Users'},
            nbins=30
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # User age distribution
        fig2 = px.histogram(
            users_df,
            x='age',
            title="User Age Distribution",
            labels={'age': 'Age', 'count': 'Number of Users'},
            nbins=20
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Movie analysis
    st.markdown("**Movie Analysis**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Ratings per movie distribution
        movie_rating_counts = ratings_df['item_id'].value_counts()
        fig3 = px.histogram(
            x=movie_rating_counts.values,
            title="Distribution of Ratings per Movie",
            labels={'x': 'Number of Ratings', 'y': 'Number of Movies'},
            nbins=30
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Average rating by movie
        movie_avg_ratings = ratings_df.groupby('item_id')['rating'].mean()
        fig4 = px.histogram(
            x=movie_avg_ratings.values,
            title="Distribution of Average Movie Ratings",
            labels={'x': 'Average Rating', 'y': 'Number of Movies'},
            nbins=20
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Genre analysis
    st.markdown("**Genre Analysis**")
    
    # Get genre columns
    genre_columns = ['action', 'adventure', 'animation', 'children', 'comedy', 'crime',
                    'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 'musical',
                    'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']
    
    # Calculate genre popularity
    genre_counts = {}
    for genre in genre_columns:
        if genre in movies_df.columns:
            genre_counts[genre] = movies_df[genre].sum()
    
    genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
    genre_df = genre_df.sort_values('Count', ascending=True)
    
    fig5 = px.bar(
        genre_df,
        x='Count',
        y='Genre',
        orientation='h',
        title="Number of Movies by Genre",
        labels={'Count': 'Number of Movies', 'Genre': 'Genre'}
    )
    st.plotly_chart(fig5, use_container_width=True)

def show_movie_explorer_page(cb_system, movies_df):
    """Display the movie explorer page"""
    
    st.markdown('<div class="sub-header">Movie Explorer</div>', unsafe_allow_html=True)
    
    # Movie search
    movie_search = st.text_input("Search for a movie:", placeholder="Enter movie title...")
    
    if movie_search:
        # Filter movies based on search
        filtered_movies = movies_df[movies_df['title'].str.contains(movie_search, case=False, na=False)]
        
        if len(filtered_movies) > 0:
            selected_movie_id = st.selectbox(
                "Select a movie:",
                options=filtered_movies['movie_id'],
                format_func=lambda x: movies_df[movies_df['movie_id'] == x]['title'].iloc[0]
            )
        else:
            st.warning("No movies found matching your search.")
            selected_movie_id = None
    else:
        # Show random movie
        selected_movie_id = st.selectbox(
            "Select a movie:",
            options=movies_df['movie_id'].sample(20),
            format_func=lambda x: movies_df[movies_df['movie_id'] == x]['title'].iloc[0]
        )
    
    if selected_movie_id:
        # Display movie information
        movie_info = movies_df[movies_df['movie_id'] == selected_movie_id].iloc[0]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**{movie_info['title']}**")
            
            # Get genres
            genre_columns = ['action', 'adventure', 'animation', 'children', 'comedy', 'crime',
                           'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 'musical',
                           'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']
            
            genres = [genre for genre in genre_columns if movie_info.get(genre, 0) == 1]
            
            if genres:
                st.markdown(f"**Genres:** {', '.join(genres)}")
            
            if 'year' in movie_info and pd.notna(movie_info['year']):
                st.markdown(f"**Year:** {int(movie_info['year'])}")
        
        with col2:
            # Similar movies
            st.markdown("**Similar Movies:**")
            
            try:
                similar_movies = cb_system.get_similar_movies(selected_movie_id, 5)
                
                for i, (movie_id, similarity) in enumerate(similar_movies, 1):
                    movie_title = cb_system.get_movie_title(movie_id)
                    st.write(f"{i}. {movie_title}")
                    st.write(f"   Similarity: {similarity:.3f}")
            
            except Exception as e:
                st.error(f"Error finding similar movies: {e}")

def show_settings_page(hybrid_system):
    """Display the system settings page"""
    
    st.markdown('<div class="sub-header">System Settings</div>', unsafe_allow_html=True)
    
    # OMDb API Key input
    st.markdown("**OMDb API Key**")
    api_key_input = st.text_input("Enter your OMDb API Key:", type="password")
    if api_key_input:
        st.session_state["OMDB_API_KEY"] = api_key_input
        st.success("OMDb API Key saved for this session.")
    
    # Hybrid system weights
    st.markdown("**Hybrid System Configuration**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cf_weight = st.slider(
            "Collaborative Filtering Weight:",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1
        )
    
    with col2:
        cb_weight = st.slider(
            "Content-Based Filtering Weight:",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1
        )
    
    if st.button("Update Weights"):
        if abs(cf_weight + cb_weight - 1.0) < 0.01:
            hybrid_system.set_weights(cf_weight, cb_weight)
            st.success("Weights updated successfully!")
        else:
            st.error("Weights must sum to 1.0")
    
    # System information
    st.markdown("**System Information**")
    
    try:
        system_info = hybrid_system.get_system_info()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Users", f"{system_info['total_users']:,}")
            st.metric("Total Movies", f"{system_info['total_movies']:,}")
        
        with col2:
            st.metric("Total Ratings", f"{system_info['total_ratings']:,}")
            st.metric("Avg Ratings/User", f"{system_info['avg_ratings_per_user']:.1f}")
        
        with col3:
            st.metric("Avg Ratings/Movie", f"{system_info['avg_ratings_per_movie']:.1f}")
            st.metric("CF Weight", f"{system_info['cf_weight']:.1f}")
    
    except Exception as e:
        st.error(f"Error getting system information: {e}")

def display_recommendations(recommendations, system):
    """Display recommendations in a formatted way"""
    
    if not recommendations:
        st.warning("No recommendations available.")
        return
    
    for i, (movie_id, score) in enumerate(recommendations, 1):
        movie_title = system.get_movie_title(movie_id)
        
        with st.container():
            st.markdown(f"""
            <div class="recommendation-card">
                <strong>{i}. {movie_title}</strong><br>
                <span style="color: #666;">Score: {score:.3f}</span>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()