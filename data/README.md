# MovieLens 100k Dataset

This folder should contain the MovieLens 100k dataset files.

## Required Files

Download the MovieLens 100k dataset from: https://grouplens.org/datasets/movielens/100k/

Extract the following files to this folder:

- `u.data` - User ratings data
- `u.item` - Movie information
- `u.user` - User information

## File Descriptions

### u.data
- Format: Tab-separated values
- Columns: user_id, item_id, rating, timestamp
- Contains: 100,000 ratings from 943 users on 1,682 movies

### u.item
- Format: Pipe-separated values
- Contains: Movie information including title, release date, genres
- Genres: Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western

### u.user
- Format: Pipe-separated values
- Columns: user_id, age, gender, occupation, zip_code
- Contains: Demographic information for 943 users

## Usage

Once the files are placed in this folder, you can run:

```bash
python main.py
```

or

```bash
streamlit run app.py
```

The system will automatically load and preprocess the data.

