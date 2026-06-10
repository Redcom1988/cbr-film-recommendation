import sqlite3
import pandas as pd
import json
import os

def build_casebase():
    print("Starting Case Base Construction (SQLite)...")
    os.makedirs('models', exist_ok=True)
    
    db_path = 'models/cases.db'
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE films (
            movieId INTEGER PRIMARY KEY,
            title TEXT,
            genres TEXT,
            overview TEXT,
            vote_average REAL,
            tfidf_vector_idx INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE index_file (
            movieId INTEGER PRIMARY KEY,
            tfidf_norm REAL,
            genre_vector TEXT,
            language TEXT,
            vote_avg REAL,
            is_validated BOOLEAN
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE retained_cases (
            case_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT,
            recommended_ids TEXT,
            accepted_ids TEXT,
            rejected_ids TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    print("Loading data...")
    cb_df = pd.read_csv('output/content_based_training.csv')
    cb_df['tfidf_vector_idx'] = cb_df.index
    
    movies_df = pd.read_csv('output/movies_clean.csv')
    
    merged_df = pd.merge(cb_df, movies_df[['movieId', 'vote_average', 'original_language']], on='movieId', how='left')
    merged_df = merged_df.drop_duplicates(subset=['movieId'])
    
    merged_df['vote_average'] = merged_df['vote_average'].fillna(0.0)
    merged_df['original_language'] = merged_df['original_language'].fillna('en')
    merged_df['genres_clean'] = merged_df['genres_clean'].fillna('')
    merged_df['overview_clean'] = merged_df['overview_clean'].fillna('')
    
    print(f"Inserting {len(merged_df)} records into SQLite...")
    
    films_data = []
    index_data = []
    
    for _, row in merged_df.iterrows():
        m_id = int(row['movieId'])
        title = row['title']
        genres = row['genres_clean']
        overview = row['overview_clean']
        vote_avg = float(row['vote_average'])
        idx = int(row['tfidf_vector_idx'])
        
        films_data.append((m_id, title, genres, overview, vote_avg, idx))
        
        norm = 1.0 if len(overview.split()) > 10 else 0.0
        genre_list = genres.split() if genres else []
        genre_json = json.dumps(genre_list)
        lang = row['original_language']
        is_validated = True
        
        index_data.append((m_id, norm, genre_json, lang, vote_avg, is_validated))
        
    cursor.executemany('''
        INSERT INTO films (movieId, title, genres, overview, vote_average, tfidf_vector_idx)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', films_data)
    
    cursor.executemany('''
        INSERT INTO index_file (movieId, tfidf_norm, genre_vector, language, vote_avg, is_validated)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', index_data)
    
    conn.commit()
    conn.close()
    print("Case Base built successfully!")

if __name__ == '__main__':
    build_casebase()
