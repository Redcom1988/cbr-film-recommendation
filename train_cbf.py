import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import pickle
import os

def train_cbf():
    print("Starting Content-Based Filtering Training...")
    os.makedirs('models', exist_ok=True)
    
    df = pd.read_csv('output/content_based_training.csv')
    print(f"Loaded {len(df)} movies for CBF training.")
    
    df['genres_clean'] = df['genres_clean'].fillna('')
    df['overview_clean'] = df['overview_clean'].fillna('')
    df['text_combined'] = df['overview_clean'] + ' ' + df['genres_clean']
    
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['text_combined']).toarray().astype('float32')
    
    faiss.normalize_L2(tfidf_matrix)
    
    print("Building FAISS index...")
    d = tfidf_matrix.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(tfidf_matrix)
    
    print("Saving FAISS index and Vectorizer...")
    faiss.write_index(index, 'models/cbf_index.faiss')
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("CBF Training Completed!")

if __name__ == '__main__':
    train_cbf()
