import pandas as pd
from surprise import Dataset, Reader, SVD
import pickle
import os

def train_cf():
    print("Starting Collaborative Filtering Training...")
    os.makedirs('models', exist_ok=True)
    
    df = pd.read_csv('output/collaborative_training.csv')
    print(f"Loaded {len(df)} ratings for CF training.")
    
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    
    print("Building trainset...")
    trainset = data.build_full_trainset()
    
    print("Training SVD Model (50 factors)...")
    algo = SVD(n_factors=50, n_epochs=20, random_state=42)
    algo.fit(trainset)
    
    print("Saving SVD Model...")
    with open('models/cf_model.pkl', 'wb') as f:
        pickle.dump(algo, f)
        
    print("CF Training Completed!")

if __name__ == '__main__':
    train_cf()
