"""
train_cf.py
Phase 3: Latih SVD (Collaborative Filtering).
  - Dataset: collaborative_training.csv (671 user, 6003 film, MovieLens Small)
  - CF hanya digunakan sebagai komponen pendukung ranking (30%)
Jalankan: python train_cf.py
"""

import os
import pickle
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# ─── Paths ───────────────────────────────────────────────────────────────────
CF_CSV = "output/collaborative_training.csv"
MODELS_DIR = "models"
CF_MODEL = f"{MODELS_DIR}/cf_model.pkl"


def train_svd(cf_csv: str = CF_CSV) -> tuple:
    print(f"[3] Melatih SVD dari {cf_csv} ...")
    df = pd.read_csv(cf_csv)

    print(
        f"    Data: {len(df)} rating | {df['userId'].nunique()} user | {df['movieId'].nunique()} film"
    )

    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()

    model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    model.fit(trainset)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(CF_MODEL, "wb") as f:
        pickle.dump(model, f)

    print(f"[OK] CF model tersimpan: {CF_MODEL}")

    # Quick cross-val sanity check (3-fold, hanya saat run langsung)
    print("    Cross-validation 3-fold (RMSE) ...")
    cv_data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
    cv_model = SVD(n_factors=50, n_epochs=20, random_state=42)
    results = cross_validate(cv_model, cv_data, measures=["RMSE"], cv=3, verbose=False)
    rmse_mean = results["test_rmse"].mean()
    print(f"    RMSE rata-rata: {rmse_mean:.4f}")

    return model, df


if __name__ == "__main__":
    model, df = train_svd()

    # Isi predicted_ratings dengan movieId (untuk keperluan hitung jumlah film)
    import sqlite3

    conn = sqlite3.connect("models/cases.db")
    cur = conn.cursor()
    mids = df["movieId"].unique()
    for mid in mids:
        cur.execute(
            "INSERT OR IGNORE INTO predicted_ratings (userId, movieId, predicted_rating) VALUES (0, ?, 0)",
            (int(mid),),
        )
    conn.commit()
    conn.close()
    print(f"[OK] predicted_ratings: {len(mids)} movieId dimasukkan.")

    # Demo prediksi satu pasang user-film
    sample_uid = int(df["userId"].iloc[0])
    sample_mid = int(df["movieId"].iloc[0])
    pred = model.predict(sample_uid, sample_mid)
    print(f"\nContoh prediksi: user={sample_uid}, movie={sample_mid} => {pred.est:.3f}")
    print("\n=== CF training selesai ===")
