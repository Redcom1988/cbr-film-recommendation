"""
train_cbf.py
Phase 2: TF-IDF + dua FAISS index.
  - movie_index.faiss  → kemiripan antar film (untuk movie_similarity)
  - case_index.faiss   → kemiripan antar kasus (untuk Retrieve CBR)
Jalankan: python train_cbf.py
"""

import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── Paths ───────────────────────────────────────────────────────────────────
DB_PATH = "models/cases.db"
CB_CSV = "output/content_based_training.csv"
MODELS_DIR = "models"

MOVIE_INDEX = f"{MODELS_DIR}/movie_index.faiss"
CASE_INDEX = f"{MODELS_DIR}/case_index.faiss"
VECTORIZER_PKL = f"{MODELS_DIR}/tfidf_vectorizer.pkl"
MOVIE_IDS_PKL = f"{MODELS_DIR}/movie_ids.pkl"
CASE_IDS_PKL = f"{MODELS_DIR}/case_ids.pkl"


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalisasi baris agar inner-product = cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _build_text(genres: str, overview: str, title: str = "") -> str:
    """
    Gabung genre (3×) + title + overview untuk TF-IDF.
    Urutan konsisten dengan case embedding di recommend.py.
    """
    g = str(genres).strip() if pd.notna(genres) else ""
    t = str(title).strip() if pd.notna(title) else ""
    o = str(overview).strip() if pd.notna(overview) else ""
    return (g + " ") * 3 + t + " " + o


# ─── Phase 2a: Movie Index ────────────────────────────────────────────────────


def build_movie_index() -> tuple[TfidfVectorizer, list[int]]:
    """
    Fit TF-IDF dari content_based_training.csv → buat movie_index.faiss.
    Return: (vectorizer, movie_ids)
    """
    print("[2a] Membangun movie_index.faiss ...")
    df = pd.read_csv(CB_CSV)

    df["text_combined"] = df.apply(
        lambda r: _build_text(
            r.get("genres_clean", ""),
            r.get("overview_clean", ""),
            r.get("title", ""),
        ),
        axis=1,
    )

    movie_ids = df["movieId"].astype(int).tolist()
    texts = df["text_combined"].tolist()

    # Fit TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
    tfidf_mat = vectorizer.fit_transform(texts).toarray().astype("float32")
    tfidf_norm = _normalize(tfidf_mat)

    # Build FAISS IndexFlatIP (cosine setelah L2-norm)
    dim = tfidf_norm.shape[1]
    movie_index = faiss.IndexFlatIP(dim)
    movie_index.add(tfidf_norm)

    # Simpan
    os.makedirs(MODELS_DIR, exist_ok=True)
    faiss.write_index(movie_index, MOVIE_INDEX)
    with open(VECTORIZER_PKL, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MOVIE_IDS_PKL, "wb") as f:
        pickle.dump(movie_ids, f)

    # Update tfidf_vector_idx di SQLite
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for idx, mid in enumerate(movie_ids):
        cur.execute(
            "UPDATE films SET tfidf_vector_idx = ? WHERE movieId = ?", (idx, mid)
        )
    conn.commit()
    conn.close()

    print(f"[OK] movie_index.faiss: {len(movie_ids)} film, dim={dim}")
    return vectorizer, movie_ids


# ─── Phase 2b: Case Index ─────────────────────────────────────────────────────


def build_case_index(vectorizer: TfidfVectorizer) -> list[int]:
    """
    Ambil semua kasus dari retained_cases → buat case_index.faiss.
    Case embedding = query_text + (genres_ref * 3) + title_ref
    """
    print("[2b] Membangun case_index.faiss dari seed cases ...")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT rc.case_id, rc.query_text, rc.reference_movie,
               f.title, f.genres, f.overview
        FROM   retained_cases rc
        LEFT JOIN films f ON f.movieId = rc.reference_movie
        ORDER  BY rc.case_id
    """)
    cases = cur.fetchall()
    conn.close()

    if not cases:
        print("[WARN] Tidak ada kasus di DB. case_index tidak dibuat.")
        return []

    case_ids = []
    case_texts = []

    for c in cases:
        query = str(c["query_text"] or "")
        title = str(c["title"] or "")
        genres = str(c["genres"] or "")
        overview = str(c["overview"] or "")

        # Case embedding: query + weighted genres + title + overview
        text = query + " " + (genres + " ") * 3 + title + " " + overview
        case_ids.append(c["case_id"])
        case_texts.append(text)

    # Vektorisasi (transform saja, vectorizer sudah fit)
    case_mat = vectorizer.transform(case_texts).toarray().astype("float32")
    case_norm = _normalize(case_mat)

    dim = case_norm.shape[1]
    case_index = faiss.IndexFlatIP(dim)
    case_index.add(case_norm)

    faiss.write_index(case_index, CASE_INDEX)
    with open(CASE_IDS_PKL, "wb") as f:
        pickle.dump(case_ids, f)

    print(f"[OK] case_index.faiss: {len(case_ids)} kasus, dim={dim}")
    return case_ids


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    vectorizer, _ = build_movie_index()
    build_case_index(vectorizer)
    print("\n=== CBF training selesai ===")
    print(f"  Artifacts: {MOVIE_INDEX}, {CASE_INDEX}, {VECTORIZER_PKL}")
