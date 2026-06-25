"""
build_casebase.py
Phase 1: Inisialisasi database, isi tabel films, buat 500 seed cases.
Jalankan: python build_casebase.py
"""

import os
import sqlite3
import json
import random
import pandas as pd
from collections import defaultdict

# ─── Paths ───────────────────────────────────────────────────────────────────
DB_PATH = "models/cases.db"
MOVIES_CLEAN = "output/movies_clean.csv"
CB_CSV = "output/content_based_training.csv"
CF_CSV = "output/collaborative_training.csv"
N_SEED_CASES = 3278


# ─── DB helpers ──────────────────────────────────────────────────────────────


def get_db_connection(db_path=DB_PATH):
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(conn):
    """Buat skema tabel jika belum ada."""
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS films (
        movieId          INTEGER PRIMARY KEY,
        title            TEXT    NOT NULL,
        overview         TEXT,
        genres           TEXT,
        vote_average     REAL,
        tfidf_vector_idx INTEGER
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS retained_cases (
        case_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id          INTEGER,
        query_text       TEXT    NOT NULL,
        reference_movie  INTEGER,
        recommended_ids  TEXT    NOT NULL DEFAULT '[]',
        accepted_ids     TEXT    NOT NULL DEFAULT '[]',
        rejected_ids     TEXT    NOT NULL DEFAULT '[]',
        from_case_id     INTEGER,
        timestamp        TEXT    DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(reference_movie) REFERENCES films(movieId) ON DELETE SET NULL,
        FOREIGN KEY(from_case_id)    REFERENCES retained_cases(case_id) ON DELETE SET NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS case_similarity_cache (
        case_a    INTEGER,
        case_b    INTEGER,
        similarity REAL NOT NULL,
        PRIMARY KEY (case_a, case_b),
        FOREIGN KEY(case_a) REFERENCES retained_cases(case_id) ON DELETE CASCADE,
        FOREIGN KEY(case_b) REFERENCES retained_cases(case_id) ON DELETE CASCADE
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS predicted_ratings (
        userId           INTEGER NOT NULL,
        movieId          INTEGER NOT NULL,
        predicted_rating REAL    NOT NULL,
        PRIMARY KEY (userId, movieId),
        FOREIGN KEY(movieId) REFERENCES films(movieId) ON DELETE CASCADE
    );
    """)

    # Indeks performa
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_films_tfidf    ON films(tfidf_vector_idx);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_cases_user     ON retained_cases(user_id);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_cases_refmovie ON retained_cases(reference_movie);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_cases_from     ON retained_cases(from_case_id);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_pred_user      ON predicted_ratings(userId);"
    )

    conn.commit()
    print("[OK] Skema database siap.")

    # ── Migrasi: tambah kolom is_seed jika belum ada
    try:
        cur.execute("ALTER TABLE retained_cases ADD COLUMN is_seed INTEGER DEFAULT 0")
        print("[OK] Kolom is_seed ditambahkan ke retained_cases.")
    except sqlite3.OperationalError:
        pass  # sudah ada


# ─── Phase 1a: Populate films ─────────────────────────────────────────────────


def populate_films(conn, movies_csv=MOVIES_CLEAN):
    """Isi tabel films dari movies_clean.csv."""
    print(f"[1a] Mengisi tabel films dari {movies_csv} ...")
    df = pd.read_csv(movies_csv)

    # Pastikan kolom yang dibutuhkan ada
    required = {"movieId", "title"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom tidak ditemukan di {movies_csv}: {missing}")

    cur = conn.cursor()
    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        movie_id = int(row["movieId"])
        title = str(row.get("title", ""))
        overview = (
            str(row.get("overview_clean", ""))
            if pd.notna(row.get("overview_clean", ""))
            else ""
        )
        genres = str(row.get("genres", "")) if pd.notna(row.get("genres", "")) else ""
        vote_avg = (
            float(row.get("vote_average", 0.0))
            if pd.notna(row.get("vote_average", 0.0))
            else 0.0
        )

        try:
            cur.execute(
                "INSERT OR IGNORE INTO films (movieId, title, overview, genres, vote_average) "
                "VALUES (?, ?, ?, ?, ?)",
                (movie_id, title, overview, genres, vote_avg),
            )
            if cur.rowcount:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  [WARN] Skip movieId={movie_id}: {e}")

    conn.commit()
    print(f"[OK] Films: {inserted} dimasukkan, {skipped} sudah ada.")


# ─── Phase 1b: Generate seed cases ───────────────────────────────────────────


def _build_query_from_genres(genres_str: str, n_words: int = 2) -> str:
    """Buat query teks dari genre film referensi."""
    words = [w for w in genres_str.split() if len(w) > 2]
    if not words:
        return "movie"
    sample = random.sample(words, min(n_words, len(words)))
    return " ".join(sample).lower() + " movie"


def generate_seed_cases(conn, cb_csv=CB_CSV, cf_csv=CF_CSV):
    """
    Buat seed cases secara masif dari data rating tinggi (>= 4.0).
    Untuk menghindari noise, film favorit seorang user dikelompokkan berdasarkan genre utamanya.
    Setiap kelompok (user, genre) yang memiliki >= 2 film akan menjadi 1 kasus.
    """
    print(f"[1b] Mengekstrak seed cases dari seluruh dataset (massive scale) ...")

    cb_df = pd.read_csv(cb_csv)
    cf_df = pd.read_csv(cf_csv)

    # Lookup: movieId → primary genre
    movie_primary_genre: dict[int, str] = {}
    for _, row in cb_df.iterrows():
        movie_genres[int(row["movieId"])] = str(row.get("genres_clean", ""))

    # Kumpulkan film yang disukai per user (rating >= 4.0)
    high = cf_df[cf_df["rating"] >= 4.0].copy()
    user_liked: dict[int, list[int]] = defaultdict(list)
    for _, row in high.iterrows():
        mid = int(row["movieId"])
        if mid in movie_genres and movie_genres[mid].strip():
            user_liked[int(row["userId"])].append(mid)

    eligible = [(uid, films) for uid, films in user_liked.items() if len(films) >= 3]
    if not eligible:
        print("[WARN] Tidak ada user dengan ≥ 3 film favorit. Seed cases tidak dibuat.")
        return 0

    cur = conn.cursor()
    # Bersihkan seed cases lama (baik yg is_seed=1 maupun yg user_id IS NULL)
    cur.execute(
        "DELETE FROM retained_cases WHERE is_seed = 1 OR (user_id IS NULL AND from_case_id IS NULL)"
    )
    conn.commit()

    generated = 0
    attempts = 0
    max_try = n_cases * 20

    for uid, genre_dict in user_genre_groups.items():
        for p_genre, movies in genre_dict.items():
            if len(movies) >= 2:  # Minimal 1 ref movie + 1 accepted
                # Ambil 1 film acak sebagai referensi
                ref_id = random.choice(movies)
                query = f"{p_genre} movie"
                accepted = [m for m in movies if m != ref_id]
                recommended = list(accepted)
                
                batch_data.append((
                    None,
                    query,
                    ref_id,
                    json.dumps(recommended),
                    json.dumps(accepted),
                    json.dumps([]),
                    None
                ))
                generated += 1
                
                # Insert per 10k untuk menghindari memory overload
                if len(batch_data) >= 10000:
                    cur.executemany("""
                        INSERT INTO retained_cases
                            (user_id, query_text, reference_movie,
                             recommended_ids, accepted_ids, rejected_ids, from_case_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, batch_data)
                    conn.commit()
                    batch_data = []

        # Pilih film referensi
        ref_id = random.choice(liked)
        ref_genre = movie_genres.get(ref_id, "")
        if not ref_genre.strip():
            continue

        # Query sintetis dari genre
        query = _build_query_from_genres(ref_genre)

        # Film lain yang disukai → accepted
        others = [m for m in liked if m != ref_id]
        n_accept = min(random.randint(2, 4), len(others))
        accepted = random.sample(others, n_accept)
        recommended = list(accepted)  # seed: recommended = accepted, tanpa rejected

        cur.execute(
            """
            INSERT INTO retained_cases
                (user_id, query_text, reference_movie,
                 recommended_ids, accepted_ids, rejected_ids, from_case_id, is_seed)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
        """,
            (
                uid,
                query,
                ref_id,
                json.dumps(recommended),
                json.dumps(accepted),
                json.dumps([]),
                None,
            ),
        )
        generated += 1

    print(f"[OK] Seed cases dibuat secara masif: {generated} kasus.")
    return generated


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    conn = get_db_connection(DB_PATH)
    initialize_database(conn)
    populate_films(conn)
    generate_seed_cases(conn)
    n_cases = conn.execute("SELECT COUNT(*) FROM retained_cases").fetchone()[0]
    n_films = conn.execute("SELECT COUNT(*) FROM films").fetchone()[0]
    conn.close()
    print(f"\n=== Case base siap: {n_films} films, {n_cases} seed cases ===")
