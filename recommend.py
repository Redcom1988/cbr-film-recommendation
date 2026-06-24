"""
recommend.py
Phase 4: CBR Engine — Retrieve, Reuse, Revise, Retain.
Kelas utama: CBRRecommender
"""

import json
import os
import pickle
import sqlite3
import uuid
from collections import defaultdict
from typing import Optional

import faiss
import numpy as np
import pandas as pd

# ─── Paths default ───────────────────────────────────────────────────────────
_DEFAULT = {
    "db": "models/cases.db",
    "case_idx": "models/case_index.faiss",
    "movie_idx": "models/movie_index.faiss",
    "vectorizer": "models/tfidf_vectorizer.pkl",
    "movie_ids": "models/movie_ids.pkl",
    "case_ids": "models/case_ids.pkl",
    "cf_model": "models/cf_model.pkl",
}


class CBRRecommender:
    # ── Hyperparameters ───────────────────────────────────────────────────────
    MIN_CASEBASE_SIZE = 10  # minimum kasus agar CBR aktif
    MIN_SIMILARITY = 0.30  # threshold minimum case_similarity
    TOP_K_CASES = 5  # jumlah kasus yang diambil tiap Retrieve
    BRUTE_FORCE_LIMIT = 1000  # gunakan brute-force jika kasus < nilai ini
    TOP_K_DEFAULT = 5  # default top-k rekomendasi ke user

    # Bobot case_similarity (query, movie, genre)
    W_FULL = (0.50, 0.30, 0.20)  # query + film ref + genre
    W_NO_MOV = (0.70, 0.00, 0.30)  # query + genre (tanpa film ref)
    W_QONLY = (1.00, 0.00, 0.00)  # query saja

    # Bobot case_similarity dengan user preference (query, user, movie, genre)
    W_FULL_USER = (0.30, 0.30, 0.20, 0.20)  # query + user + film ref + genre
    W_NO_MOV_USER = (0.40, 0.35, 0.00, 0.25)  # query + user + genre
    W_QONLY_USER = (0.55, 0.45, 0.00, 0.00)  # query + user saja

    # Bobot final_score (case_score, cf_score, genre_adapt)
    W_REUSE = (0.65, 0.25, 0.10)  # case + cf + genre adaptation
    W_COLD = (1.00, 0.00, 0.00)  # cold start (no cf, no genre adapt)

    # User clustering
    USER_SIM_TOP_N = 30  # jumlah similar user untuk filtering kasus
    USER_SIM_MIN = 0.10  # minimum user similarity threshold

    # Novelty / feedback-loop prevention
    NOVELTY_PENALTY = (
        0.85  # kurangi 85% final score untuk film yang sudah pernah di-accept user
    )

    # ── Init ─────────────────────────────────────────────────────────────────
    def __init__(self, **paths):
        p = {**_DEFAULT, **paths}

        self.db_path = p["db"]

        with open(p["vectorizer"], "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(p["movie_ids"], "rb") as f:
            self.movie_ids: list[int] = pickle.load(f)
        with open(p["case_ids"], "rb") as f:
            self.case_ids: list[int] = pickle.load(f)
        with open(p["cf_model"], "rb") as f:
            self.cf_model = pickle.load(f)

        self.movie_index: faiss.Index = faiss.read_index(p["movie_idx"])
        self.case_index: faiss.Index = faiss.read_index(p["case_idx"])

        # Lookup cepat: movieId → indeks di movie_index
        self._movie_id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}

        # Load user genre preference profiles
        self.user_genre_vecs, self.all_genres, self.movie_genre_map = (
            self._load_user_genre_profiles()
        )
        print(
            f"[CBR] User profiles: {len(self.user_genre_vecs)} users | {len(self.all_genres)} genres"
        )

        print(f"[CBR] Loaded: {len(self.movie_ids)} film | {len(self.case_ids)} kasus")

    # ── Vectorize ─────────────────────────────────────────────────────────────
    def _vec(self, text: str) -> np.ndarray:
        """TF-IDF → L2-norm (1D array of shape (D,))."""
        v = self.vectorizer.transform([text]).toarray().astype("float32").ravel()
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    @staticmethod
    def _clean_overview(ov: str | None) -> str:
        o = (ov or "").strip()[:200]
        ol = o.lower()
        if not ol or ol in (
            "overview",
            "overview found",
            "overview yet",
            "movie overview available",
            "movie overview available please add one themoviedborg",
            "no overview found.",
            "no overview yet.",
        ):
            return ""
        return o

    def _case_embedding_text(
        self, query: str, genres: str, title: str, overview: str = ""
    ) -> str:
        """Teks representasi kasus: query + (genres*3) + title + overview."""
        g = str(genres or "").strip()
        t = str(title or "").strip()
        q = str(query or "").strip()
        o = str(overview or "").strip()
        return q + " " + (g + " ") * 3 + t + " " + o

    # ── DB helpers ────────────────────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _get_film(self, movie_id: int) -> Optional[sqlite3.Row]:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM films WHERE movieId=?", (movie_id,)
        ).fetchone()
        conn.close()
        return row

    def find_movie_by_title(self, title_query: str) -> Optional[sqlite3.Row]:
        """Cari film berdasarkan judul (LIKE partial match)."""
        conn = self._conn()
        exact = conn.execute(
            "SELECT * FROM films WHERE LOWER(title)=?", (title_query.lower(),)
        ).fetchone()
        if exact:
            conn.close()
            return exact
        like = conn.execute(
            "SELECT * FROM films WHERE LOWER(title) LIKE ? LIMIT 1",
            (f"%{title_query.lower()}%",),
        ).fetchone()
        conn.close()
        return like

    def _get_cf_score(self, user_id: Optional[int], movie_id: int) -> float:
        """Prediksi rating CF, normalisasi ke [0,1]. 0 jika cold start."""
        if user_id is None:
            return 0.0
        try:
            pred = self.cf_model.predict(user_id, movie_id).est
            return float(np.clip((pred - 0.5) / 4.5, 0.0, 1.0))  # scale 0.5-5 → 0-1
        except Exception:
            return 0.0

    # ── User Genre Profiles ───────────────────────────────────
    def _load_user_genre_profiles(self) -> tuple[dict, list, dict]:
        """Build user genre preference vectors from ratings history.

        Returns:
            user_vecs:  dict[int, np.ndarray] — user_id → genre preference vector [0,1]
            all_genres: list[str] — sorted list of all genre names
            movie_genres: dict[int, list[str]] — movieId → list of genres
        """
        genres_csv = "output/genres.csv"
        cf_csv = "output/collaborative_training.csv"

        gdf = pd.read_csv(genres_csv)
        movie_genres: dict[int, list[str]] = defaultdict(list)
        for _, row in gdf.iterrows():
            movie_genres[int(row["movieId"])].append(str(row["genre"]).strip())

        all_genres = sorted(gdf["genre"].unique())
        genre_idx = {g: i for i, g in enumerate(all_genres)}

        rdf = pd.read_csv(cf_csv)
        n_gen = len(all_genres)
        user_vecs: dict[int, np.ndarray] = {}

        for uid, group in rdf.groupby("userId"):
            vec = np.zeros(n_gen, dtype=np.float32)
            count = np.zeros(n_gen, dtype=np.float32)
            for _, r in group.iterrows():
                mid = int(r["movieId"])
                rating = float(r["rating"])
                for g in movie_genres.get(mid, []):
                    idx = genre_idx[g]
                    vec[idx] += rating
                    count[idx] += 1.0

            mask = count > 0
            if mask.any():
                vec[mask] /= count[mask]  # avg rating per genre
                vec[mask] /= 5.0  # normalize ke [0, 1]
            user_vecs[int(uid)] = vec

        return user_vecs, all_genres, dict(movie_genres)

    def _user_genre_vec(self, user_id: Optional[int]) -> Optional[np.ndarray]:
        if user_id is None:
            return None
        return self.user_genre_vecs.get(int(user_id))

    def _user_cosine_sim(self, uid_a: Optional[int], uid_b: Optional[int]) -> float:
        """Cosine similarity between two users' genre profiles."""
        if uid_a is None or uid_b is None:
            return 0.5  # neutral — can't compare
        va = self._user_genre_vec(uid_a)
        vb = self._user_genre_vec(uid_b)
        if va is None or vb is None:
            return 0.3  # slightly below neutral (unknown user)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom < 1e-10:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def _find_similar_users(self, user_id: int, top_n: int = 30) -> list[int]:
        """Find top-N users with most similar genre preferences."""
        target = self._user_genre_vec(user_id)
        if target is None:
            return []
        scores: list[tuple[int, float]] = []
        for uid, vec in self.user_genre_vecs.items():
            if uid == user_id:
                continue
            denom = np.linalg.norm(vec) * np.linalg.norm(target)
            sim = float(np.dot(vec, target) / denom) if denom > 1e-10 else 0.0
            if sim >= self.USER_SIM_MIN:
                scores.append((uid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [uid for uid, _ in scores[:top_n]]

    # ── Seen films (feedback-loop prevention) ────────────────────────────────
    def _get_seen_films(self, user_id: Optional[int]) -> set[int]:
        """Get all movie IDs the user has ever accepted across all retained cases."""
        if user_id is None:
            return set()
        conn = self._conn()
        rows = conn.execute(
            "SELECT accepted_ids FROM retained_cases WHERE user_id = ?",
            (int(user_id),),
        ).fetchall()
        conn.close()
        seen: set[int] = set()
        for row in rows:
            seen.update(int(x) for x in json.loads(row["accepted_ids"] or "[]"))
        return seen

    # ── Genre helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        sa = set(a.lower().split()) if a else set()
        sb = set(b.lower().split()) if b else set()
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _movie_similarity(self, movie_id_a: int, movie_id_b: int) -> float:
        """Cosine similarity antara dua vektor film di movie_index."""
        if movie_id_a == movie_id_b:
            return 1.0
        ia = self._movie_id_to_idx.get(movie_id_a)
        ib = self._movie_id_to_idx.get(movie_id_b)
        if ia is None or ib is None:
            return 0.0
        try:
            va = self.movie_index.reconstruct(ia)
            vb = self.movie_index.reconstruct(ib)
            return float(np.dot(va, vb))
        except Exception:
            return 0.0

    # ── Genre pattern extraction dari retrieved cases ─────────────────────────
    def _extract_genre_patterns(self, similar_cases: list[dict]) -> dict[str, float]:
        """Extract net genre preference from accepted/rejected films in cases.

        Returns dict mapping genre -> net preference in [-1, 1].
        Positive = users similar to this query tend to accept this genre.
        """
        genre_accept: dict[str, float] = {}
        genre_reject: dict[str, float] = {}

        all_mids: set[int] = set()
        for case in similar_cases:
            all_mids.update(case["accepted_ids"])
            all_mids.update(case["rejected_ids"])

        if not all_mids:
            return {}

        conn = self._conn()
        placeholders = ",".join("?" * len(all_mids))
        rows = conn.execute(
            f"SELECT movieId, genres FROM films WHERE movieId IN ({placeholders})",
            list(all_mids),
        ).fetchall()
        conn.close()

        film_genre_map: dict[int, list[str]] = {
            r["movieId"]: str(r["genres"] or "").split() for r in rows
        }

        for case in similar_cases:
            sim = case["similarity"]
            for mid in case["accepted_ids"]:
                for g in film_genre_map.get(int(mid), []):
                    genre_accept[g] = genre_accept.get(g, 0.0) + sim
            for mid in case["rejected_ids"]:
                for g in film_genre_map.get(int(mid), []):
                    genre_reject[g] = genre_reject.get(g, 0.0) + sim

        net: dict[str, float] = {}
        all_gs = set(genre_accept.keys()) | set(genre_reject.keys())
        for g in all_gs:
            net[g] = genre_accept.get(g, 0.0) - genre_reject.get(g, 0.0)

        max_abs = max((abs(v) for v in net.values()), default=0.0)
        if max_abs > 0:
            net = {g: v / max_abs for g, v in net.items()}
        return net

    # ── RETRIEVE ──────────────────────────────────────────────────────────────
    def _retrieve(
        self,
        query_text: str,
        ref_movie_id: Optional[int],
        ref_genres: str,
        ref_title: str,
        user_id: Optional[int] = None,
        ref_overview: str = "",
    ) -> tuple[list[dict], bool]:
        """
        Cari Top-K kasus paling mirip.
        Jika user_id diberikan, kasus dari user berbeda di-filter
        berdasarkan kemiripan profil genre.
        """
        n_cases = len(self.case_ids)

        if n_cases < self.MIN_CASEBASE_SIZE:
            return [], True

        # Cari similar user terlebih dahulu
        similar_user_ids: set[int] = set()
        has_user_profile = user_id is not None and user_id in self.user_genre_vecs
        if has_user_profile:
            similar_user_ids = set(
                self._find_similar_users(user_id, self.USER_SIM_TOP_N)
            )

        # Buat query embedding
        emb_text = self._case_embedding_text(
            query_text, ref_genres, ref_title, ref_overview
        )
        q_vec = self._vec(emb_text).reshape(1, -1)

        # Cari di case_index (ambil lebih banyak untuk filtering user)
        k = min(self.TOP_K_CASES * 3, n_cases)
        sims, indices = self.case_index.search(q_vec, k)
        sims = sims[0]
        indices = indices[0]

        valid_case_ids = [self.case_ids[i] for i in indices if i >= 0]
        if not valid_case_ids:
            return [], True

        conn = self._conn()
        placeholders = ",".join("?" * len(valid_case_ids))
        rows = conn.execute(
            f"""
            SELECT rc.case_id, rc.query_text, rc.reference_movie,
                   rc.accepted_ids, rc.rejected_ids,
                   rc.user_id,
                   f.genres, f.title
            FROM   retained_cases rc
            LEFT JOIN films f ON f.movieId = rc.reference_movie
            WHERE  rc.case_id IN ({placeholders})
        """,
            valid_case_ids,
        ).fetchall()
        conn.close()

        row_map = {r["case_id"]: r for r in rows}

        qv = self._vec(str(query_text))

        results = []
        for i, cid in enumerate(valid_case_ids):
            if cid not in row_map:
                continue
            row = row_map[cid]
            case_uid = row["user_id"]

            # Skip cases from dissimilar users (keep seed cases with NULL user)
            if has_user_profile and similar_user_ids and case_uid is not None:
                if int(case_uid) not in similar_user_ids:
                    continue

            # Query similarity
            case_qv = self._vec(str(row["query_text"] or ""))
            query_sim = float(np.dot(qv, case_qv))

            # User similarity (0.5 neutral for seed cases)
            user_sim = self._user_cosine_sim(user_id, case_uid)

            # Movie + genre similarity
            c_movie_id = row["reference_movie"]
            c_genres = str(row["genres"] or "")

            if has_user_profile and ref_movie_id and c_movie_id:
                wq, wu, wm, wg = self.W_FULL_USER
                movie_sim = self._movie_similarity(ref_movie_id, int(c_movie_id))
                genre_sim = self._jaccard(ref_genres, c_genres)
            elif has_user_profile and ref_movie_id is None and ref_genres:
                wq, wu, wm, wg = self.W_NO_MOV_USER
                movie_sim = 0.0
                genre_sim = self._jaccard(ref_genres, c_genres)
            elif has_user_profile:
                wq, wu, wm, wg = self.W_QONLY_USER
                movie_sim = 0.0
                genre_sim = 0.0
            elif ref_movie_id and c_movie_id:
                wq, wu, wm, wg = (self.W_FULL[0], 0.0, self.W_FULL[1], self.W_FULL[2])
                movie_sim = self._movie_similarity(ref_movie_id, int(c_movie_id))
                genre_sim = self._jaccard(ref_genres, c_genres)
            elif ref_movie_id is None and ref_genres:
                wq, wu, wm, wg = (
                    self.W_NO_MOV[0],
                    0.0,
                    self.W_NO_MOV[1],
                    self.W_NO_MOV[2],
                )
                movie_sim = 0.0
                genre_sim = self._jaccard(ref_genres, c_genres)
            else:
                wq, wu, wm, wg = (
                    self.W_QONLY[0],
                    0.0,
                    self.W_QONLY[1],
                    self.W_QONLY[2],
                )
                movie_sim = 0.0
                genre_sim = 0.0

            total_w = wq + wu + wm + wg
            if total_w == 0:
                total_w = 1.0
            case_sim = (
                wq * query_sim + wu * user_sim + wm * movie_sim + wg * genre_sim
            ) / total_w

            if case_sim >= self.MIN_SIMILARITY:
                results.append(
                    {
                        "case_id": cid,
                        "similarity": round(case_sim, 4),
                        "query_sim": round(query_sim, 4),
                        "user_sim": round(user_sim, 4),
                        "movie_sim": round(movie_sim, 4),
                        "genre_sim": round(genre_sim, 4),
                        "query_text": row["query_text"],
                        "ref_movie": c_movie_id,
                        "accepted_ids": json.loads(row["accepted_ids"] or "[]"),
                        "rejected_ids": json.loads(row["rejected_ids"] or "[]"),
                    }
                )

        # Fallback: if user filter killed all results, retry without it
        if not results and has_user_profile and similar_user_ids:
            for i, cid in enumerate(valid_case_ids):
                if cid not in row_map:
                    continue
                row = row_map[cid]
                case_qv = self._vec(str(row["query_text"] or ""))
                qs = float(np.dot(qv, case_qv))
                ms = (
                    self._movie_similarity(ref_movie_id, int(row["reference_movie"]))
                    if ref_movie_id and row["reference_movie"]
                    else 0.0
                )
                gs = (
                    self._jaccard(ref_genres, str(row["genres"] or ""))
                    if ref_genres
                    else 0.0
                )
                cs = 0.50 * qs + 0.30 * ms + 0.20 * gs
                if cs >= self.MIN_SIMILARITY:
                    results.append(
                        {
                            "case_id": cid,
                            "similarity": round(cs, 4),
                            "query_sim": round(qs, 4),
                            "user_sim": round(
                                self._user_cosine_sim(user_id, row["user_id"]), 4
                            ),
                            "movie_sim": round(ms, 4),
                            "genre_sim": round(gs, 4),
                            "query_text": row["query_text"],
                            "ref_movie": row["reference_movie"],
                            "accepted_ids": json.loads(row["accepted_ids"] or "[]"),
                            "rejected_ids": json.loads(row["rejected_ids"] or "[]"),
                        }
                    )

        if not results:
            return [], True

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[: self.TOP_K_CASES], False

    # ── REUSE ─────────────────────────────────────────────────────────────────
    def _reuse(
        self,
        similar_cases: list[dict],
        user_id: Optional[int],
        top_k: int,
        fallback_query: str,
        fallback_genres: str,
    ) -> list[dict]:
        """
        Agregasi skor kandidat dari kasus mirip, re-ranking dengan CF
        dan genre-level adaptation.
        """
        agg_accepted: dict[int, float] = {}
        agg_rejected: dict[int, float] = {}

        for case in similar_cases:
            sim = case["similarity"]
            for mid in case["accepted_ids"]:
                agg_accepted[int(mid)] = agg_accepted.get(int(mid), 0.0) + sim
            for mid in case["rejected_ids"]:
                agg_rejected[int(mid)] = agg_rejected.get(int(mid), 0.0) + sim

        # Hard exclusion: buang film yang ada di rejected_ids kasus manapun
        rejected_set: set[int] = set()
        for case in similar_cases:
            rejected_set.update(int(m) for m in case.get("rejected_ids", []))

        # Net score: buang yang net <= 0 + hard exclusion
        candidates: dict[int, float] = {}
        for mid, acc_score in agg_accepted.items():
            mid_int = int(mid)
            if mid_int in rejected_set:
                continue
            net = acc_score - agg_rejected.get(mid_int, 0.0)
            if net > 0:
                candidates[mid_int] = net

        if not candidates:
            return []

        max_score = max(candidates.values())
        if max_score > 0:
            candidates = {k: v / max_score for k, v in candidates.items()}

        # Genre-level adaptation dari retrieved cases
        genre_net = self._extract_genre_patterns(similar_cases)

        # Cari film yang sudah pernah di-accept user (novelty penalty)
        seen_films = self._get_seen_films(user_id)

        # Re-ranking: case + cf + genre adaptation + novelty
        is_cold = user_id is None
        wc, wf, wg = self.W_COLD if is_cold else self.W_REUSE

        conn = self._conn()
        results = []
        for mid, case_score in candidates.items():
            cf = 0.0 if is_cold else self._get_cf_score(user_id, mid)

            # Genre boost: match candidate film genres vs learned patterns
            fg_row = conn.execute(
                "SELECT genres FROM films WHERE movieId=?", (mid,)
            ).fetchone()
            film_genres = str(fg_row["genres"] or "").split() if fg_row else []
            genre_boost = 0.0
            if film_genres and genre_net:
                gs = [genre_net.get(g, 0.0) for g in film_genres]
                genre_boost = sum(gs) / len(gs)

            novelty = 1.0
            if mid in seen_films:
                novelty -= self.NOVELTY_PENALTY

            fs = (wc * case_score + wf * cf + wg * genre_boost) * novelty

            film_row = conn.execute(
                "SELECT movieId, title, genres, overview FROM films WHERE movieId=?",
                (mid,),
            ).fetchone()

            if film_row:
                results.append(
                    {
                        "movieId": mid,
                        "title": film_row["title"],
                        "genres": film_row["genres"] or "",
                        "overview": self._clean_overview(film_row["overview"]),
                        "case_score": round(case_score, 4),
                        "cf_score": round(cf, 4),
                        "genre_adapt_score": round(genre_boost, 4),
                        "final_score": round(fs, 4),
                        "from_cases": [
                            c["case_id"]
                            for c in similar_cases
                            if mid in c["accepted_ids"]
                        ],
                    }
                )
        conn.close()

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    # ── CBF Fallback ──────────────────────────────────────────────────────────
    def _fallback_cbf(
        self,
        query_text: str,
        ref_genres: str,
        user_id: Optional[int],
        top_k: int,
    ) -> list[dict]:
        """Fallback ke movie_index.faiss langsung jika case base belum cukup."""
        text = (ref_genres + " ") * 3 + query_text
        q_vec = self._vec(text).reshape(1, -1)
        k = min(top_k + 5, len(self.movie_ids))
        sims, indices = self.movie_index.search(q_vec, k)

        wc, wf, wg = self.W_COLD

        seen_films = self._get_seen_films(user_id)

        conn = self._conn()
        results = []
        for sim, idx in zip(sims[0], indices[0]):
            if idx < 0:
                continue
            mid = self.movie_ids[idx]
            novelty = 1.0
            if mid in seen_films:
                novelty -= self.NOVELTY_PENALTY
            fs = wc * float(sim) * novelty

            film_row = conn.execute(
                "SELECT movieId, title, genres, overview FROM films WHERE movieId=?",
                (mid,),
            ).fetchone()
            if film_row:
                results.append(
                    {
                        "movieId": mid,
                        "title": film_row["title"],
                        "genres": film_row["genres"] or "",
                        "overview": self._clean_overview(film_row["overview"]),
                        "case_score": round(float(sim), 4),
                        "cf_score": 0.0,
                        "final_score": round(fs, 4),
                        "from_cases": [],
                    }
                )
        conn.close()

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    # ── RECOMMEND (Retrieve + Reuse) ──────────────────────────────────────────
    def recommend(
        self,
        query_text: str,
        reference_movie: Optional[str | int] = None,
        user_id: Optional[int] = None,
        top_k: int = TOP_K_DEFAULT,
    ) -> dict:
        """
        Jalankan Retrieve + Reuse.
        Return dict berisi CBR flow info + rekomendasi.
        """
        # ── Resolve reference_movie ──────────────────────
        ref_movie_id = None
        ref_genres = ""
        ref_title = ""
        ref_overview = ""

        if reference_movie is not None:
            if isinstance(reference_movie, int):
                row = self._get_film(reference_movie)
            else:
                row = self.find_movie_by_title(str(reference_movie))

            if row:
                ref_movie_id = int(row["movieId"])
                ref_genres = str(row["genres"] or "")
                ref_title = str(row["title"] or "")
                ref_overview = str(row["overview"] or "")

        # ── RETRIEVE ─────────────────────────────────────
        similar_cases, used_fallback = self._retrieve(
            query_text, ref_movie_id, ref_genres, ref_title, user_id, ref_overview
        )

        # ── REUSE / Fallback ──────────────────────────────
        if used_fallback:
            recommendations = self._fallback_cbf(query_text, ref_genres, user_id, top_k)
            retrieve_info = {
                "cases_found": 0,
                "cases_passed_threshold": 0,
                "used_fallback": True,
                "top_case": None,
            }
        else:
            recommendations = self._reuse(
                similar_cases, user_id, top_k, query_text, ref_genres
            )
            # Always mix in fallback candidates for diversity
            existing_ids = {r["movieId"] for r in recommendations}
            rejected_ids = set()
            for case in similar_cases:
                rejected_ids.update(int(m) for m in case.get("rejected_ids", []))
            padding = self._fallback_cbf(query_text, ref_genres, user_id, top_k + 10)
            for r in padding:
                if (
                    r["movieId"] not in existing_ids
                    and r["movieId"] not in rejected_ids
                ):
                    r["from_cases"] = []
                    recommendations.append(r)
                    existing_ids.add(r["movieId"])
            # Re-sort so fresh content can rank above penalized seen movies
            recommendations.sort(key=lambda x: x["final_score"], reverse=True)
            recommendations = recommendations[:top_k]
            retrieve_info = {
                "cases_found": len(similar_cases),
                "cases_passed_threshold": len(similar_cases),
                "used_fallback": False,
                "top_case": {
                    "case_id": similar_cases[0]["case_id"],
                    "query": similar_cases[0]["query_text"],
                    "similarity": similar_cases[0]["similarity"],
                }
                if similar_cases
                else None,
            }

        # Sumber case_id utama (untuk RETAIN nanti)
        from_case_id = (
            similar_cases[0]["case_id"]
            if (similar_cases and not used_fallback)
            else None
        )

        return {
            "session_id": str(uuid.uuid4()),
            "query_text": query_text,
            "reference_movie": ref_movie_id,
            "ref_title": ref_title,
            "user_id": user_id,
            "retrieve": retrieve_info,
            "reuse": {
                "total_candidates": len(recommendations),
                "used_cf": (user_id is not None),
            },
            "recommendations": recommendations,
            "from_case_id": from_case_id,
        }

    def retain(
        self,
        user_id: Optional[int],
        query_text: str,
        reference_movie: Optional[int],
        recommended_ids: list[int],
        accepted_ids: list[int],
        rejected_ids: list[int],
        from_case_id: Optional[int] = None,
    ) -> int:
        """
        Simpan kasus baru ke retained_cases dan perbarui case_index.faiss.
        Jika pengguna menambahkan film baru (added_ids), gabungkan ke accepted_ids.
        Return: case_id baru.
        """
        
        final_accepted = list(set(accepted_ids + added_ids))
        
        conn = self._conn()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO retained_cases
                (user_id, query_text, reference_movie,
                 recommended_ids, accepted_ids, rejected_ids, from_case_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                user_id,
                query_text,
                reference_movie,
                json.dumps(recommended_ids),
                json.dumps(accepted_ids),
                json.dumps(rejected_ids),
                from_case_id,
            ),
        )
        new_case_id = cur.lastrowid
        conn.commit()

        # Ambil detail film referensi untuk case embedding
        genres = ""
        title = ""
        overview = ""
        if reference_movie:
            row = conn.execute(
                "SELECT genres, title, overview FROM films WHERE movieId=?",
                (reference_movie,),
            ).fetchone()
            if row:
                genres = str(row["genres"] or "")
                title = str(row["title"] or "")
                overview = str(row["overview"] or "")
        conn.close()

        # Tambahkan vektor kasus baru ke case_index live
        text = self._case_embedding_text(query_text, genres, title, overview)
        v = self._vec(text).reshape(1, -1)
        self.case_index.add(v)
        self.case_ids.append(new_case_id)

        # Simpan case_ids yang diperbarui
        with open("models/case_ids.pkl", "wb") as f:
            pickle.dump(self.case_ids, f)

        # Simpan case_index yang diperbarui
        faiss.write_index(self.case_index, "models/case_index.faiss")

        return new_case_id

    # ── EXPLAIN ───────────────────────────────────────────────────────────────
    def explain(
        self,
        query_text: str,
        reference_movie: Optional[str | int] = None,
        user_id: Optional[int] = None,
    ) -> str:
        """Tampilkan detail debug CBR flow."""
        result = self.recommend(query_text, reference_movie, user_id, top_k=5)

        lines = [
            "=" * 60,
            f"QUERY     : {result['query_text']}",
            f"REF FILM  : {result['ref_title'] or '-'} (id={result['reference_movie']})",
            f"USER      : {result['user_id'] or 'cold-start'}",
            "",
            f"── RETRIEVE ──────────────────────────────────────",
            f"Fallback CBF : {result['retrieve']['used_fallback']}",
            f"Kasus ditemukan: {result['retrieve']['cases_found']}",
        ]

        if result["retrieve"]["top_case"]:
            tc = result["retrieve"]["top_case"]
            lines += [
                f"Top case   : case_id={tc['case_id']} | sim={tc['similarity']}",
                f'           → "{tc["query"]}"',
            ]

        lines += ["", f"── REUSE ─────────────────────────────────────────"]
        for r in result["recommendations"]:
            lines.append(
                f"  {r['title']:<35} "
                f"case={r['case_score']:.3f} cf={r['cf_score']:.3f} "
                f"final={r['final_score']:.3f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rec = CBRRecommender()
    print(rec.explain("space exploration movie", reference_movie="Interstellar"))
