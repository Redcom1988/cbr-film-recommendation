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
from typing import Optional

import faiss
import numpy as np

# ─── Paths default ───────────────────────────────────────────────────────────
_DEFAULT = {
    "db":         "models/cases.db",
    "case_idx":   "models/case_index.faiss",
    "movie_idx":  "models/movie_index.faiss",
    "vectorizer": "models/tfidf_vectorizer.pkl",
    "movie_ids":  "models/movie_ids.pkl",
    "case_ids":   "models/case_ids.pkl",
    "cf_model":   "models/cf_model.pkl",
}


class CBRRecommender:
    # ── Hyperparameters ───────────────────────────────────────────────────────
    MIN_CASEBASE_SIZE  = 10      # minimum kasus agar CBR aktif
    MIN_SIMILARITY     = 0.30    # threshold minimum case_similarity
    TOP_K_CASES        = 5       # jumlah kasus yang diambil tiap Retrieve
    BRUTE_FORCE_LIMIT  = 1000    # gunakan brute-force jika kasus < nilai ini
    TOP_K_DEFAULT      = 5       # default top-k rekomendasi ke user

    # Bobot case_similarity (query, movie, genre)
    W_FULL    = (0.50, 0.30, 0.20)  # query + film ref + genre
    W_NO_MOV  = (0.70, 0.00, 0.30)  # query + genre (tanpa film ref)
    W_QONLY   = (1.00, 0.00, 0.00)  # query saja

    # Bobot final_score (case_score, cf_score)
    W_REUSE   = (0.70, 0.30)        # user dikenali
    W_COLD    = (1.00, 0.00)        # cold start

    # ── Init ─────────────────────────────────────────────────────────────────
    def __init__(self, **paths):
        p = {**_DEFAULT, **paths}

        self.db_path = p["db"]

        with open(p["vectorizer"], 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(p["movie_ids"], 'rb') as f:
            self.movie_ids: list[int] = pickle.load(f)
        with open(p["case_ids"], 'rb') as f:
            self.case_ids: list[int] = pickle.load(f)
        with open(p["cf_model"], 'rb') as f:
            self.cf_model = pickle.load(f)

        self.movie_index: faiss.Index = faiss.read_index(p["movie_idx"])
        self.case_index:  faiss.Index = faiss.read_index(p["case_idx"])

        # Lookup cepat: movieId → indeks di movie_index
        self._movie_id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}

        print(f"[CBR] Loaded: {len(self.movie_ids)} film | {len(self.case_ids)} kasus")

    # ── Vectorize ─────────────────────────────────────────────────────────────
    def _vec(self, text: str) -> np.ndarray:
        """TF-IDF → L2-norm (1×D float32)."""
        v = self.vectorizer.transform([text]).toarray().astype('float32')
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    def _case_embedding_text(self, query: str, genres: str, title: str) -> str:
        """Teks representasi kasus: query + (genres*3) + title."""
        g = str(genres or '').strip()
        t = str(title  or '').strip()
        q = str(query  or '').strip()
        return q + ' ' + (g + ' ') * 3 + t

    # ── DB helpers ────────────────────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _get_film(self, movie_id: int) -> Optional[sqlite3.Row]:
        conn = self._conn()
        row  = conn.execute("SELECT * FROM films WHERE movieId=?", (movie_id,)).fetchone()
        conn.close()
        return row

    def find_movie_by_title(self, title_query: str) -> Optional[sqlite3.Row]:
        """Cari film berdasarkan judul (LIKE partial match)."""
        conn  = self._conn()
        exact = conn.execute(
            "SELECT * FROM films WHERE LOWER(title)=?", (title_query.lower(),)
        ).fetchone()
        if exact:
            conn.close()
            return exact
        like = conn.execute(
            "SELECT * FROM films WHERE LOWER(title) LIKE ? LIMIT 1",
            (f'%{title_query.lower()}%',)
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


    # ── RETRIEVE ──────────────────────────────────────────────────────────────
    def _retrieve(
        self,
        query_text:      str,
        ref_movie_id:    Optional[int],
        ref_genres:      str,
        ref_title:       str,
    ) -> tuple[list[dict], bool]:
        """
        Cari Top-K kasus paling mirip. 
        Return: (list_of_case_dicts, used_fallback)
        Setiap dict berisi: case_id, similarity, accepted_ids, rejected_ids, query_text
        """
        n_cases = len(self.case_ids)

        # ── Fallback ke CBF jika case base terlalu kecil ──
        if n_cases < self.MIN_CASEBASE_SIZE:
            return [], True

        # ── Buat query embedding ──────────────────────────
        emb_text = self._case_embedding_text(query_text, ref_genres, ref_title)
        q_vec    = self._vec(emb_text)

        # ── Cari di case_index ────────────────────────────
        k      = min(self.TOP_K_CASES, n_cases)
        sims, indices = self.case_index.search(q_vec, k)
        sims    = sims[0]
        indices = indices[0]

        # ── Ambil detail setiap kasus ────────────────────
        valid_case_ids = [self.case_ids[i] for i in indices if i >= 0]
        if not valid_case_ids:
            return [], True

        conn = self._conn()
        placeholders = ','.join('?' * len(valid_case_ids))
        rows = conn.execute(f"""
            SELECT rc.case_id, rc.query_text, rc.reference_movie,
                   rc.accepted_ids, rc.rejected_ids,
                   f.genres, f.title
            FROM   retained_cases rc
            LEFT JOIN films f ON f.movieId = rc.reference_movie
            WHERE  rc.case_id IN ({placeholders})
        """, valid_case_ids).fetchall()
        conn.close()

        row_map = {r['case_id']: r for r in rows}

        # ── Hitung case_similarity komponen per komponen ──
        # Query vector (hanya teks query, bukan full embedding)
        qv = self._vec(str(query_text))

        results = []
        for i, cid in enumerate(valid_case_ids):
            if cid not in row_map:
                continue
            row = row_map[cid]

            # Komponen 1: query_similarity
            case_qv       = self._vec(str(row['query_text'] or ''))
            query_sim     = float(np.dot(qv, case_qv.T))

            # Komponen 2 & 3: movie_similarity + genre_overlap
            c_movie_id    = row['reference_movie']
            c_genres      = str(row['genres'] or '')

            if ref_movie_id and c_movie_id:
                wq, wm, wg = self.W_FULL
                movie_sim  = self._movie_similarity(ref_movie_id, int(c_movie_id))
                genre_sim  = self._jaccard(ref_genres, c_genres)
            elif ref_movie_id is None and ref_genres:
                wq, wm, wg = self.W_NO_MOV
                movie_sim  = 0.0
                genre_sim  = self._jaccard(ref_genres, c_genres)
            else:
                wq, wm, wg = self.W_QONLY
                movie_sim  = 0.0
                genre_sim  = 0.0

            # Normalisasi bobot
            total = wq + wm + wg
            if total == 0:
                total = 1.0

            case_sim = (wq * query_sim + wm * movie_sim + wg * genre_sim) / total

            if case_sim >= self.MIN_SIMILARITY:
                results.append({
                    "case_id":      cid,
                    "similarity":   round(case_sim, 4),
                    "query_sim":    round(query_sim, 4),
                    "movie_sim":    round(movie_sim, 4),
                    "genre_sim":    round(genre_sim, 4),
                    "query_text":   row['query_text'],
                    "ref_movie":    c_movie_id,
                    "accepted_ids": json.loads(row['accepted_ids'] or '[]'),
                    "rejected_ids": json.loads(row['rejected_ids'] or '[]'),
                })

        if not results:
            return [], True  # semua di bawah threshold → fallback

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:self.TOP_K_CASES], False

    # ── REUSE ─────────────────────────────────────────────────────────────────
    def _reuse(
        self,
        similar_cases:  list[dict],
        user_id:        Optional[int],
        top_k:          int,
        fallback_query: str,
        fallback_genres: str,
    ) -> list[dict]:
        """
        Agregasi skor kandidat dari kasus mirip, re-ranking dengan CF.
        Return: list of dicts sorted by final_score.
        """
        agg_accepted: dict[int, float] = {}
        agg_rejected: dict[int, float] = {}

        for case in similar_cases:
            sim = case['similarity']
            for mid in case['accepted_ids']:
                agg_accepted[int(mid)] = agg_accepted.get(int(mid), 0.0) + sim
            for mid in case['rejected_ids']:
                agg_rejected[int(mid)] = agg_rejected.get(int(mid), 0.0) + sim

        # Net score + buang yang net ≤ 0
        candidates: dict[int, float] = {}
        for mid, acc_score in agg_accepted.items():
            net = acc_score - agg_rejected.get(mid, 0.0)
            if net > 0:
                candidates[mid] = net

        if not candidates:
            return []

        # Normalisasi ke [0, 1]
        max_score = max(candidates.values())
        if max_score > 0:
            candidates = {k: v / max_score for k, v in candidates.items()}

        # CF re-ranking
        is_cold = (user_id is None)
        wc, wf  = self.W_COLD if is_cold else self.W_REUSE

        conn    = self._conn()
        results = []
        for mid, case_score in candidates.items():
            cf  = 0.0 if is_cold else self._get_cf_score(user_id, mid)
            fs  = wc * case_score + wf * cf

            film_row = conn.execute(
                "SELECT movieId, title, genres, overview FROM films WHERE movieId=?", (mid,)
            ).fetchone()

            if film_row:
                results.append({
                    "movieId":      mid,
                    "title":        film_row['title'],
                    "genres":       film_row['genres'] or '',
                    "overview":     (film_row['overview'] or '')[:200],
                    "case_score":   round(case_score, 4),
                    "cf_score":     round(cf, 4),
                    "final_score":  round(fs, 4),
                    "from_cases":   [c['case_id'] for c in similar_cases
                                     if mid in c['accepted_ids']],
                })
        conn.close()

        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:top_k]

    # ── CBF Fallback ──────────────────────────────────────────────────────────
    def _fallback_cbf(
        self,
        query_text: str,
        ref_genres: str,
        user_id:    Optional[int],
        top_k:      int,
    ) -> list[dict]:
        """Fallback ke movie_index.faiss langsung jika case base belum cukup."""
        text  = (ref_genres + ' ') * 3 + query_text
        q_vec = self._vec(text)
        k     = min(top_k + 5, len(self.movie_ids))
        sims, indices = self.movie_index.search(q_vec, k)

        is_cold = (user_id is None)
        wc, wf  = self.W_COLD if is_cold else self.W_REUSE

        conn    = self._conn()
        results = []
        for sim, idx in zip(sims[0], indices[0]):
            if idx < 0:
                continue
            mid      = self.movie_ids[idx]
            cf       = 0.0 if is_cold else self._get_cf_score(user_id, mid)
            fs       = wc * float(sim) + wf * cf

            film_row = conn.execute(
                "SELECT movieId, title, genres, overview FROM films WHERE movieId=?", (mid,)
            ).fetchone()
            if film_row:
                results.append({
                    "movieId":     mid,
                    "title":       film_row['title'],
                    "genres":      film_row['genres'] or '',
                    "overview":    (film_row['overview'] or '')[:200],
                    "case_score":  round(float(sim), 4),
                    "cf_score":    round(cf, 4),
                    "final_score": round(fs, 4),
                    "from_cases":  [],
                })
        conn.close()

        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:top_k]

    # ── RECOMMEND (Retrieve + Reuse) ──────────────────────────────────────────
    def recommend(
        self,
        query_text:      str,
        reference_movie: Optional[str | int] = None,
        user_id:         Optional[int]        = None,
        top_k:           int                  = TOP_K_DEFAULT,
    ) -> dict:
        """
        Jalankan Retrieve + Reuse.
        Return dict berisi CBR flow info + rekomendasi.
        """
        # ── Resolve reference_movie ──────────────────────
        ref_movie_id = None
        ref_genres   = ''
        ref_title    = ''

        if reference_movie is not None:
            if isinstance(reference_movie, int):
                row = self._get_film(reference_movie)
            else:
                row = self.find_movie_by_title(str(reference_movie))

            if row:
                ref_movie_id = int(row['movieId'])
                ref_genres   = str(row['genres'] or '')
                ref_title    = str(row['title'] or '')

        # ── RETRIEVE ─────────────────────────────────────
        similar_cases, used_fallback = self._retrieve(
            query_text, ref_movie_id, ref_genres, ref_title
        )

        # ── REUSE / Fallback ──────────────────────────────
        if used_fallback:
            recommendations = self._fallback_cbf(query_text, ref_genres, user_id, top_k)
            retrieve_info   = {
                "cases_found": 0,
                "cases_passed_threshold": 0,
                "used_fallback": True,
                "top_case": None,
            }
        else:
            recommendations = self._reuse(similar_cases, user_id, top_k, query_text, ref_genres)
            retrieve_info   = {
                "cases_found":            len(similar_cases),
                "cases_passed_threshold": len(similar_cases),
                "used_fallback":          False,
                "top_case": {
                    "case_id":    similar_cases[0]['case_id'],
                    "query":      similar_cases[0]['query_text'],
                    "similarity": similar_cases[0]['similarity'],
                } if similar_cases else None,
            }

        # Sumber case_id utama (untuk RETAIN nanti)
        from_case_id = similar_cases[0]['case_id'] if (similar_cases and not used_fallback) else None

        return {
            "session_id":      str(uuid.uuid4()),
            "query_text":      query_text,
            "reference_movie": ref_movie_id,
            "ref_title":       ref_title,
            "user_id":         user_id,
            "retrieve":        retrieve_info,
            "reuse": {
                "total_candidates":   len(recommendations),
                "used_cf":            (user_id is not None),
            },
            "recommendations": recommendations,
            "from_case_id":    from_case_id,
        }

    def retain(
        self,
        user_id:         Optional[int],
        query_text:      str,
        reference_movie: Optional[int],
        recommended_ids: list[int],
        accepted_ids:    list[int],
        rejected_ids:    list[int],
        added_ids:       list[int] = [],
        from_case_id:    Optional[int] = None,
    ) -> int:
        """
        Simpan kasus baru ke retained_cases dan perbarui case_index.faiss.
        Jika pengguna menambahkan film baru (added_ids), gabungkan ke accepted_ids.
        Return: case_id baru.
        """
        
        final_accepted = list(set(accepted_ids + added_ids))
        
        conn = self._conn()
        cur  = conn.cursor()

        cur.execute("""
            INSERT INTO retained_cases
                (user_id, query_text, reference_movie,
                 recommended_ids, accepted_ids, rejected_ids, from_case_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            query_text,
            reference_movie,
            json.dumps(recommended_ids),
            json.dumps(final_accepted),
            json.dumps(rejected_ids),
            from_case_id,
        ))
        new_case_id = cur.lastrowid
        conn.commit()

        # Ambil detail film referensi untuk case embedding
        genres = ''
        title  = ''
        if reference_movie:
            row = conn.execute(
                "SELECT genres, title FROM films WHERE movieId=?", (reference_movie,)
            ).fetchone()
            if row:
                genres = str(row['genres'] or '')
                title  = str(row['title']  or '')
        conn.close()

        # Tambahkan vektor kasus baru ke case_index live
        text  = self._case_embedding_text(query_text, genres, title)
        v     = self._vec(text)
        self.case_index.add(v)
        self.case_ids.append(new_case_id)

        # Simpan case_ids yang diperbarui
        with open("models/case_ids.pkl", 'wb') as f:
            pickle.dump(self.case_ids, f)

        # Simpan case_index yang diperbarui
        faiss.write_index(self.case_index, "models/case_index.faiss")

        return new_case_id

    # ── EXPLAIN ───────────────────────────────────────────────────────────────
    def explain(
        self,
        query_text:      str,
        reference_movie: Optional[str | int] = None,
        user_id:         Optional[int]        = None,
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

        if result['retrieve']['top_case']:
            tc = result['retrieve']['top_case']
            lines += [
                f"Top case   : case_id={tc['case_id']} | sim={tc['similarity']}",
                f"           → \"{tc['query']}\"",
            ]

        lines += ["", f"── REUSE ─────────────────────────────────────────"]
        for r in result['recommendations']:
            lines.append(
                f"  {r['title']:<35} "
                f"case={r['case_score']:.3f} cf={r['cf_score']:.3f} "
                f"final={r['final_score']:.3f}"
            )
        lines.append("=" * 60)
        return '\n'.join(lines)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rec = CBRRecommender()
    print(rec.explain("space exploration movie", reference_movie="Interstellar"))
