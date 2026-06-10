import faiss
import pickle
import sqlite3
import json
import math

class CBRRecommender:
    def __init__(self, db_path="models/cases.db", faiss_path="models/cbf_index.faiss", vectorizer_path="models/tfidf_vectorizer.pkl", cf_model_path="models/cf_model.pkl"):
        self.db_path = db_path
        self.cbf_index = faiss.read_index(faiss_path)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(cf_model_path, 'rb') as f:
            self.cf_model = pickle.load(f)
            
    def _get_historical_frequencies(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT accepted_ids FROM retained_cases')
        rows = cursor.fetchall()
        conn.close()
        
        freq = {}
        for row in rows:
            try:
                acc_ids = json.loads(row[0])
                for aid in acc_ids:
                    freq[aid] = freq.get(aid, 0) + 1
            except:
                pass
        return freq

    def recommend(self, query_text, user_id=None, top_k=6):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 1. Check if query is a movie title
        cursor.execute("SELECT overview, genres FROM films WHERE title COLLATE NOCASE = ?", (query_text.strip(),))
        exact_movie = cursor.fetchone()
        
        if exact_movie:
            # It's a movie! Use its description and genres as the search query
            query_content = str(exact_movie['overview']) + " " + str(exact_movie['genres'])
        else:
            query_content = query_text
            
        query_vec = self.vectorizer.transform([query_content]).toarray().astype('float32')
        faiss.normalize_L2(query_vec)
        
        # 2. Retrieve valid cases from index
        cursor.execute('SELECT movieId, tfidf_norm, language, vote_avg, is_validated FROM index_file')
        index_rows = cursor.fetchall()
        
        valid_movie_ids = set()
        for row in index_rows:
            # Filters I2, I4
            if row['language'] == 'en' and row['vote_avg'] >= 5.0 and row['is_validated'] and row['tfidf_norm'] >= 0.1:
                valid_movie_ids.add(row['movieId'])
                
        if not valid_movie_ids:
            return []
            
        # 3. Retrieve from FAISS
        k_search = min(200, self.cbf_index.ntotal)
        D, I = self.cbf_index.search(query_vec, k_search)
        idx_to_score = {int(I[0][i]): float(D[0][i]) for i in range(len(I[0])) if I[0][i] != -1}
        
        # 4. Score Fusion
        hist_freq = self._get_historical_frequencies()
        max_freq = max(hist_freq.values()) if hist_freq else 0
        
        placeholders = ','.join(['?'] * len(idx_to_score))
        cursor.execute(f'SELECT * FROM films WHERE tfidf_vector_idx IN ({placeholders})', list(idx_to_score.keys()))
        film_rows = cursor.fetchall()
        
        results = []
        for row in film_rows:
            m_id = row['movieId']
            if m_id not in valid_movie_ids:
                continue
                
            skor_cbf = float(idx_to_score[row['tfidf_vector_idx']])
            
            # Collaborative Filtering (Cold Start Aware)
            if user_id is not None:
                pred = self.cf_model.predict(user_id, m_id).est
                skor_cf = (pred - 0.5) / 4.5
                alpha_cf = 0.25
                w_cbf = 0.60
            else:
                skor_cf = 0.0
                alpha_cf = 0.0
                w_cbf = 0.85
                
            freq = hist_freq.get(m_id, 0)
            skor_hist = math.log1p(freq) / math.log1p(max_freq) if max_freq > 0 else 0.0
            
            skor_akhir = (w_cbf * skor_cbf) + (alpha_cf * skor_cf) + (0.15 * skor_hist)
            
            results.append({
                'movieId': m_id,
                'title': row['title'],
                'genres': row['genres'],
                'overview': row['overview'],
                'score': float(skor_akhir)
            })
            
        conn.close()
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def revise_and_retain(self, query_text, accepted_ids, rejected_ids):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO retained_cases (query_text, accepted_ids, rejected_ids)
            VALUES (?, ?, ?)
        ''', (query_text, json.dumps(accepted_ids), json.dumps(rejected_ids)))
        conn.commit()
        conn.close()
