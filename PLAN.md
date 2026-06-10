# PLAN.md — CBR Film Recommendation System

**Mata Kuliah:** Metode Penalaran  
**Topik:** Sistem Rekomendasi Film berbasis Case-Based Reasoning (CBR) + Hybrid Filtering  
**Dataset:** TMDB 5000 + MovieLens 25M  

---

## Daftar Isi

1. [Arsitektur](#1-arsitektur)
2. [File dalam Project](#2-file-dalam-project)
3. [Mengapa SQLite + FAISS, bukan Neo4j/NoSQL](#3-mengapa-sqlite--faiss-bukan-neo4jnosql)
4. [Preprocessing](#4-preprocessing)
5. [Training](#5-training)
6. [CBR Pipeline (4R Cycle)](#6-cbr-pipeline-4r-cycle)
7. [Cara Pakai](#7-cara-pakai)
8. [Query Rekomendasi](#8-query-rekomendasi)

---

## 1. Arsitektur

```
┌──────────────────────┐     ┌──────────────────────────┐
│  tmdb_5000_movies    │     │  MovieLens 25M           │
│  .csv                │     │  ratings.csv             │
│  (4.803 film)        │     │  (25jt rating, 162k user)│
└──────────┬───────────┘     └──────────────┬───────────┘
           │                                │
           ▼                                ▼
   ┌────────────────────────────────────────────┐
   │                PREPROCESSING               │
   │  A. Merge title matching (TMDB ↔ MovieLens)│
   │  B. Build text_combined (overview + genre  │
   │     + cast + keywords)                     │
   │  C. One-hot encode genre (19 kategori)     │
   │  D. Filter: bahasa Inggris, vote_count ≥ 20│
   │  E. Filter rating: user ≥ 5 ratings        │
   └──────┬──────────────────────────┬──────────┘
          │                          │
          ▼                          ▼
  ┌──────────────┐           ┌──────────────────┐
  │  CB Training │           │  CF Training     │
  │  Data        │           │  Data            │
  │  (movieId,   │           │  (userId,        │
  │   title,     │           │   movieId,       │
  │   text_comb, │           │   rating)        │
  │   genres_oh) │           └────────┬─────────┘
  └──────┬───────┘                    │
         │                            ▼
         ▼                    ┌──────────────────┐
  ┌──────────────┐            │  SVD (Surprise   │
  │  TF-IDF      │            │  library)        │
  │  Vectorizer  │            │  50 latent factor│
  │  5.000 fitur │            │  per film & user │
  └──────┬───────┘            └────────┬─────────┘
         │                             │
         ▼                             ▼
  ┌──────────────┐            ┌──────────────────┐
  │  FAISS Index │            │  cf_scores.pkl   │
  │  (L2/cosine) │            │  (predicted      │
  │  cbf_index   │            │   rating matrix) │
  │  .faiss      │            └────────┬─────────┘
  └──────┬───────┘                     │
         │                             │
         └──────────────┬──────────────┘
                        ▼
               ┌──────────────────┐
               │  SQLite Case Base│
               │  (cases.db)      │
               │  • film metadata │
               │  • index file    │
               │  • retained cases│
               └────────┬─────────┘
                        │
                        ▼
              ┌──────────────────────┐
              │   CBR 4R CYCLE       │
              │                      │
              │  RETRIEVE            │
              │  ├─ Filter kasar     │
              │  │  (I2/I3/I4)       │
              │  └─ Score Fusion     │
              │     0.6·CBF + 0.25·CF│
              │     + 0.15·Historis  │
              │                      │
              │  REUSE               │
              │  └─ Top-3 awal       │
              │                      │
              │  REVISE              │
              │  └─ Feedback user    │
              │     → expand ke Top-5│
              │                      │
              │  RETAIN              │
              │  └─ Simpan kasus baru│
              │     ke SQLite        │
              └────────┬─────────────┘
                       ▼
               Top-N Rekomendasi + Alasan
```

---

## 2. File dalam Project

### Files Utama

| File | Fungsi |
|------|--------|
| `preprocess.py` | Merge dataset, build text_combined, one-hot genre, filter, output CSV siap training |
| `train_cbf.py` | TF-IDF vectorization → FAISS index → simpan `cbf_index.faiss` + `tfidf_vectorizer.pkl` |
| `train_cf.py` | SVD via Surprise library → simpan `cf_model.pkl` + predicted rating matrix |
| `build_casebase.py` | Load FAISS + CF model → isi SQLite `cases.db` dengan metadata + index file |
| `recommend.py` | CBR pipeline utama (Retrieve → Reuse → Revise → Retain) |
| `evaluate.py` | Hitung Precision@K, Recall@K, NDCG, dan coverage case base |
| `PLAN.md` | Dokumentasi ini |

### Folder

| Folder | Isi |
|--------|-----|
| `dataset/` | Dataset mentah (tmdb_5000_movies.csv, ratings.csv, dll) |
| `output/` | Hasil preprocessing — CSV siap training |
| `models/` | Artifact hasil training (FAISS index, pickle model, SQLite DB) |

### Output Files (`output/`)

| File | Deskripsi |
|------|-----------|
| `cb_training.csv` | Film dengan text_combined + genres one-hot (±4.500 baris) |
| `cf_training.csv` | userId, movieId, rating — sudah difilter (subset MovieLens) |
| `index_file.csv` | Ringkasan numerik per film untuk filter kasar (I2/I3/I4) |
| `preprocessing_report.txt` | Laporan statistik sebelum/sesudah filter |

### Model Files (`models/`)

| File | Deskripsi |
|------|-----------|
| `cbf_index.faiss` | FAISS index vektor TF-IDF seluruh film |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer (untuk transform query baru) |
| `cf_model.pkl` | SVD model (Surprise) |
| `cases.db` | SQLite: tabel `films`, `index_file`, `retained_cases` |

---

## 3. Mengapa SQLite + FAISS, bukan Neo4j/NoSQL

Proyek sebelumnya (Mata Kuliah NoSQL) menggunakan Neo4j karena memang ditugaskan untuk NoSQL. Untuk sistem CBR ini, pilihan stack yang lebih tepat adalah **SQLite + FAISS**, dengan alasan:

| Aspek | Neo4j (NoSQL) | SQLite + FAISS (dipilih) |
|-------|--------------|--------------------------|
| **Operasi utama** | Graph traversal | Vector similarity search + tabel relasional sederhana |
| **Kemiripan film** | SIMILAR_TO relationship (pre-computed) | FAISS nearest-neighbor on-the-fly — lebih fleksibel untuk query baru |
| **Case retention** | Node + relationship baru | Append row ke tabel SQLite — jauh lebih ringan |
| **Instalasi** | Neo4j Desktop + driver | `pip install faiss-cpu sqlite3` — tanpa server |
| **Portabilitas** | Perlu instance Neo4j berjalan | Satu file `.db` + `.faiss` — bisa langsung dicommit ke repo |
| **Cocok untuk CBR?** | Kurang — CBR tidak butuh graph traversal multi-hop | Ya — CBR butuh vector search + simple key-value store untuk case base |

**Kesimpulan:** Neo4j masuk akal kalau sistem perlu memodelkan relasi kompleks multi-hop (misalnya "temukan aktor yang sering bermain bersama sutradara X"). Untuk CBR yang intinya adalah *nearest-neighbor retrieval + case storage*, FAISS + SQLite lebih simpel, lebih cepat, dan tanpa overhead server.

---

## 4. Preprocessing

### Langkah-langkah

| Step | Operasi | File Input | Catatan |
|------|---------|-----------|---------|
| A | Load & parse TMDB (genre JSON, cast, keywords) | `tmdb_5000_movies.csv` | Ambil top-5 cast, top-10 keywords |
| B | Title matching ke MovieLens (fuzzy match) | `movies.csv` (ML) | Gunakan `rapidfuzz`; threshold ≥ 90 |
| C | Build `text_combined` = overview + genres + cast + keywords | — | Lowercase, hapus stopwords, tanda baca |
| D | One-hot encode 19 genre | — | Kolom: `Action`, `Animation`, ..., `Western` |
| E | Filter: `original_language == 'en'`, `vote_count ≥ 20` | — | Buang film tanpa sinopsis |
| F | Filter rating: user dengan ≥ 5 rating | `ratings.csv` (ML) | Kurangi noise CF |
| G | Gabung TMDB + rating via movieId hasil matching | — | Inner join; film tanpa rating tetap masuk (CB-only) |

### Jalankan

```bash
python preprocess.py
```

Output muncul di folder `output/`.

---

## 5. Training

### Content-Based Filtering (CBF)

1. Baca `output/cb_training.csv` — ambil kolom `text_combined`
2. TF-IDF vectorization, `max_features=5000`, `ngram_range=(1,2)`
3. Normalisasi vektor (L2) → siap cosine similarity via inner product
4. Build FAISS `IndexFlatIP` (inner product = cosine setelah normalisasi)
5. Simpan index ke `models/cbf_index.faiss` dan vectorizer ke `models/tfidf_vectorizer.pkl`

### Collaborative Filtering (CF)

1. Baca `output/cf_training.csv`
2. Train SVD (`n_factors=50`, `n_epochs=20`) via Surprise library
3. Simpan model ke `models/cf_model.pkl`
4. Pre-compute predicted ratings untuk seluruh (user, film) pair yang belum dirating — simpan ke tabel `predicted_ratings` di SQLite (`cases.db`)

> **Catatan CF:** MovieLens 25M besar. Gunakan subset (misal 1M rating / 10k user) saat development, lalu full dataset saat final run.

### Build Case Base (SQLite)

```bash
python build_casebase.py
```

Mengisi `models/cases.db` dengan:

- Tabel `films`: movieId, title, genres, overview, vote_average, tfidf_vector_idx (posisi di FAISS)
- Tabel `index_file`: movieId, tfidf_norm, genre_vector (JSON), language, vote_avg, is_validated
- Tabel `retained_cases`: case_id, query_text, recommended_ids, feedback, timestamp

### Jalankan Training

```bash
python train_cbf.py
python train_cf.py
python build_casebase.py
```

Urutan harus sesuai. Durasi estimasi: CBF ~2 menit, CF ~10-30 menit (tergantung ukuran subset).

---

## 6. CBR Pipeline (4R Cycle)

### RETRIEVE — Dua Tahap

**Tahap 1: Filter Kasar (index_file)**

Sebelum menghitung cosine similarity penuh, saring dulu dengan 3 indikator:

| Indikator | Kondisi Lolos |
|-----------|--------------|
| I2 (kemiripan dokumen minimum) | `tfidf_norm ≥ 0.1` (film dengan sinopsis cukup panjang) |
| I3 (genre overlap) | Minimal 1 genre yang sama dengan query |
| I4 (filter noise) | `language == 'en'`, `vote_avg ≥ 5.0`, `is_validated == True` |

Film yang tidak lolos ketiga filter langsung dibuang tanpa masuk FAISS.

**Tahap 2: Score Fusion (film yang lolos filter)**

```python
skor_akhir = 0.60 * skor_cbf + 0.25 * skor_cf + 0.15 * skor_historis_cbr
```

- `skor_cbf`: cosine similarity dari FAISS (query TF-IDF vs film)
- `skor_cf`: predicted rating (SVD) ternormalisasi ke [0,1] — 0 jika user baru (cold start)
- `skor_historis_cbr`: frekuensi film muncul di `retained_cases` sebagai rekomendasi yang diterima

Hasil: daftar film diurutkan dari skor tertinggi.

### REUSE

Ambil Top-3 dari hasil Retrieve → sajikan ke pengguna sebagai rekomendasi awal beserta alasan:

```
Film: Toy Story 2
Alasan: Sangat mirip dengan "Toy Story" yang Anda sukai (genre Animation+Family,
        karakter Woody & Buzz muncul di sinopsis). Skor kesamaan konten: 0.87.
        Penonton dengan selera serupa memberi rating rata-rata 4.2/5.
```

### REVISE

Pengguna memberi feedback pada Top-3:

- **👍 Relevan** → film masuk ke daftar final
- **👎 Tidak relevan** → film dibuang, sistem expand ke Top-5 dengan mengambil film berikutnya dari Retrieve
- Film yang sudah ditandai tidak relevan tidak muncul lagi dalam sesi yang sama

### RETAIN

Setelah sesi selesai, simpan kasus baru ke SQLite:

```sql
INSERT INTO retained_cases
  (query_text, recommended_ids, accepted_ids, rejected_ids, timestamp)
VALUES (?, ?, ?, ?, ?);
```

Case base dioptimasi berkala (jalankan `python optimize_casebase.py`) dengan menghapus kasus yang terlalu mirip satu sama lain (`skor_cbf ≥ 0.95` dengan kasus lain yang sudah ada).

---

## 7. Cara Pakai

### A. Rekomendasi untuk User yang Sudah Ada

```python
from recommend import CBRRecommender

rec = CBRRecommender(
    db_path="models/cases.db",
    faiss_path="models/cbf_index.faiss",
    vectorizer_path="models/tfidf_vectorizer.pkl",
    cf_model_path="models/cf_model.pkl"
)

# User mengetik judul film yang disukai
results = rec.recommend(
    query_title="Toy Story",
    user_id=42,       # None jika user baru (cold start)
    top_k=3
)

for r in results:
    print(r["title"], r["score"], r["reason"])
```

### B. Cold Start (User Baru)

Jika `user_id=None`, sistem otomatis set `skor_cf = 0` dan `alpha_cf = 0` — rekomendasi murni berbasis konten:

```python
results = rec.recommend(query_title="The Dark Knight", user_id=None, top_k=5)
```

### C. Feedback & Retain

```python
rec.revise_and_retain(
    query_text="action superhero dark",
    results=results,
    accepted_ids=[272, 49026],   # movieId yang diterima
    rejected_ids=[155]           # movieId yang ditolak
)
```

### D. Evaluasi

```bash
python evaluate.py --k 5 --test_split 0.2
```

Output: `Precision@5`, `Recall@5`, `NDCG@5`, jumlah kasus di case base.

---

## 8. Query Rekomendasi (SQL + Python)

### Lihat film paling sering direkomendasikan (dari retained cases)

```sql
SELECT f.title, COUNT(*) AS frekuensi
FROM retained_cases rc
JOIN films f ON f.movieId IN (
    SELECT value FROM json_each(rc.accepted_ids)
)
GROUP BY f.movieId
ORDER BY frekuensi DESC
LIMIT 10;
```

### Lihat case base growth over time

```sql
SELECT DATE(timestamp) AS tanggal, COUNT(*) AS kasus_baru
FROM retained_cases
GROUP BY tanggal
ORDER BY tanggal;
```

### Debug: lihat skor komponen untuk satu query

```python
rec.explain(query_title="Inception", user_id=42)
# Output:
# CBF score: 0.81  (TF-IDF cosine similarity)
# CF score:  0.74  (SVD predicted rating, normalized)
# CBR hist:  0.12  (2x muncul di retained_cases)
# FINAL:     0.60*0.81 + 0.25*0.74 + 0.15*0.12 = 0.688
```

---

## Catatan Tambahan

- **Stopwords:** Gunakan `nltk.corpus.stopwords` (Inggris) saat cleaning `text_combined`
- **Title matching TMDB ↔ MovieLens:** Gunakan `rapidfuzz.fuzz.token_sort_ratio` dengan threshold 90. Film yang tidak cocok tetap masuk sebagai CB-only (tanpa skor CF).
- **Subset MovieLens:** Untuk development, ambil user dengan ≥ 50 rating agar SVD lebih stabil. File `output/cf_training.csv` bisa dikecilkan dengan flag `--max_users 5000` di `preprocess.py`.
- **FAISS variant:** `IndexFlatIP` cukup untuk ≤ 50k film. Jika dataset lebih besar, ganti ke `IndexIVFFlat` dengan `nlist=100` untuk kecepatan.
- **Skor historis CBR:** Dihitung sebagai `log(1 + frekuensi) / log(1 + max_frekuensi)` agar tidak mendominasi skor akhir.
