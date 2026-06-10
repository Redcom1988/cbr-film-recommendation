# PLAN.md — CBR Film Recommendation System

**Mata Kuliah:** Metode Penalaran  
**Topik:** Sistem Rekomendasi Film berbasis Case-Based Reasoning (CBR) + Hybrid Filtering  
**Dataset:** TMDB 5000 + MovieLens 25M  

---

## Daftar Isi

1. [Paradigma CBR](#1-paradigma-cbr)
2. [Arsitektur](#2-arsitektur)
3. [Struktur Kasus](#3-struktur-kasus)
4. [File dalam Project](#4-file-dalam-project)
5. [Mengapa SQLite + FAISS, bukan Neo4j/NoSQL](#5-mengapa-sqlite--faiss-bukan-neo4jnosql)
6. [Preprocessing](#6-preprocessing)
7. [Training](#7-training)
8. [CBR Pipeline (4R Cycle)](#8-cbr-pipeline-4r-cycle)
9. [Cara Pakai](#9-cara-pakai)
10. [Query Rekomendasi](#10-query-rekomendasi)
11. [Evaluasi](#11-evaluasi)

---

## 1. Paradigma CBR

Sistem ini menggunakan **Case-Based Reasoning** sebagai metode utama. Berbeda dari sistem lama yang menjadikan **film** sebagai objek pencarian, sistem baru ini menjadikan **kasus interaksi pengguna** sebagai unit utama.

```
Case-Based Reasoning Recommender
├─ Retrieve  → cari kasus lama paling mirip (case_index.faiss)
├─ Reuse     → gunakan solusi (accepted_ids) dari kasus lama
├─ Revise    → feedback pengguna (👍 like / 👎 dislike)
├─ Retain    → simpan kasus baru ke retained_cases
└─ CF        → faktor pendukung ranking (bukan sumber utama)
```

**Prinsip utama:**
- CBR adalah metode utama — setiap interaksi disimpan sebagai kasus
- CBF (TF-IDF + FAISS) hanya digunakan sebagai alat hitung *similarity*, bukan sumber rekomendasi langsung
- CF hanya membantu *ranking ulang* kandidat di tahap Reuse
- `retained_cases` adalah inti sistem, bukan sekadar log
- Sistem menggunakan **dua FAISS index terpisah**:
  - `case_index.faiss` → mencari kasus mirip (CBR Retrieve)
  - `movie_index.faiss` → menghitung `movie_similarity` antar film referensi

---

## 2. Arsitektur

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
         │                    ┌──────────────────┐
         │                    │  SVD (Surprise)  │
         │                    │  50 latent factor│
         │                    └────────┬─────────┘
         │                             │
         ▼                             ▼
  ┌──────────────────────────────────────────────┐
  │              TF-IDF Vectorizer               │
  │              max_features=5000               │
  └──────┬──────────────────┬───────────┬────────┘
         │                  │           │
         ▼                  ▼           ▼
  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐
  │ case_index  │  │ movie_index  │  │ cf_scores.pkl │
  │ .faiss      │  │ .faiss       │  │ (predicted    │
  │             │  │              │  │  rating matrix│
  │ Index kasus │  │ Index film   │  │  per user)    │
  │ (query+movie│  │ (overview+   │  └───────┬───────┘
  │ +genres)    │  │  genre+cast) │          │
  └──────┬──────┘  └──────┬───────┘          │
         │                │                  │
         └────────────────┴──────────────────┘
                          │
                          ▼
               ┌──────────────────┐
               │  SQLite Case Base│
               │  (cases.db)      │
               │  • films         │
               │  • retained_cases│
               │  • case_sim_cache│
               └────────┬─────────┘
                        │
              ┌─────────┴──────────┐
              │   Seed Cases       │
              │   (otomatis dari   │
              │   dataset TMDB+ML) │
              └─────────┬──────────┘
                        │
                        ▼

              ┌─────────────────────────┐
              │   User Input            │
              │   Query [wajib]         │
              │   Film Referensi [opsional]│
              └──────────┬──────────────┘
                         │
                         ▼
              ┌─────────────────┐
              │    RETRIEVE     │
              │                 │
              │  Vektorisasi    │
              │  input → TF-IDF │
              │                 │
              │  case_index     │
              │  .faiss         │
              │  → Top-5 kasus  │
              │                 │
              │  Hitung         │
              │  case_similarity│
              │  threshold ≥0.30│
              │                 │
              │  [fallback CBF] │
              │  jika tidak ada │
              │  kasus > 0.30   │
              └────────┬────────┘
                       │
                       ▼
              Top-5 Kasus Mirip
                       │
                       ▼
              ┌─────────────────┐
              │     REUSE       │
              │                 │
              │  Agregasi skor  │
              │  per film dari  │
              │  semua kasus    │
              │                 │
              │  Penalti untuk  │
              │  film di        │
              │  rejected_ids   │
              │                 │
              │  Ranking ulang: │
              │  0.70 * agg     │
              │  + 0.30 * cf    │
              │                 │
              │  Cold start:    │
              │  1.00 * agg     │
              └────────┬────────┘
                       │
                       ▼
              Top-K Rekomendasi
                       │
                       ▼
              ┌─────────────────┐
              │     REVISE      │
              │                 │
              │  👍 Like (3-5★) │
              │  👎 Dislike(1-2★)│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     RETAIN      │
              │                 │
              │  Simpan kasus   │
              │  baru ke        │
              │  retained_cases │
              │  (from_case_id  │
              │   nullable)     │
              └─────────────────┘
```

---

## 3. Struktur Kasus

Objek utama sistem adalah **kasus**, bukan film. Setiap kasus merepresentasikan satu sesi interaksi pengguna.

### Definisi Kasus

```
CASE
├─ case_id            → ID unik kasus
├─ user_id            → ID pengguna (untuk personalisasi CF)
├─ query_text         → teks yang diketik pengguna
├─ reference_movie    → film yang dijadikan acuan (opsional)
├─ recommended_ids    → film yang disarankan sistem
├─ accepted_ids       → film yang di-like pengguna (👍 / bintang 3-5)
├─ rejected_ids       → film yang di-dislike pengguna (👎 / bintang 1-2)
├─ from_case_id       → case_id sumber (NULL jika dari fallback CBF)
└─ timestamp          → waktu kasus dibuat
```

> **Catatan `user_id`:** Kolom ini wajib untuk memungkinkan CF melakukan re-ranking yang dipersonalisasi per pengguna. Tanpa `user_id`, CF tidak dapat membedakan preferensi antar pengguna.

> **Catatan `from_case_id`:** Bernilai NULL jika rekomendasi dihasilkan dari mode fallback CBF (bukan dari kasus lama). Setelah `retained_cases` memiliki cukup kasus, kolom ini terisi.

### Contoh Kasus

```json
{
  "case_id": 123,
  "user_id": 42,
  "query": "space exploration movie",
  "reference_movie": 157336,
  "recommended": [157336, 286217, 508442],
  "accepted": [508442],
  "rejected": [286217],
  "from_case_id": 88,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Representasi Kasus untuk FAISS

Setiap kasus direpresentasikan sebagai satu vektor gabungan dari:

```
case_embedding = TF-IDF(
  query_text + " " +
  title(reference_movie) + " " +
  genres(reference_movie)
)
```

Vektor ini diindeks ke `case_index.faiss` untuk pencarian kasus mirip.

### Perhitungan Case Similarity

```
case_similarity =
  0.50 × query_similarity     ← cosine TF-IDF query baru vs query kasus lama
+ 0.30 × movie_similarity     ← cosine movie_index.faiss (film referensi baru vs lama)
+ 0.20 × genre_overlap        ← Jaccard similarity genre film referensi
```

**Penyesuaian bobot jika input tidak lengkap:**

| Input Pengguna | Formula |
|----------------|---------|
| Query + Film Referensi | `0.50 × query + 0.30 × movie + 0.20 × genre` |
| Query saja | `0.70 × query + 0.30 × genre_from_query` |
| Query saja (tanpa genre) | `1.00 × query` |

---

## 4. File dalam Project

### Files Utama

| File | Fungsi |
|------|--------|
| `preprocess.py` | Merge dataset, build text_combined, one-hot genre, filter, output CSV siap training |
| `train_cbf.py` | TF-IDF vectorization → dua FAISS index (`case_index.faiss` + `movie_index.faiss`) |
| `train_cf.py` | SVD via Surprise library → simpan `cf_model.pkl` + predicted rating matrix |
| `build_casebase.py` | Isi SQLite `cases.db` + buat seed cases otomatis dari TMDB + MovieLens |
| `recommend.py` | CBR pipeline utama (Retrieve → Reuse → Revise → Retain) |
| `evaluate.py` | Hitung Precision@K, Recall@K, NDCG — ground truth dari `accepted_ids` |
| `optimize_casebase.py` | Hapus kasus redundan (`case_similarity ≥ 0.95`) dari case base |
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
| `seed_cases.csv` | Kasus seed yang dibuat otomatis untuk bootstrap case base |
| `preprocessing_report.txt` | Laporan statistik sebelum/sesudah filter |

### Model Files (`models/`)

| File | Deskripsi |
|------|-----------|
| `case_index.faiss` | FAISS index vektor kasus (query + film + genre) — untuk Retrieve |
| `movie_index.faiss` | FAISS index vektor film (overview + genre + cast) — untuk `movie_similarity` |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer (dipakai untuk kedua index) |
| `cf_model.pkl` | SVD model (Surprise) |
| `cases.db` | SQLite: tabel `films`, `retained_cases`, `case_similarity_cache` |

---

## 5. Mengapa SQLite + FAISS, bukan Neo4j/NoSQL

Proyek sebelumnya (Mata Kuliah NoSQL) menggunakan Neo4j karena memang ditugaskan untuk NoSQL. Untuk sistem CBR ini, pilihan stack yang lebih tepat adalah **SQLite + FAISS**, dengan alasan:

| Aspek | Neo4j (NoSQL) | SQLite + FAISS (dipilih) |
|-------|--------------|--------------------------|
| **Operasi utama** | Graph traversal | Vector similarity search + tabel relasional sederhana |
| **Kemiripan kasus** | SIMILAR_TO relationship (pre-computed) | FAISS nearest-neighbor on-the-fly — lebih fleksibel untuk query baru |
| **Case retention** | Node + relationship baru | Append row ke tabel SQLite — jauh lebih ringan |
| **Instalasi** | Neo4j Desktop + driver | `pip install faiss-cpu sqlite3` — tanpa server |
| **Portabilitas** | Perlu instance Neo4j berjalan | Satu file `.db` + `.faiss` — bisa langsung dicommit ke repo |
| **Cocok untuk CBR?** | Kurang — CBR tidak butuh graph traversal multi-hop | Ya — CBR butuh vector search + simple key-value store untuk case base |

**Kesimpulan:** Neo4j masuk akal kalau sistem perlu memodelkan relasi kompleks multi-hop. Untuk CBR yang intinya adalah *nearest-neighbor case retrieval + case storage*, FAISS + SQLite lebih simpel, lebih cepat, dan tanpa overhead server.

---

## 6. Preprocessing

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
| H | Generate seed cases dari data film + rating | TMDB + ML | Output: `output/seed_cases.csv` |

### Jalankan

```bash
python preprocess.py
```

Output muncul di folder `output/`.

---

## 7. Training

### Dua FAISS Index Terpisah

Sistem menggunakan dua FAISS index dengan peran yang berbeda dan jelas:

| Index | File | Objek yang Diindeks | Digunakan untuk |
|-------|------|---------------------|-----------------|
| **Case Index** | `case_index.faiss` | Vektor kasus (`query_text + title(ref_movie) + genres`) | Retrieve kasus mirip (CBR utama) |
| **Movie Index** | `movie_index.faiss` | Vektor film (`overview + genre + cast + keywords`) | Hitung `movie_similarity` komponen `case_similarity` |

Pemisahan ini membuat peran masing-masing index jelas secara akademik dan menghindari pencampuran antara pencarian kasus dan pencarian film.

### Langkah Training CBF

```bash
python train_cbf.py
```

1. Baca `output/cb_training.csv`
2. TF-IDF vectorization, `max_features=5000`, `ngram_range=(1,2)`
3. Normalisasi vektor (L2) → siap cosine similarity via inner product
4. Build `movie_index.faiss` dari vektor `text_combined` seluruh film
5. Build `case_embedding` untuk setiap seed case → build `case_index.faiss`
6. Simpan kedua index dan vectorizer ke folder `models/`

### Collaborative Filtering (CF) — Faktor Pendukung Ranking

CF **tidak** menjadi sumber rekomendasi utama. CF hanya digunakan untuk re-ranking kandidat di tahap Reuse:

```
# User dikenali
final_score = 0.70 × aggregated_case_score + 0.30 × cf_score

# Cold start (user baru / user_id tidak dikenali)
final_score = 1.00 × aggregated_case_score
```

Langkah training:

```bash
python train_cf.py
```

1. Baca `output/cf_training.csv`
2. Train SVD (`n_factors=50`, `n_epochs=20`) via Surprise library
3. Simpan model ke `models/cf_model.pkl`
4. Pre-compute predicted ratings → simpan ke tabel `predicted_ratings` di `cases.db`

> **Catatan CF:** MovieLens 25M besar. Gunakan subset (misal 1M rating / 10k user) saat development, lalu full dataset saat final run.

### Build Case Base (SQLite)

```bash
python build_casebase.py
```

Mengisi `models/cases.db` dengan:
- Tabel `films` → metadata seluruh film
- Tabel `retained_cases` → seed cases awal + kasus dari interaksi pengguna
- Tabel `case_similarity_cache` → cache opsional untuk mempercepat Retrieve

### Seed Cases

Sebelum ada interaksi pengguna, case base diisi dengan **seed cases** yang dibuat otomatis dari kombinasi data film TMDB dan rating MovieLens. Seed cases berfungsi sebagai "pengalaman pertama" sistem sehingga Retrieve dapat bekerja sejak awal tanpa menunggu interaksi nyata.

Contoh seed case yang dibuat otomatis:
```json
{
  "case_id": 1,
  "user_id": null,
  "query": "space sci-fi adventure",
  "reference_movie": 157336,
  "recommended": [286217, 508442, 329865],
  "accepted": [286217, 508442],
  "rejected": [],
  "from_case_id": null,
  "timestamp": "2024-01-01T00:00:00"
}
```

### Urutan Jalankan Training

```bash
python train_cbf.py
python train_cf.py
python build_casebase.py
```

Urutan harus sesuai. Durasi estimasi: CBF ~2 menit, CF ~10-30 menit (tergantung ukuran subset).

---

## 8. CBR Pipeline (4R Cycle)

### RETRIEVE — Cari Kasus Mirip

Sistem mencari kasus lama di `retained_cases` yang paling mirip dengan kasus baru menggunakan `case_index.faiss`.

**Langkah Retrieve:**

```
1. Terima input pengguna: query_text [wajib] + reference_movie [opsional]
2. Buat case_embedding dari input
3. Cari Top-5 kasus mirip di case_index.faiss
4. Hitung case_similarity untuk setiap kasus
5. Filter: buang kasus dengan case_similarity < 0.30
6. Jika tidak ada kasus yang lolos threshold → fallback ke CBF
```

**Parameter Retrieve:**

| Parameter | Nilai Default | Keterangan |
|-----------|--------------|------------|
| `top_k_cases` | 5 | Jumlah kasus yang diambil per Retrieve |
| `min_similarity` | 0.30 | Threshold minimum case_similarity |
| `min_casebase_size` | 10 | Minimum kasus agar CBR aktif (dapat dikonfigurasi) |
| `brute_force_threshold` | 1000 | Gunakan brute-force jika kasus < 1000 |

**Rumus Case Similarity:**

```
case_similarity =
  0.50 × query_similarity     ← cosine TF-IDF (query baru vs query kasus lama)
+ 0.30 × movie_similarity     ← cosine movie_index.faiss (film ref baru vs lama)
+ 0.20 × genre_overlap        ← Jaccard similarity genre film referensi
```

**Penyesuaian bobot jika input tidak lengkap:**

| Kondisi Input | Formula |
|---------------|---------|
| Query + Film Referensi | `0.50 × query + 0.30 × movie + 0.20 × genre` |
| Query saja | `0.70 × query + 0.30 × genre` |
| Query saja (tanpa genre) | `1.00 × query` |

**Fallback CBF:**

Jika semua kasus yang ditemukan memiliki `case_similarity < 0.30` (atau case base masih kosong / `< min_casebase_size`), sistem melakukan fallback ke CBF:
- Cari film mirip langsung dari `movie_index.faiss`
- Hasilnya tetap disimpan ke `retained_cases` setelah feedback
- `from_case_id = NULL` pada kasus yang dihasilkan dari fallback

**Contoh Retrieve:**

```
Kasus Baru:
  Query:           "space exploration movie"
  Film Referensi:  Interstellar (movieId=157336)

Kasus di retained_cases yang ditemukan:
  case_id=88 | query="space sci-fi movie" | film=Interstellar

Hitung:
  query_similarity = 0.85  (TF-IDF cosine)
  movie_similarity = 1.00  (film sama: Interstellar)
  genre_overlap    = 0.90  (Jaccard)

  case_similarity = 0.50×0.85 + 0.30×1.00 + 0.20×0.90
                  = 0.425 + 0.300 + 0.180 = 0.905 ✓ (> 0.30)
```

---

### REUSE — Gunakan Solusi Kasus Lama

Rekomendasi diambil dari `accepted_ids` kasus-kasus mirip. Jika sebuah film muncul di beberapa kasus, skornya diagregasikan — bukan union biasa.

**Agregasi Skor:**

```python
aggregated_case_score[film] = Σ case_similarity(kasus yang mengandung film di accepted_ids)
```

**Contoh Agregasi:**

```
Kasus A (case_similarity = 0.85) → accepted: [Arrival, Gravity]
Kasus B (case_similarity = 0.80) → accepted: [Arrival, The Martian]

aggregated_case_score:
  Arrival     = 0.85 + 0.80 = 1.65  ← muncul di 2 kasus, dianggap lebih terpercaya
  Gravity     = 0.85
  The Martian = 0.80

→ Normalisasi ke [0,1], lalu re-ranking dengan CF
```

**Penalti untuk Film yang Pernah Ditolak:**

Jika film X muncul di `accepted_ids` kasus A dan `rejected_ids` kasus B, skor akhirnya dikurangi:

```python
net_score[film] = aggregated_accepted_score - aggregated_rejected_score
# Buang film dengan net_score ≤ 0
```

**Formula Final Score:**

```python
# User dikenali
final_score = 0.70 * aggregated_case_score_normalized + 0.30 * cf_score

# Cold start (user baru / user_id tidak dikenali di CF)
final_score = 1.00 * aggregated_case_score_normalized
```

Hasil: **Top-K rekomendasi** disajikan ke pengguna.

---

### REVISE — Feedback Pengguna

Pengguna memberikan feedback terhadap rekomendasi yang diberikan:

| Feedback | Keterangan |
|----------|------------|
| 👍 **Like** / ⭐ Bintang 3–5 | Rekomendasi akurat → masuk ke `accepted_ids` kasus baru |
| 👎 **Dislike** / ⭐ Bintang 1–2 | Rekomendasi kurang tepat → masuk ke `rejected_ids`, menjadi koreksi agar sistem tidak mengulangi kesalahan yang sama |

Film yang ditolak tidak akan muncul kembali dalam sesi yang sama.

---

### RETAIN — Simpan Kasus Baru

Setelah sesi selesai, kasus baru disimpan ke `retained_cases`:

```sql
INSERT INTO retained_cases
  (user_id, query_text, reference_movie,
   recommended_ids, accepted_ids, rejected_ids,
   from_case_id, timestamp)
VALUES (?, ?, ?, ?, ?, ?, ?, ?);
```

Setelah retain, `case_index.faiss` diperbarui dengan vektor kasus baru.

> **Optimasi opsional:** Jalankan `python optimize_casebase.py` untuk menghapus kasus yang terlalu mirip (`case_similarity ≥ 0.95`) agar case base tetap efisien dan tidak redundan.

---

## 9. Cara Pakai

### A. Rekomendasi — Query + Film Referensi

```python
from recommend import CBRRecommender

rec = CBRRecommender(
    db_path="models/cases.db",
    case_index_path="models/case_index.faiss",
    movie_index_path="models/movie_index.faiss",
    vectorizer_path="models/tfidf_vectorizer.pkl",
    cf_model_path="models/cf_model.pkl"
)

results = rec.recommend(
    query_text="space exploration movie",
    reference_movie="Interstellar",   # opsional
    user_id=42,                       # None = cold start
    top_k=5
)

for r in results:
    print(r["title"], r["final_score"], r["from_case_id"])
```

### B. Rekomendasi — Query Saja (Tanpa Film Referensi)

```python
results = rec.recommend(
    query_text="dark psychological thriller",
    reference_movie=None,    # bobot dinormalisasi otomatis
    user_id=None,            # cold start → final_score = 1.00 × case_score
    top_k=5
)
```

### C. Feedback & Retain

```python
rec.revise_and_retain(
    user_id=42,
    query_text="space exploration movie",
    reference_movie="Interstellar",
    recommended_ids=[508442, 286217, 329865],
    accepted_ids=[508442],          # 👍 / bintang 3-5
    rejected_ids=[286217],          # 👎 / bintang 1-2
    from_case_id=88                 # case_id sumber (None jika fallback CBF)
)
```

### D. Evaluasi

```bash
python evaluate.py --k 5 --test_split 0.2
```

Output: `Precision@5`, `Recall@5`, `NDCG@5`, jumlah kasus di case base.

---

## 10. Query Rekomendasi (SQL + Python)

### Lihat film paling sering diterima (accepted)

```sql
SELECT f.title, COUNT(*) AS frekuensi_diterima
FROM retained_cases rc
JOIN films f ON f.movieId IN (
    SELECT value FROM json_each(rc.accepted_ids)
)
GROUP BY f.movieId
ORDER BY frekuensi_diterima DESC
LIMIT 10;
```

### Lihat film yang sering ditolak (rejected)

```sql
SELECT f.title, COUNT(*) AS frekuensi_ditolak
FROM retained_cases rc
JOIN films f ON f.movieId IN (
    SELECT value FROM json_each(rc.rejected_ids)
)
GROUP BY f.movieId
ORDER BY frekuensi_ditolak DESC
LIMIT 10;
```

### Lihat case base growth over time

```sql
SELECT DATE(timestamp) AS tanggal, COUNT(*) AS kasus_baru
FROM retained_cases
GROUP BY tanggal
ORDER BY tanggal;
```

### Lihat kasus yang berasal dari fallback CBF

```sql
SELECT COUNT(*) AS kasus_fallback
FROM retained_cases
WHERE from_case_id IS NULL;
```

### Debug: lihat skor komponen untuk satu query

```python
rec.explain(
    query_text="space exploration movie",
    reference_movie="Interstellar",
    user_id=42
)
# Output:
# Top-5 kasus mirip:
#   case_id=88 | query="space sci-fi movie" | film=Interstellar
#     query_similarity : 0.85
#     movie_similarity : 1.00
#     genre_overlap    : 0.90
#     case_similarity  : 0.905
#
#   case_id=72 | query="astronaut survival film" | film=Gravity
#     query_similarity : 0.72
#     movie_similarity : 0.65
#     genre_overlap    : 0.80
#     case_similarity  : 0.717
#
# Agregasi skor kandidat:
#   Arrival     : 0.905 + 0.717 = 1.622 (muncul di 2 kasus)
#   The Martian : 0.905
#   Gravity     : 0.717 - 0.500 = 0.217 (penalti: ada di rejected case lain)
#
# Re-ranking dengan CF (user_id=42):
#   Arrival      → final_score = 0.70×0.89 + 0.30×0.82 = 0.869
#   The Martian  → final_score = 0.70×0.50 + 0.30×0.74 = 0.572
#   Gravity      → dibuang (net_score rendah setelah penalti)
```

---

## 11. Evaluasi

### Ground Truth

Ground truth untuk evaluasi berasal dari `accepted_ids` pada kasus-kasus di `retained_cases` (termasuk seed cases). Film yang diterima pengguna dianggap sebagai solusi benar yang seharusnya berhasil ditemukan kembali oleh sistem.

### Prosedur Evaluasi

1. Seed cases dipisah: 80% sebagai case base, 20% sebagai test set
2. Untuk setiap kasus test, sistem melakukan Retrieve + Reuse tanpa melihat `accepted_ids`-nya
3. Hasil rekomendasi dibandingkan dengan `accepted_ids` ground truth

### Metrik

```bash
python evaluate.py --k 5 --test_split 0.2
```

| Metrik | Keterangan |
|--------|-----------|
| `Precision@K` | Proporsi rekomendasi Top-K yang ada di `accepted_ids` |
| `Recall@K` | Proporsi `accepted_ids` yang berhasil ditemukan di Top-K |
| `NDCG@K` | Kualitas ranking — film yang lebih relevan harus muncul lebih atas |
| `Coverage` | Jumlah kasus unik di case base saat evaluasi |

---

## Struktur Database

### Tabel `films`

| Kolom | Tipe | Keterangan |
|-------|------|-----------|
| `movieId` | INTEGER PK | ID film dari MovieLens |
| `title` | TEXT | Judul film |
| `overview` | TEXT | Sinopsis |
| `genre` | TEXT | Genre (JSON array) |
| `tfidf_vector_idx` | INTEGER | Posisi vektor di `movie_index.faiss` |

### Tabel `retained_cases` *(inti sistem)*

| Kolom | Tipe | Keterangan |
|-------|------|-----------|
| `case_id` | INTEGER PK | ID kasus |
| `user_id` | INTEGER | ID pengguna — untuk personalisasi CF (NULL pada seed cases) |
| `query_text` | TEXT | Query yang diketik pengguna |
| `reference_movie` | INTEGER | movieId film referensi (NULL jika tidak diisi) |
| `recommended_ids` | TEXT | JSON array movieId yang direkomendasikan |
| `accepted_ids` | TEXT | JSON array movieId yang diterima (👍) — ground truth evaluasi |
| `rejected_ids` | TEXT | JSON array movieId yang ditolak (👎) |
| `from_case_id` | INTEGER | case_id sumber (NULL jika dari fallback CBF atau seed) |
| `timestamp` | TEXT | Waktu kasus dibuat |

### Tabel `case_similarity_cache` *(opsional)*

| Kolom | Tipe | Keterangan |
|-------|------|-----------|
| `case_a` | INTEGER | case_id kasus A |
| `case_b` | INTEGER | case_id kasus B |
| `similarity` | REAL | Nilai case_similarity antara A dan B |

> Digunakan untuk mempercepat Retrieve dengan meng-cache similarity antar kasus yang sudah pernah dihitung.

---

## Catatan Tambahan

- **Stopwords:** Gunakan `nltk.corpus.stopwords` (Inggris) saat cleaning `text_combined`
- **Title matching TMDB ↔ MovieLens:** Gunakan `rapidfuzz.fuzz.token_sort_ratio` threshold 90. Film tanpa match tetap masuk sebagai CB-only (tanpa skor CF).
- **Subset MovieLens:** Untuk development, ambil user dengan ≥ 50 rating agar SVD lebih stabil.
- **FAISS variant:** `IndexFlatIP` cukup untuk ≤ 50k vektor. Jika dataset lebih besar, ganti ke `IndexIVFFlat` dengan `nlist=100`.
- **Posisi CBF:** CBF adalah *alat ukur similarity* (melalui `movie_index.faiss`), bukan sumber rekomendasi. Ini yang membedakan arsitektur CBR murni dari sistem hybrid biasa.
- **Cold Start:** Saat `retained_cases < min_casebase_size (default: 10)`, sistem fallback ke CBF langsung. Nilai ini dapat dikonfigurasi dan disesuaikan berdasarkan hasil evaluasi.
- **Brute-force Retrieve:** Digunakan jika jumlah kasus < 1000 sebagai alternatif FAISS yang lebih sederhana secara implementasi.
- **Pembaruan case_index:** Setiap kali kasus baru di-retain, vektor case_embedding-nya ditambahkan ke `case_index.faiss`.
