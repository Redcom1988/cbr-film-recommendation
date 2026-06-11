# Walkthrough — CBR Film Recommendation System

## Ringkasan

Sistem rekomendasi film berbasis **Case-Based Reasoning (CBR)** berhasil diimplementasikan end-to-end, mulai dari database, training, CBR engine, hingga web UI.

---

## Screenshots

![Halaman Awal CineCBR](file:///C:/Users/USER/.gemini/antigravity-ide/brain/27a3fb5c-ac38-4ded-9f3d-1765bb1c8327/initial_page_1781109702502.png)
*Tampilan awal: Netflix dark theme, CBR 4R Flow panel, statistik case base*

![Hasil Rekomendasi](file:///C:/Users/USER/.gemini/antigravity-ide/brain/27a3fb5c-ac38-4ded-9f3d-1765bb1c8327/recommendation_results_1781109731996.png)
*Kartu rekomendasi dengan skor CBR%, CF%, dan tombol Like/Dislike*

---

## File yang Dibuat / Diubah

| File | Status | Deskripsi |
|------|--------|-----------|
| [build_casebase.py](file:///c:/Users/USER/Downloads/cbr-film-recommendation/build_casebase.py) | Dibuat Ulang | Inisialisasi DB + populate films + 500 seed cases |
| [train_cbf.py](file:///c:/Users/USER/Downloads/cbr-film-recommendation/train_cbf.py) | Baru | TF-IDF + movie_index.faiss + case_index.faiss |
| [train_cf.py](file:///c:/Users/USER/Downloads/cbr-film-recommendation/train_cf.py) | Baru | SVD training (RMSE=0.8937) |
| [recommend.py](file:///c:/Users/USER/Downloads/cbr-film-recommendation/recommend.py) | Baru | CBRRecommender class — 4R cycle engine |
| [app.py](file:///c:/Users/USER/Downloads/cbr-film-recommendation/app.py) | Baru | Flask server + API routes |
| [templates/index.html](file:///c:/Users/USER/Downloads/cbr-film-recommendation/templates/index.html) | Baru | Netflix-dark Web UI + Font Awesome |
| [evaluate.py](file:///c:/Users/USER/Downloads/cbr-film-recommendation/evaluate.py) | Baru | Precision/Recall/NDCG evaluation |

---

## Hasil Evaluasi

Split: 80% train / 20% test dari 500 seed cases

| Metrik | Nilai |
|--------|-------|
| **Precision@5** | **0.4820** |
| **Recall@5** | **0.8342** |
| **NDCG@5** | **0.7737** |
| Kasus dievaluasi | 100 |
| Case base size | 500 |

> Recall@5 yang tinggi (0.83) menunjukkan sistem berhasil menemukan kembali sebagian besar film yang relevan dari case base. NDCG@5 yang baik (0.77) menunjukkan urutan rekomendasi sudah tepat.

---

## Cara Menjalankan Ulang dari Nol

```powershell
# 1. Build case base + seed cases
python build_casebase.py

# 2. Train TF-IDF + dua FAISS index
python train_cbf.py

# 3. Train SVD (Collaborative Filtering)
python train_cf.py

# 4. Jalankan web app
python app.py
# Buka: http://localhost:5001
```

---

## Arsitektur Akhir

```
User Query
    │
    ▼
[RETRIEVE] case_index.faiss
    │  case_similarity = 0.50*query + 0.30*movie + 0.20*genre
    │  threshold = 0.30  |  top_k = 5
    │  fallback: movie_index.faiss jika kasus < 10 atau sim < 0.30
    ▼
Top-5 Similar Cases
    │
    ▼
[REUSE] Agregasi accepted_ids
    │  aggregated_case_score[film] = Σ case_similarity
    │  Penalti: kurangi rejected_score
    │  final_score = 0.70 * case_score + 0.30 * cf_score
    ▼
Rekomendasi Top-K
    │
    ▼
[REVISE] Like / Dislike
    │
    ▼
[RETAIN] → retained_cases + update case_index live
```

---

## Catatan Teknis

- **TF-IDF**: `(genres_clean + " ") * 3 + overview_clean` — genre diberi bobot 3× lebih tinggi
- **Dua FAISS index**: `case_index.faiss` untuk Retrieve, `movie_index.faiss` untuk komponen `movie_similarity`
- **CF**: SVD dari MovieLens Small (671 user, 6003 film) — hanya sebagai faktor pendukung ranking (30%)
- **Cold start**: `final_score = 1.00 × case_score` (tanpa CF)
- **Live update**: Setiap `retain()` langsung memperbarui `case_index.faiss` dan `case_ids.pkl`
- **Web**: Flask + Font Awesome 6.5 + Inter font — berjalan di `http://localhost:5001`
