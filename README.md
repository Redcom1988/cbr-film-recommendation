# CBR Film Recommendation System

Sistem Rekomendasi Film berbasis **Case-Based Reasoning (CBR)** dengan pendekatan **Hybrid Filtering** menggunakan dataset TMDB 5000 dan MovieLens.

Berbeda dengan sistem rekomendasi tradisional yang berfokus pada kemiripan antar-film secara langsung, sistem ini menggunakan **interaksi/kasus pengguna** sebagai entitas utama pencarian (Case Base).

---

## 🚀 Paradigma CBR (4R Cycle)

Alur kerja sistem rekomendasi diimplementasikan secara utuh mengikuti siklus 4R CBR:

1. **Retrieve**: Mencari Top-K kasus lama yang paling mirip dengan query pengguna (`query_text`, `reference_movie`, `genres`). Proses ini menggunakan case embedding yang diindeks dengan **FAISS** (`case_index.faiss`) untuk pencarian kemiripan kosinus (TF-IDF) secara instan.
2. **Reuse**: Mengambil solusi (`accepted_ids` / film yang disukai) dari kasus lama yang mirip, menormalkannya, lalu mengombinasikannya dengan prediksi rating **SVD Collaborative Filtering** (MovieLens Small) untuk menghasilkan ranking rekomendasi akhir.
3. **Revise**: Pengguna memberikan umpan balik secara langsung melalui Web UI dengan menyukai (👍 **Like**) atau tidak menyukai (👎 **Dislike**) film yang direkomendasikan.
4. **Retain**: Menyimpan interaksi baru tersebut sebagai kasus baru ke dalam database (`retained_cases`) dan memperbarui index FAISS secara dinamis agar sistem menjadi lebih cerdas pada pencarian berikutnya.

---

## 📁 Struktur Proyek

```text
cbr-film-recommendation/
│
├── app.py                     # Flask Web Server & API Endpoint
├── recommend.py               # Core CBRRecommender Class (CBR & Hybrid Pipeline)
├── build_casebase.py          # Skrip inisialisasi Database SQLite & 500 Seed Cases
├── train_cbf.py               # Melatih TF-IDF & membangun index FAISS (movie & case index)
├── train_cf.py                # Melatih model SVD Collaborative Filtering
├── evaluate.py                # Skrip evaluasi performa CBR (Precision@5, Recall@5, NDCG@5)
├── requirements.txt           # Dependensi Python library
├── PLAN.md                    # Dokumentasi arsitektur dan perencanaan detail
├── README.md                  # Panduan proyek
│
├── models/                    # (Diabaikan git) SQLite DB & model binary (.faiss, .pkl)
└── output/                    # (Diabaikan git) Dataset hasil pre-processing (.csv)
```

---

## 🛠️ Instalasi & Setup

### 1. Clone & Masuk ke Folder Project
```bash
git clone https://github.com/Redcom1988/cbr-film-recommendation.git
cd cbr-film-recommendation
```

### 2. Buat Virtual Environment & Install Dependensi
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Install dependensi
pip install -r requirements.txt
```

### 3. Jalankan Pipeline Inisialisasi & Training
Untuk men-generate dataset, membangun database case base, melatih TF-IDF/FAISS index, serta Collaborative Filtering:

```bash
# Step 1: Inisialisasi Database & Seed Cases (500 seed cases)
python build_casebase.py

# Step 2: Bangun Movie & Case Index (FAISS)
python train_cbf.py

# Step 3: Latih Model Collaborative Filtering (SVD)
python train_cf.py
```

Setelah langkah di atas selesai, folder `models/` dan `output/` akan terisi otomatis dengan database SQLite, model pickle, dan file FAISS.

---

## 📊 Evaluasi Performa (Precision@5, Recall@5, NDCG@5)

Skrip evaluasi memisahkan 20% data kasus secara acak untuk menguji seberapa baik sistem memprediksi kembali film yang disukai (`accepted_ids`).

Untuk menjalankan evaluasi:
```bash
python evaluate.py
```

### Hasil Evaluasi Terakhir:
- **Precision@5**: `0.4820` (Rata-rata ~2.4 film dalam Top-5 sesuai dengan preferensi user)
- **Recall@5**: `0.8342` (Berhasil mencakup ~83% film yang disukai user dari total history)
- **NDCG@5**: `0.7737` (Ranking film yang relevan berada di posisi atas rekomendasi)
- **Case Base Size**: `500 kasus`

---

## 💻 Cara Menjalankan Web UI

Web UI dibuat dengan tema gelap premium (Netflix-style) yang menampilkan visualisasi proses CBR secara lengkap:
- Autocomplete pencarian film referensi.
- Panel detail skor kemiripan kasus lama yang ditemukan.
- Interaksi Like (👍) dan Dislike (👎) untuk revisi kasus.
- Penambahan kasus baru secara dinamis ke case base secara visual.

Untuk memulai web server Flask:
```bash
python app.py
```
Aplikasi akan berjalan di [http://localhost:5001](http://localhost:5001).

---

## 🔧 Perubahan Terbaru

### 1. Persistent User ID (Multi-User Support)
- Sebelumnya: `user_id` tidak pernah dikirim dari frontend — semua kasus tersimpan sebagai seed case (`user_id = NULL`).
- Sekarang: Setiap browser mendapat `user_id` unik yang disimpan di `localStorage`, dikirim otomatis di setiap request.
- Dampak: Kasus yang disimpan terkait dengan user tertentu, memungkinkan CF Score dan riwayat pribadi.

### 2. Fallback CBF — Label Akurat
- Saat case base belum cukup, sistem jatuh ke Content-Based Filtering (TF-IDF + FAISS).
- **CBF** (bukan CBR) ditampilkan di kartu rekomendasi, dengan `cf_score = 0`.
- Bobot `W_COLD = (1.0, 0.0, 0.0)` — murni content similarity.

### 3. Overview Placeholder Dibersihkan
- Dataset TMDB menyimpan `"No overview found."`, `"overview found"`, `"overview yet"`, dll. untuk film tanpa sinopsis.
- Method `_clean_overview()` menyaring placeholder tersebut menjadi string kosong.

### 4. Novelty Penalty Diperkuat (85%)
- Sebelumnya: 35% — belum cukup menggeser film yang sudah pernah di-like.
- Sekarang: 85% — film yang sudah pernah di-accept user turun ke 15% skor asli.
- Mencegah positive feedback loop tanpa hard-exclusion.

### 5. Fix Shape FAISS
- `faiss.search()` dan `faiss.add()` membutuhkan array 2D `(n, d)`.
- `_vec()` mengembalikan 1D via `.ravel()`, ditambahkan `.reshape(1, -1)` di semua pemanggilan.
