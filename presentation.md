---
marp: true
theme: default
class: lead
backgroundColor: #f8f9fa
---

# CineCBR: Sistem Rekomendasi Film Hybrid
**Menguraikan Tahap Awal Pengembangan hingga Implementasi**

*Presentasi Proyek*

---

## Daftar Isi
1. **Pendahuluan & Konsep Dasar**
2. **Tahap 1:** Data Preparation & Preprocessing
3. **Tahap 2:** Knowledge Engineering (Basis Kasus)
4. **Tahap 3:** Pemodelan (FAISS & SVD)
5. **Tahap 4:** Implementasi CBR (4R Cycle)
6. **Tahap 5:** Integrasi & UI (Web Application)
7. **Kesimpulan & Tanya Jawab**

---

## 1. Pendahuluan & Konsep Dasar

**Tantangan:** 
Sistem rekomendasi tradisional (seperti murni *Collaborative Filtering*) sering mengalami masalah *cold-start* (ketika pengguna baru tidak memiliki riwayat rating). 

**Solusi (CineCBR):** 
Menggunakan pendekatan **Case-Based Reasoning (CBR)** dipadukan dengan **Hybrid Filtering**:
- *Content-Based Filtering* (Memahami sinopsis & genre film)
- *Collaborative Filtering* (Memahami pola preferensi pengguna lain)

Sistem akan memecahkan masalah baru berdasarkan solusi dari "kasus-kasus" lama yang relevan.

---

## 2. Tahap 1: Data Preparation & Preprocessing

Data adalah pondasi utama. Kami menggunakan dataset TMDB 5000 dan MovieLens.

- **Pembersihan Data (`preprocess.py`):**
  - Menghapus film dengan bahasa non-Inggris dan *vote count* rendah.
  - Normalisasi teks (menghapus tanda baca, *lowercasing*).
  - Melakukan *fuzzy matching* untuk menyatukan dataset ID TMDB dengan ID MovieLens.
- **Output:** `movies_clean.csv`, `content_based_training.csv`, dan `collaborative_training.csv`.

---

## 3. Tahap 2: Knowledge Engineering (Basis Kasus)

Pada CBR, "Pengetahuan" disimpan dalam sebuah *Case Base* (Basis Kasus).
*Skrip: `build_casebase.py`*

- **Desain Database:** Menggunakan SQLite (`cases.db`) agar ringan dan persisten.
- **Tabel `films`:** Menyimpan informasi katalog film dan letak indeks fiturnya.
- **Tabel `retained_cases`:** Menyimpan riwayat pencarian (kasus) dan film mana yang disukai/ditolak.
- **Seed Cases:** Karena sistem masih baru, kami membangkitkan 500 *seed cases* simulasi agar sistem sudah pintar sejak hari pertama beroperasi.

---

## 4. Tahap 3: Pemodelan (AI & Machine Learning)

Sebelum mesin bisa berpikir, ia perlu dilatih:

- **Content-Based Model (`train_cbf.py`):**
  - Teks sinopsis dan genre diubah menjadi angka menggunakan **TF-IDF Vectorizer**.
  - Ruang vektor tersebut diindeks menggunakan **FAISS** (metode *similarity search* yang sangat cepat).
- **Collaborative Model (`train_cf.py`):**
  - Menggunakan algoritma **SVD (Singular Value Decomposition)** dari library *Surprise*.
  - Menemukan pola tersembunyi (*latent factors*) dari 96.000+ data rating pengguna.

---

## 5. Tahap 4: Implementasi CBR (4R Cycle)

Inti dari kecerdasan sistem berada di `recommend.py`.

1. **RETRIEVE (Ambil):** Mencari kasus lama di SQLite yang *query*-nya paling mirip secara semantik (menggunakan FAISS).
2. **REUSE (Gunakan):** Menarik rekomendasi dari kasus lama tersebut (menggabungkan *accepted_ids*). Jika ada *User ID*, sistem membobot ulang (*re-ranking*) prediksi menggunakan skor SVD.
3. **REVISE (Revisi):** Menampilkan kandidat ke pengguna untuk dikoreksi (pengguna memberikan *Like* atau *Dislike*).
4. **RETAIN (Simpan):** Keputusan akhir pengguna disimpan kembali menjadi Kasus Baru di dalam *database* agar sistem semakin pintar ke depannya.

---

## 6. Tahap 5: Integrasi & UI (Web Application)

Sistem rekomendasi tidak akan berguna tanpa antarmuka yang baik.

- **Backend (`app.py`):** Dibangun menggunakan **Flask** (Python) untuk menyediakan RESTful API (`/api/recommend`, `/api/feedback`, dll).
- **Frontend (`index.html`, `style.css`, `app.js`):**
  - Mengadopsi desain bergaya Netflix (Gelap, Modern, Glassmorphism).
  - *Dynamic Search* dengan *Autocomplete* judul film.
  - Memiliki fitur *tracking feedback* (Tombol *Thumbs Up / Down*).
  - Terdapat panel administrasi kecil untuk melihat status dan jumlah kasus di *database* secara *real-time*.

---

## 7. Kesimpulan

**Mengapa arsitektur ini kuat?**
- **Sistem yang Berkembang (Evolving):** Melalui fase *Retain*, sistem otomatis belajar dari preferensi pencarian terbaru tanpa harus di-*training* ulang dari nol secara konstan.
- **Penanganan Cold Start:** Pengguna anonim tetap mendapat rekomendasi berbobot dari *Content-Based* dan *Historical Frequency*.
- **Performa Tinggi:** SQLite dipadukan dengan FAISS memungkinkan pencarian dari puluhan ribu data hanya dalam hitungan milidetik.

---

## Sesi Tanya Jawab
Terima kasih atas perhatiannya.

*Apakah ada bagian teknis spesifik yang ingin kita diskusikan lebih lanjut?*
