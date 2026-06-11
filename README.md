# CBR Film Recommendation System

Sistem Rekomendasi Film berbasis **Case-Based Reasoning (CBR)** dengan pendekatan **Hybrid Filtering** menggunakan dataset TMDB 5000 dan MovieLens.

Berbeda dengan sistem rekomendasi tradisional yang berfokus pada kemiripan antar-film secara langsung, sistem ini menggunakan **interaksi/kasus pengguna** sebagai entitas utama pencarian (Case Base).

---

## 🚀 Paradigma & Siklus CBR (Dari Indexing hingga Retain)

Alur kerja sistem rekomendasi diimplementasikan secara utuh dengan memadukan pencarian vektor cepat (**FAISS**), database relasional (**SQLite**), dan siklus **4R CBR (Case-Based Reasoning)**. Berikut adalah alur lengkap proses dari pembentukan indeks hingga penyimpanan kasus:

### 0. Tahap Indexing (Offline Setup & Dynamic Update)
Sebelum siklus CBR berjalan, sistem melakukan persiapan representasi data:
*   **Movie Indexing**: Menghitung representasi TF-IDF gabungan genre dan sinopsis untuk semua film, kemudian menyimpannya ke `movie_index.faiss`. Ini digunakan untuk menghitung kemiripan antar-film secara cepat.
*   **Case Indexing**: Membangun indeks kasus awal (`case_index.faiss`) menggunakan teks representasi kasus:
    $$\text{case\_text} = \text{query\_text} + (\text{genres} \times 3) + \text{title\_film\_ref}$$
*   **Dynamic Update**: Setiap kali ada kasus baru yang disimpan pada tahap **Retain**, vektor kasus baru langsung ditambahkan ke dalam indeks FAISS aktif (`case_index.faiss`) secara *real-time* tanpa perlu melatih ulang dari awal.

---

### 1. Retrieve (Pencarian Kasus Serupa)
Ketika pengguna memasukkan pencarian berupa teks kueri dan/atau memilih film referensi:
1.  **Vektorisasi**: Input pengguna diubah menjadi vektor menggunakan model TF-IDF yang telah dilatih.
2.  **Pencarian FAISS**: Indeks `case_index.faiss` digunakan untuk mencari $K$ kasus terdekat secara instan menggunakan pencarian *Cosine Similarity*.
3.  **Perhitungan Kemiripan Kasus (Case Similarity)**: Nilai kemiripan kasus dihitung secara detail dengan formula pembobotan gabungan:
    $$\text{Similarity} = \frac{0.50 \times \text{Sim}_{\text{query}} + 0.30 \times \text{Sim}_{\text{movie}} + 0.20 \times \text{Sim}_{\text{genre}}}{\text{Total Bobot Aktif}}$$
    *   $\text{Sim}_{\text{query}}$: Kemiripan kosinus antara teks kueri baru vs kueri kasus lama.
    *   $\text{Sim}_{\text{movie}}$: Kemiripan kosinus film referensi di `movie_index.faiss`.
    *   $\text{Sim}_{\text{genre}}$: Jaccard overlap dari genre film referensi.
4.  **Threshold & Fallback**: Hanya kasus dengan kemiripan $\ge 0.30$ yang dipertahankan. Jika tidak ada kasus yang memenuhi batas minimal (atau jumlah basis data kurang dari 10 kasus), sistem akan melakukan **fallback** ke pencarian langsung menggunakan Content-Based Filtering (CBF) pada `movie_index.faiss`.

---

### 2. Reuse (Agregasi & Perangkingan Ulang)
Kasus-kasus mirip yang lolos seleksi kemudian digunakan untuk merekomendasikan solusi:
1.  **Agregasi Skor**: Film-film yang disukai (`accepted_ids`) dari kasus-kasus mirip tersebut diekstrak. Skornya diakumulasikan berdasarkan kemiripan kasusnya:
    $$\text{Score}_{\text{kandidat}}(f) = \sum_{c \in \text{Cases}} \text{Sim}(c) \quad \text{untuk setiap } f \in \text{accepted\_ids}(c)$$
2.  **Penalti Kasus Ditolak**: Jika film kandidat tersebut juga muncul di daftar ditolak (`rejected_ids`) kasus mirip lainnya, skornya dikurangi sebagai penalti agar kesalahan rekomendasi lama tidak diulangi.
3.  **Normalisasi & Hybrid Re-ranking**: Skor kandidat dinormalisasi ke rentang $[0, 1]$. Jika pengguna dikenali (bukan *cold-start*), skor ini digabungkan dengan skor dari model **SVD Collaborative Filtering**:
    $$\text{Skor Akhir} = 0.70 \times \text{Skor Kasus} + 0.30 \times \text{Skor SVD}$$
    Untuk pengguna baru (*cold-start*), bobot kasus adalah $1.00$ sepenuhnya.

---

### 3. Revise (Umpan Balik Pengguna)
Rekomendasi ditampilkan di Web UI. Pengguna merevisi solusi rekomendasi secara langsung dengan memberikan umpan balik:
*   👍 **Like** (atau rating tinggi 3-5 bintang) $\rightarrow$ Film dimasukkan ke daftar **`accepted_ids`**.
*   👎 **Dislike** (atau rating rendah 1-2 bintang) $\rightarrow$ Film dimasukkan ke daftar **`rejected_ids`**.

---

### 4. Retain (Penyimpanan & Pembelajaran)
Setelah sesi interaksi pengguna selesai:
1.  **Simpan Kasus**: Kasus interaksi baru beserta preferensi (kueri, film referensi, rekomendasi, accepted, rejected) disimpan ke dalam database SQLite `retained_cases`.
2.  **Perbarui Indeks**: Vektor kasus baru tersebut dihitung dan langsung disisipkan ke indeks FAISS aktif (`case_index.faiss`), serta daftar ID kasus (`case_ids.pkl`) diperbarui.
3.  **Siap Dipakai Kembali**: Pada pencarian berikutnya, kasus yang baru saja disimpan ini sudah siap untuk di-*retrieve* dan menjadi referensi baru bagi pengguna lain.

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
