# 🐟 Klasifikasi Kesegaran Ikan Menggunakan CNN MobileNetV2

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Aplikasi web untuk mendeteksi kesegaran ikan secara otomatis menggunakan deep learning berbasis arsitektur **MobileNetV2** dengan teknik transfer learning dan fine-tuning. Dibangun sebagai proyek skripsi dengan antarmuka berbasis **Streamlit**.

---

## 📋 Daftar Isi

- [Demo](#-demo)
- [Fitur](#-fitur)
- [Arsitektur Model](#-arsitektur-model)
- [Dataset](#-dataset)
- [Struktur Proyek](#-struktur-proyek)
- [Instalasi](#-instalasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Hasil Evaluasi](#-hasil-evaluasi)
- [Deploy ke Streamlit Cloud](#-deploy-ke-streamlit-cloud)
- [Teknologi](#-teknologi)

---

## 🎬 Demo

> Akses aplikasi secara langsung di: **[link-streamlit-kamu.streamlit.app](https://share.streamlit.io)**

![Demo App](output/07_confusion_matrix.png)

---

## ✨ Fitur

- **Klasifikasi Biner** — mendeteksi ikan sebagai **Segar (Fresh)** atau **Tidak Segar (Not Fresh)**
- **Validasi Gambar** — sistem otomatis menolak gambar yang bukan foto ikan (foto manusia, kendaraan, diagram, dll.)
- **Probabilitas Visual** — menampilkan tingkat keyakinan model dalam bentuk progress bar
- **Detail Teknis** — informasi lengkap hasil prediksi (probabilitas per kelas, threshold, arsitektur)
- **Antarmuka Responsif** — UI bersih yang bekerja di browser desktop maupun mobile

---

## 🧠 Arsitektur Model

Model dibangun dengan dua fase training:

```
Input (224×224×3)
        ↓
MobileNetV2 Pretrained (ImageNet) ← Feature Extractor
        ↓
GlobalAveragePooling2D
        ↓
Dense(256, ReLU)
        ↓
BatchNormalization
        ↓
Dropout(0.4)
        ↓
Dense(1, Sigmoid) ← Output: 0 = Fresh, 1 = Not Fresh
```

### Fase Training

| Fase | Keterangan | Epoch | Learning Rate |
|------|-----------|-------|---------------|
| **Fase 1** | Base model di-freeze, hanya head yang dilatih | 15 | 1e-3 |
| **Fase 2** | Fine-tuning layer terakhir MobileNetV2 (dari layer 130) | 25 | 1e-5 |

### Callbacks yang Digunakan
- `EarlyStopping` — menghentikan training saat tidak ada peningkatan
- `ModelCheckpoint` — menyimpan bobot terbaik otomatis
- `ReduceLROnPlateau` — menurunkan learning rate saat plateau

---

## 📁 Dataset

Dataset terdiri dari dua kelas:

| Kelas | Label | Keterangan |
|-------|-------|-----------|
| **Fresh** | `0` | Ikan segar |
| **Not Fresh** | `1` | Ikan tidak segar |

### Pembagian Data

| Split | Persentase | Keterangan |
|-------|-----------|-----------|
| Train | 80% | Melatih model |
| Validation | 10% | Memantau performa saat training |
| Test | 10% | Evaluasi akhir model |

### Augmentasi Data (Train)
- Rotasi ±20°
- Pergeseran horizontal & vertikal 10%
- Shear 10%
- Zoom 20%
- Flip horizontal

---

## 📂 Struktur Proyek

```
klasifikasi-kesegaran-ikan/
│
├── app.py                          # Aplikasi Streamlit
├── requirements.txt                # Dependensi Python
├── klasifikasi_ikan_skripsi.ipynb  # Notebook training
│
├── dataset ikan/                   # Folder dataset (tidak di-upload ke GitHub)
│   ├── fresh/
│   └── not fresh/
│
└── output/                         # Hasil training
    ├── model_final.keras           # Model terbaik (format Keras)
    ├── model_final.h5              # Model terbaik (format H5)
    ├── best_phase1.keras           # Checkpoint fase 1
    ├── best_model_final.keras      # Checkpoint fase 2
    ├── history_p1.npy              # Riwayat training fase 1
    ├── history_p2.npy              # Riwayat training fase 2
    └── laporan_evaluasi.txt        # Laporan hasil evaluasi
```

---

## ⚙️ Instalasi

### Prasyarat
- Python 3.10 atau lebih baru
- pip

### Langkah Instalasi

**1. Clone repositori**
```bash
git clone https://github.com/username/klasifikasi-kesegaran-ikan.git
cd klasifikasi-kesegaran-ikan
```

**2. Buat virtual environment (opsional tapi disarankan)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependensi**
```bash
pip install -r requirements.txt
```

**4. Pastikan model tersedia**

Letakkan file `model_final.keras` (atau `model_final.h5`) di folder yang sama dengan `app.py`:
```
klasifikasi-kesegaran-ikan/
├── app.py
├── requirements.txt
└── model_final.keras   ← di sini
```

---

## 🚀 Cara Penggunaan

### Menjalankan Aplikasi

```bash
streamlit run app.py
```

Buka browser dan akses `http://localhost:8501`

### Cara Memakai Aplikasi

1. Klik tombol **"Browse files"** atau seret foto ikan ke area upload
2. Tunggu proses validasi dan analisis (beberapa detik)
3. Lihat hasil prediksi: **Segar ✅** atau **Tidak Segar ❌**
4. Buka bagian **Detail Teknis** untuk melihat probabilitas lengkap

### Melatih Ulang Model

Buka dan jalankan notebook secara berurutan:
```bash
jupyter notebook klasifikasi_ikan_skripsi.ipynb
```

Sesuaikan path dataset di bagian **Parameter**:
```python
DATASET_PATH = 'dataset ikan'   # sesuaikan dengan lokasi dataset
```

---

## 📊 Hasil Evaluasi

Evaluasi dilakukan pada **data test (10%)** yang tidak pernah dilihat model selama training.

| Metrik | Nilai |
|--------|-------|
| **Accuracy** | - |
| **Precision** | - |
| **Recall** | - |
| **F1-Score** | - |

> Isi tabel di atas dengan hasil aktual dari `output/laporan_evaluasi.txt` setelah training selesai.

---

## ☁️ Deploy ke Streamlit Cloud

**1. Push ke GitHub**
```bash
git init
git add app.py requirements.txt
git commit -m "initial commit"
git remote add origin https://github.com/username/nama-repo.git
git push -u origin main
```

> Jika ukuran model > 100MB, gunakan Git LFS:
> ```bash
> git lfs install
> git lfs track "*.keras" "*.h5"
> git add .gitattributes model_final.keras
> git commit -m "add model with LFS"
> git push
> ```

**2. Deploy di [share.streamlit.io](https://share.streamlit.io)**
- Login dengan akun GitHub
- Klik **New app** → pilih repositori
- Set **Main file path** → `app.py`
- Klik **Deploy**

---

## 🛠️ Teknologi

| Library | Versi | Kegunaan |
|---------|-------|---------|
| TensorFlow / Keras | ≥ 2.13 | Membangun dan melatih model CNN |
| MobileNetV2 | ImageNet pretrained | Base model transfer learning |
| Streamlit | ≥ 1.35 | Antarmuka web aplikasi |
| OpenCV | ≥ 4.8 | Validasi gambar (deteksi wajah) |
| NumPy | ≥ 1.24 | Operasi array dan preprocessing |
| Pillow | ≥ 10.0 | Membaca dan memproses gambar |
| scikit-learn | - | Evaluasi model (confusion matrix, classification report) |
| Matplotlib / Seaborn | - | Visualisasi data dan hasil |
| Pandas | - | Manajemen dataset |

---

## 👩‍💻 Author

**Nama Lengkap**  
Program Studi — Fakultas — Universitas  
📧 email@university.ac.id

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik (skripsi). Silakan gunakan sebagai referensi dengan mencantumkan sumber.
