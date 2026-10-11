# Pertemuan 6 — Evaluasi Model pada Data Imbalanced

Praktikum evaluasi model komprehensif menggunakan **Credit Card Fraud Dataset**.
Repository ini mencakup materi praktikum poin A–P; tugas praktikum pada poin Q
tidak termasuk.

## Persiapan

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab praktikum_evaluasi_model.ipynb
```

Dataset mentah tidak disimpan di Git karena berukuran sekitar 144 MB. Notebook
akan mengunduh dataset secara otomatis dari penyimpanan publik TensorFlow apabila
`data/creditcard.csv` belum tersedia.

## Isi Praktikum

Notebook utama: [`praktikum_evaluasi_model.ipynb`](praktikum_evaluasi_model.ipynb)

| Bagian | Cakupan |
|---|---|
| A–E | Studi kasus, load data, EDA, dan stratified train-test split |
| F–H | Naive baseline, Logistic Regression, metrik, dan confusion matrix |
| I | SMOTE yang hanya diterapkan pada training set |
| J–K | Stratified 5-fold CV dengan API dan implementasi manual |
| L–M | GridSearchCV, ROC curve, ROC-AUC, dan PR-AUC |
| N | Threshold tuning menggunakan validation set |
| O | Demonstrasi preprocessing leakage vs pipeline yang benar |
| P | Perbandingan final seluruh strategi |

> Poin Q (tugas praktikum customer churn) sengaja tidak dikerjakan sesuai batas
> pengerjaan.

## Menjalankan Notebook

```bash
jupyter lab praktikum_evaluasi_model.ipynb
```

Notebook yang tersimpan di repository sudah dieksekusi. Untuk menghasilkan ulang
seluruh output melalui terminal:

```bash
jupyter nbconvert \
  --to notebook \
  --execute praktikum_evaluasi_model.ipynb \
  --output praktikum_evaluasi_model.ipynb \
  --ExecutePreprocessor.timeout=1200
```

## Metodologi

- Split menggunakan `stratify=y` agar proporsi fraud tetap terjaga.
- `StandardScaler` berada di dalam pipeline agar tidak mempelajari test set.
- SMOTE hanya diterapkan pada training set.
- Threshold optimal dipilih dari validation set, bukan test set.
- Selain Accuracy, Precision, Recall, F1, dan ROC-AUC, evaluasi juga melaporkan
  **PR-AUC**, yang lebih informatif pada kelas positif sangat langka.
- Logistic Regression dievaluasi pada seluruh training set. Random Forest dan
  RBF-SVM pada bagian perbandingan CV memakai sampel stratified agar notebook
  dapat direproduksi dalam waktu wajar pada laptop.

## Hasil Utama

Dataset berisi **284.807 transaksi**, termasuk **492 fraud (0,1727%)**.

| Strategi | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Naive baseline | 0,0000 | 0,0000 | 0,0000 | 0,5000 | 0,0017 |
| Logistic Regression | 0,8267 | 0,6327 | 0,7168 | 0,9605 | 0,7414 |
| Logistic Regression + SMOTE | 0,0578 | **0,9184** | 0,1088 | **0,9708** | 0,7245 |
| Best GridSearch (threshold 0,5) | **0,8289** | 0,6429 | 0,7241 | 0,9480 | **0,7441** |
| Threshold tuning (0,0576) | 0,6783 | 0,7959 | **0,7324** | 0,9480 | **0,7441** |

GridSearchCV memilih `C=1.0`, penalti `l2`, dan solver `liblinear`, dengan mean
CV F1 sebesar `0,7642`. Threshold tuning memberikan kompromi terbaik pada test
set: 78 dari 98 fraud terdeteksi dengan 37 false positive. SMOTE menghasilkan
recall tertinggi (90 dari 98 fraud), tetapi false positive meningkat menjadi
1.467 sehingga precision dan F1 turun tajam.

## Struktur Repository

```text
.
├── praktikum_evaluasi_model.ipynb  # notebook A–P yang sudah dijalankan
├── figures/                        # visualisasi hasil praktikum
├── outputs/                        # hasil evaluasi dalam CSV/JSON
├── requirements.txt                # versi dependency
└── README.md
```
