# Laporan Proyek Machine Learning - Ja'far Shodiq

## Domain Proyek

Kualitas susu merupakan aspek penting dalam industri pangan dan kesehatan. Penentuan kualitas susu secara tradisional mengandalkan uji laboratorium yang mahal, memakan waktu, dan memerlukan tenaga ahli. Dengan berkembangnya teknologi, pendekatan berbasis machine learning menjadi solusi yang efisien dan cepat.

Proyek ini bertujuan membangun sistem prediksi kualitas susu berdasarkan parameter fisikokimia seperti pH, suhu, bau, rasa, lemak, dan turbiditas. Dataset diambil dari Kaggle [1], dan digunakan tiga algoritma klasifikasi untuk membandingkan performa model: Random Forest, Support Vector Machine (SVM), dan K-Nearest Neighbor (KNN).

## Business Understanding
### Problem Statements
- Bagaimana mengotomatisasi prediksi kualitas susu menggunakan data sensor fisikokimia?
- Algoritma machine learning mana yang memberikan performa terbaik dalam klasifikasi kualitas susu?

### Goals
- Membangun model klasifikasi untuk memprediksi kualitas susu berdasarkan fitur fisikokimia.
- Menentukan algoritma terbaik dengan evaluasi berbasis akurasi, precision, recall, dan F1-score.

### Solution statements
- Melakukan eksplorasi data untuk memahami distribusi fitur.
- Melakukan pembagian dataset menjadi data latih dan data uji.
- Membangun tiga model klasifikasi: Random Forest, SVM, dan KNN.
- Evaluasi performa model menggunakan metrik klasifikasi.
- Memilih model terbaik berdasarkan hasil evaluasi.

## Data Understanding
Dataset diperoleh dari platform Kaggle dengan judul "Milk Quality Prediction" [1]. Dataset ini berisi parameter fisikokimia susu yang digunakan untuk menentukan kualitasnya.

### Informasi Dataset:
- Jumlah sampel: 1059 baris
- Jumlah fitur: 7 kolom input + 1 kolom target
- Missing values: Tidak ditemukan nilai yang kosong pada semua fitur
- Outliers: Ditemukan outliers pada fitur:
  - pH: Range normal 6.0-9.0
  - Temperature: Range normal 34-90°C  
  - Colour: Range normal 240-255
- Distribusi target Grade: tidak seimbang (lebih banyak kelas Low)

### Fitur-Fitur:
- *pH:* Derajat keasaman susu.
- *Temprature:* Suhu susu.
- *Taste:* Cita rasa susu (nilai biner 0 atau 1).
- *Odor:* Bau susu (nilai biner 0 atau 1).
- *Fat:* Kandungan lemak (nilai biner 0 atau 1).
- *Turbidity:* Tingkat kekeruhan susu (nilai biner 0 atau 1).
- *Colour:* Skala warna susu.
- *Grade (target):* Kualitas susu (Low, Medium, High).

### Exploratory Data Analysis:
   - Korelasi antara fitur menunjukkan hubungan yang cukup kuat antara Turbidity dan Odor
   - Terdapat beberapa outlier pada fitur pH, Temperature, dan Colour
   - Distribusi kelas target menunjukkan ketidakseimbangan dengan kelas Low yang lebih dominan
   - Histogram fitur numerik menunjukkan distribusi yang bervariasi pada setiap fitur

## Data Preparation
Tahapan preprocessing yang dilakukan:
1. Penanganan Outlier:
   - Menggunakan metode IQR untuk mendeteksi dan menangani outlier pada fitur pH, Temperature, dan Colour
   - Outlier dipertahankan karena merepresentasikan variasi alami dalam kualitas susu

2. Standarisasi Fitur Numerik:
   - Menerapkan StandardScaler pada fitur pH, Temperature, dan Colour
   - Fitur binary (Taste, Odor, Fat, Turbidity) tidak perlu distandarisasi

3. Pembagian Dataset:
   - Training set: 80% (847 sampel)
   - Testing set: 20% (212 sampel)
   - Menggunakan stratified split untuk menjaga proporsi kelas

## Modeling
Tiga algoritma klasifikasi digunakan dengan konfigurasi berikut:

### 1. Random Forest
Model ensemble yang membangun multiple decision trees secara parallel. 
Cara Kerja:
- Random Forest adalah model ensemble yang membangun multiple decision trees dan menggabungkan hasil prediksi mereka
- Setiap pohon dilatih pada subset acak dari data dan fitur
- Hasil final ditentukan berdasarkan voting mayoritas (untuk klasifikasi) dari semua pohon
- Diversity antar pohon membantu mengurangi overfitting dan meningkatkan generalisasi

Parameter yang Digunakan:
- n_estimators: 300 (jumlah pohon dalam forest)
- min_samples_split: 2 (jumlah minimum sampel yang diperlukan untuk membagi node)
- min_samples_leaf: 1 (jumlah minimum sampel yang diperlukan pada node leaf)
- max_depth: 20 (kedalaman maksimum setiap pohon, menentukan kompleksitas model)
- bootstrap: True (pengambilan sampel dengan penggantian saat membangun pohon)

Alasan pemilihan: Kemampuan menangani data non-linear dan robust terhadap overfitting [2].

### 2. Support Vector Machine (SVM)
Model klasifikasi dengan kernel trick untuk data non-linear.
Cara Kerja:
- SVM mencari hyperplane optimal yang memisahkan kelas-kelas pada ruang fitur
- Untuk data non-linear, SVM menggunakan kernel trick untuk memproyeksikan data ke dimensi lebih tinggi
- Kernel RBF menemukan hyperplane pada ruang berdimensi tinggi tanpa benar-benar mengkomputasi transformasi
- Margin maksimum antara hyperplane dan titik terdekat dari setiap kelas membantu generalisasi yang lebih baik

Parameter yang Digunakan:
- kernel: 'rbf' (Radial Basis Function - fungsi kernel untuk data non-linear)
- C: 10 (parameter regularisasi - mengontrol trade-off antara margin yang lebar dan error klasifikasi)
- gamma: 'auto' (koefisien kernel - menentukan seberapa jauh pengaruh satu contoh pelatihan)

Alasan pemilihan: Efektif untuk dataset berdimensi tinggi dengan pemisahan non-linear [3].

### 3. K-Nearest Neighbor (KNN)
Model berbasis instance learning [4].
Cara Kerja:
- KNN adalah algoritma non-parametrik yang mengklasifikasikan data baru berdasarkan kemiripan dengan data training
- Untuk setiap titik data baru, KNN mencari K tetangga terdekat berdasarkan jarak (biasanya Euclidean)
- Kelas yang paling banyak muncul di antara K tetangga menjadi prediksi kelas untuk data baru
- KNN tidak melakukan "training" dalam arti tradisional, melainkan menyimpan semua data dan melakukan komputasi saat prediksi

Parameter yang Digunakan:
- n_neighbors: 3 (jumlah tetangga terdekat yang dipertimbangkan)
- weights: 'uniform' (default - semua tetangga memiliki bobot sama)
- metric: 'euclidean' (default - jarak dihitung menggunakan metrik Euclidean)

Alasan pemilihan: Sederhana namun efektif untuk klasifikasi multi-kelas.

## Evaluation
Untuk mengevaluasi performa model dalam memprediksi kualitas susu, digunakan empat metrik utama:

Metrik Evaluasi:
- **Akurasi**: Mengukur proporsi prediksi yang benar dari total prediksi. Cocok untuk dataset yang seimbang.
   - Formula: (TP + TN) / (TP + TN + FP + FN)
   - Relevansi: Memberikan gambaran umum performa model pada keseluruhan data

- **Presisi**: Mengukur ketepatan model dalam memprediksi kelas positif.
   - Formula: TP / (TP + FP)
   - Relevansi: Penting dalam konteks kualitas susu untuk meminimalkan false positive (misklasifikasi susu berkualitas rendah sebagai berkualitas tinggi)

- **Recall**: Mengukur kemampuan model dalam menangkap semua instance dari kelas positif. 
   - Formula: TP / (TP + FN)
   - Relevansi: Memastikan model tidak melewatkan susu berkualitas rendah yang bisa membahayakan konsumen

- **F1-Score**: Rata-rata harmonis antara presisi dan recall, memberikan keseimbangan antara kedua metrik.
   - Formula: 2 * (Precision * Recall) / (Precision + Recall)
   - Relevansi: Metrik yang baik untuk dataset dengan distribusi kelas yang tidak seimbang

Hasil Evaluasi Model:
| Model                  | Akurasi | Precision | Recall | F1-Score |
| ---------------------- | ------- | --------- | ------ | -------- |
| Random Forest          | 99.53%  | 99.54%    | 99.53% | 99.53%   |
| Support Vector Machine | 96.70%  | 96.85%    | 96.70% | 96.72%   |
| K-Nearest Neighbor     | 98.58%  | 98.61%    | 98.58% | 98.59%   |

Analisis Hasil:
- Random Forest mencapai performa tertinggi di semua metrik, dengan akurasi 99.53%. Model ini berhasil menangkap pola kompleks dalam data susu dan memberikan prediksi yang sangat akurat.

- SVM memberikan hasil yang baik (96.70% akurasi) meskipun tidak sebaik Random Forest. Kernel RBF berhasil menangani non-linearitas dalam dataset.

- KNN menunjukkan performa yang sangat baik (98.58% akurasi) dengan parameter sederhana, menunjukkan bahwa kedekatan antar sampel merupakan indikator yang baik untuk kualitas susu.

Model Random Forest dipilih sebagai model akhir untuk sistem prediksi kualitas susu karena memberikan performa terbaik di semua metrik evaluasi.

### Dampak Terhadap Business Understanding
1. Problem Statements:
   - Otomatisasi prediksi berhasil dicapai dengan akurasi 99.53% menggunakan Random Forest
   - Random Forest terbukti memberikan performa terbaik dibanding SVM dan KNN

2. Goals:
   - Model klasifikasi berhasil dibangun dengan performa tinggi
   - Evaluasi komprehensif menunjukkan Random Forest unggul di semua metrik

3. Solution Statements:
   - Eksplorasi data membantu pemahaman karakteristik dataset
   - Pembagian data 80:20 memberikan hasil evaluasi yang reliable
   - Ketiga model memberikan performa baik (>96% akurasi)
   - Evaluasi metrik menunjukkan konsistensi performa Random Forest

## Daftar Pustaka
[1] Cpluzshrijayan, “Milk Quality Prediction Dataset,” Kaggle. [Online]. Tersedia: https://www.kaggle.com/datasets/cpluzshrijayan/milkquality.
[2] L. Breiman, “Random Forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.
[3] C. Cortes and V. Vapnik, “Support-vector networks,” Machine Learning, vol. 20, no. 3, pp. 273–297, 1995.
[4] T. Cover and P. Hart, “Nearest neighbor pattern classification,” IEEE Transactions on Information Theory, vol. 13, no. 1, pp. 21–27, 1967.

