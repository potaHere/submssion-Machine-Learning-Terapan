# Laporan Proyek Machine Learning - Netflix Content Recommendation System

## Domain Proyek

Industri streaming telah mengalami pertumbuhan yang signifikan dalam beberapa tahun terakhir, dengan Netflix sebagai salah satu platform terkemuka. Dengan katalog konten yang sangat besar, tantangan utama yang dihadapi adalah membantu pengguna menemukan konten yang sesuai dengan preferensi mereka. Sistem rekomendasi menjadi komponen kritis dalam meningkatkan pengalaman pengguna dan mempertahankan engagement pengguna dengan platform.

Latar belakang ini menjadi penting karena:
1. Volume konten yang besar (8809 judul dalam dataset)
2. Kebutuhan personalisasi pengalaman pengguna
3. Peningkatan retensi pengguna melalui rekomendasi yang relevan

## Business Understanding

### Problem Statements
Berdasarkan latar belakang di atas, berikut adalah rincian masalah yang perlu diselesaikan:

1. Bagaimana cara membantu pengguna menemukan konten yang relevan dari ribuan judul yang tersedia di Netflix?
2. Bagaimana cara memberikan rekomendasi yang personal berdasarkan preferensi genre dan karakteristik konten yang disukai pengguna?
3. Bagaimana mengukur kualitas dan relevansi rekomendasi yang diberikan?

### Goals
Tujuan dari proyek ini adalah:

1. Mengembangkan sistem rekomendasi yang dapat memberikan saran konten yang relevan kepada pengguna
2. Memanfaatkan informasi genre dan deskripsi konten untuk memberikan rekomendasi yang sesuai
3. Mengevaluasi efektivitas sistem rekomendasi menggunakan metrik yang sesuai

### Solution Statements
Solusi yang ditawarkan untuk menyelesaikan masalah ini adalah:

1. Content-based Filtering
   - Menggunakan karakteristik konten (judul, deskripsi, genre) untuk memberikan rekomendasi
   - Menggunakan TF-IDF dan Cosine Similarity untuk menghitung kemiripan antar konten
   - Kelebihan: Dapat memberikan rekomendasi tanpa data rating pengguna
   - Kekurangan: Tidak mempertimbangkan preferensi komunitas pengguna

## Data Understanding
Dataset yang digunakan adalah Netflix Titles dataset yang berisi informasi tentang film dan acara TV yang tersedia di Netflix.

### Variabel-variabel pada Netflix Titles dataset adalah sebagai berikut:
1. show_id: id unik untuk setiap judul
2. type: Kategori judul ('Movie' atau 'TV Show')
3. title: Nama movie atau TV show
4. director: Nama sutradara
5. cast: Daftar aktor/aktris utama
6. country: Negara produksi
7. date_added: Tanggal ditambahkan ke Netflix
8. release_year: Tahun rilis
9. rating: Peringkat usia
10. duration: Durasi (menit untuk film, season untuk TV show)
11. listed_in: Genre
12. description: Ringkasan singkat tentang konten

### Exploratory Data Analysis:

Beberapa insight penting dari analisis data:

1. **Distribusi Tipe Konten**
   - Mayoritas konten adalah film dibandingkan TV show
   - Menunjukkan keragaman format konten

2. **Distribusi Negara**
   - Amerika Serikat mendominasi produksi konten
   - Menunjukkan keberagaman sumber konten internasional

3. **Distribusi Genre**
   - Genre "International" adalah yang terbanyak
   - Diikuti oleh Drama dan Komedi
   - Menunjukkan keragaman preferensi konten

## Data Preparation

Tahapan data preparation yang dilakukan:

1. **Feature Engineering**
   - Menggabungkan fitur title, description, dan listed_in
   - Tujuan: Menciptakan representasi kaya akan informasi untuk setiap konten
   - Mempertimbangkan berbagai aspek konten dalam perhitungan kemiripan

2. **Text Vectorization dengan TF-IDF**
   - Mengubah teks menjadi representasi numerik
   - Menggunakan TfidfVectorizer dengan stop_words='english'
   - Mempertimbangkan frekuensi kata dan kepentingannya dalam dokumen

## Modeling

Model yang dikembangkan menggunakan pendekatan Content-based Filtering dengan tahapan:

1. **Perhitungan Similarity Matrix**
   - Menggunakan Cosine Similarity
   - Menghitung kemiripan antar konten berdasarkan vektor TF-IDF
   - Range nilai: 0 (tidak mirip) hingga 1 (sangat mirip)

2. **Sistem Rekomendasi**
   - Input: Judul konten
   - Proses: Mencari dan mengurutkan konten berdasarkan similarity score
   - Output: 10 rekomendasi teratas

3. **Kelebihan dan Kekurangan**
   - Kelebihan:
     * Rekomendasi spesifik berdasarkan konten
     * Tidak memerlukan data rating
     * Mudah menambahkan konten baru
   - Kekurangan:
     * Tidak menangkap preferensi pengguna
     * Terbatas pada karakteristik eksplisit
     * Dapat melewatkan rekomendasi tidak obvious

## Evaluation

Evaluasi model menggunakan dua metrik:

1. **Genre-based Similarity**
   - Mengukur kemiripan genre antara input dan rekomendasi
   - Menggunakan Jaccard Similarity: |A∩B| / |A∪B|
   - Hasil: Rata-rata similarity score untuk berbagai judul berkisar antara 0.3-0.7

2. **Precision@K**
   - K=10 (jumlah rekomendasi yang diberikan)
   - Mengukur proporsi rekomendasi relevan
   - Hasil: Precision score berkisar 0.6-0.9 untuk berbagai judul
   - Menunjukkan bahwa 60-90% rekomendasi memiliki minimal satu genre yang sama

Hasil evaluasi menunjukkan bahwa sistem rekomendasi berhasil memberikan saran yang relevan berdasarkan karakteristik konten, dengan tingkat presisi yang baik dalam menemukan konten dengan genre serupa.

## Conclusion

Sistem rekomendasi yang dikembangkan berhasil mencapai tujuan dalam memberikan rekomendasi yang relevan berdasarkan karakteristik konten. Metrik evaluasi menunjukkan performa yang baik dalam menemukan konten serupa, terutama dari segi genre. Untuk pengembangan ke depan, sistem dapat ditingkatkan dengan:

1. Implementasi collaborative filtering untuk menangkap preferensi pengguna
2. Penambahan fitur seperti popularitas dan rating dalam perhitungan rekomendasi
3. Penggunaan teknik deep learning untuk pemahaman konten yang lebih baik