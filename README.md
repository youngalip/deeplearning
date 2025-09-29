Team : TRK
1. Alief Fathur Rahman (122140027)
2. Desty Ananta Purba (122140076)
3. JP Rafi Radiktya Arkan (122140169)

Eksperimen Arsitektur ResNet untuk Klasifikasi 5 Makanan Indonesia
Overview
Eksperimen ini membandingkan performa antara arsitektur Plain-34 (tanpa residual connection) dengan ResNet-34 (dengan residual connection) pada dataset 5 makanan Indonesia yang terdiri dari: bakso, gado-gado, nasi goreng, rendang, dan soto ayam.

Konfigurasi Eksperimen
Dataset

Total kelas: 5 (bakso, gado_gado, nasi_goreng, rendang, soto_ayam)
Training samples: 240 (60 per kelas)
Validation samples: 60 (12 per kelas)
Distribusi kelas: Seimbang untuk setiap kelas

Hyperparameter
Kedua model dilatih dengan konfigurasi yang identik untuk memastikan perbandingan yang fair:
ParameterNilaiBatch Size16Epochs10OptimizerAdamLearning Rate0.001DeviceCPUTotal Parameters21,287,237
Catatan: Kedua arsitektur memiliki jumlah parameter yang sama persis, perbedaan utama hanya pada keberadaan skip connection.

Hasil Perbandingan
Tabel Metrik Performa (Epoch Terakhir)
ModelTraining AccuracyValidation AccuracyTraining LossValidation LossPlain-3426.67%36.67%1.53321.9201ResNet-3470.42%65.00%0.85161.2138Peningkatan+43.75%+28.33%-44.5%-36.8%
Grafik Training Curves
Plain-34 Training Progress
Epoch | Train Acc | Val Acc  | Train Loss | Val Loss
------|-----------|----------|------------|----------
  1   |  20.83%   | 20.00%   |   1.7006   | 338.7608
  2   |  18.33%   | 20.00%   |   1.6272   |   1.6141
  3   |  22.08%   | 13.33%   |   1.6285   |   9.8880
  4   |  20.83%   | 15.00%   |   1.6083   |   1.9553
  5   |  18.33%   | 11.67%   |   1.6233   |   3.0204
  6   |  28.33%   | 35.00%   |   1.5648   |   3.3956
  7   |  26.67%   | 36.67%   |   1.5332   |   1.9201
ResNet-34 Training Progress
Epoch | Train Acc | Val Acc  | Train Loss | Val Loss
------|-----------|----------|------------|----------
  1   |  29.17%   | 28.33%   |   1.6282   | 121.9410
  2   |  45.83%   | 40.00%   |   1.2050   |   4.8620
  3   |  57.50%   | 33.33%   |   1.0946   |  12.6421
  4   |  57.08%   | 38.33%   |   1.2300   |   1.7589
  5   |  55.00%   | 46.67%   |   1.1034   |   2.3769
  6   |  70.42%   | 65.00%   |   0.8516   |   1.2138

Analisis Performa
Dampak Residual Connection
Dari hasil eksperimen ini, terlihat jelas bahwa residual connection memberikan dampak yang sangat signifikan terhadap performa model. ResNet-34 mengungguli Plain-34 dalam semua metrik yang diukur, dengan peningkatan akurasi validasi mencapai 28.33 poin persentase dan penurunan loss validasi sebesar 36.8%.
Yang paling menarik adalah Plain-34 mengalami kesulitan belajar yang serius. Model ini terjebak dengan akurasi training yang stagnan di sekitar 20-28%, bahkan setelah 7 epoch. Fluktuasi yang tinggi pada validation loss (dari 338.76 hingga 1.61) menunjukkan ketidakstabilan dalam proses pembelajaran. Ini adalah manifestasi dari degradation problem yang menjadi motivasi utama diciptakannya arsitektur ResNet - semakin dalam network tanpa skip connection, semakin sulit model untuk dioptimasi, bahkan dengan dataset yang relatif kecil seperti ini.
Sebaliknya, ResNet-34 menunjukkan progress pembelajaran yang jauh lebih baik dan konsisten. Model ini mampu mencapai akurasi training 70.42% dan akurasi validasi 65%, menunjukkan bahwa residual connection memang efektif memfasilitasi gradient flow yang lebih baik. Meskipun sempat mengalami overfitting ringan di epoch 3 (training 57.5%, validation 33.33%), model tetap bisa recover dan meningkat hingga epoch 6. Hal ini membuktikan bahwa skip connection tidak hanya membantu optimasi, tetapi juga membuat model lebih robust terhadap fluktuasi dalam proses training.
Kesimpulan Utama
Eksperimen ini secara empiris memvalidasi konsep fundamental ResNet: identity mapping melalui skip connection memungkinkan network yang dalam untuk dilatih secara efektif. Perbedaan performa yang dramatis antara Plain-34 dan ResNet-34 (dengan arsitektur yang identik kecuali skip connection) menunjukkan bahwa masalah degradasi bukan disebabkan oleh overfitting atau keterbatasan dataset, melainkan karena kesulitan optimasi pada deep plain network. Residual connection menyediakan "jalan pintas" bagi gradient untuk mengalir ke layer-layer awal, sehingga seluruh network dapat belajar representasi yang lebih baik.
