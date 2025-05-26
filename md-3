# Laporan Proyek Machine Learning - Muhammad Zaki Alfadilah
## Domain Proyek
Tingkat dropout mahasiswa di perguruan tinggi merupakan salah satu tantangan serius dalam sistem pendidikan tinggi di berbagai negara, termasuk Indonesia. Dropout tidak hanya merugikan institusi pendidikan dari sisi efisiensi sumber daya, tetapi juga berdampak negatif terhadap mahasiswa secara psikologis dan finansial. Mahasiswa yang gagal menyelesaikan studi sering kali mengalami kerugian biaya pendidikan yang sudah dikeluarkan, hilangnya waktu, dan berkurangnya peluang kerja di masa depan.

Dengan memanfaatkan dataset historis yang mencakup berbagai atribut seperti status sosial ekonomi, performa akademik, serta keterlibatan mahasiswa, model klasifikasi ini dapat membantu pihak kampus atau akademik dalam mengidentifikasi mahasiswa berisiko tinggi dan merancang strategi pendampingan yang lebih efektif. Sebuah studi yang dilakukan oleh [Kotsiantis et al. (2013)]((https://link.springer.com/chapter/10.1007/978-3-540-45226-3_37#preview)) berhasil menggunakan teknik klasifikasi seperti Decision Tree dan Naive Bayes untuk memprediksi mahasiswa yang cenderung dropout, dengan akurasi yang menjanjikan.

Proyek ini bertujuan untuk membangun sistem klasifikasi prediktif yang dapat membedakan antara mahasiswa yang kemungkinan besar akan graduate dan yang berpotensi mengalami dropout, sehingga dapat berkontribusi dalam menurunkan tingkat kegagalan akademik dan meningkatkan kualitas pendidikan tinggi.

## Business Understanding
### Problem 

- Tingkat dropout mahasiswa di perguruan tinggi masih tinggi dan menjadi tantangan serius yang belum sepenuhnya teratasi, baik secara institusional maupun individu.

- Banyak institusi belum memiliki sistem prediksi yang efektif untuk mengidentifikasi mahasiswa berisiko tinggi mengalami dropout sejak dini.

- Data mahasiswa yang tersedia (akademik, sosial ekonomi, dan lainnya) belum dimanfaatkan secara maksimal untuk membantu proses pengambilan keputusan strategis terkait pencegahan dropout.

### Goals

- Mengurangi angka dropout mahasiswa dengan mengembangkan sistem klasifikasi berbasis machine learning yang mampu mengidentifikasi mahasiswa yang berpotensi tidak menyelesaikan studinya.

- Membangun model prediktif yang akurat dan efisien untuk mendeteksi risiko dropout mahasiswa berdasarkan data historis dan karakteristik individu.

- Mengoptimalkan pemanfaatan data multidimensional mahasiswa sebagai dasar dalam proses analisis dan pengambilan kebijakan intervensi akademik secara proaktif dan berbasis data.

**Rubrik/Kriteria Tambahan (Opsional):**

- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut:
    ### Solution statements
- Proyek ini mengimplementasikan dua model klasifikasi berbasis machine learning, yaitu **Random Forest** dan **Logistic Regression**, untuk memprediksi kemungkinan mahasiswa mengalami dropout.  
Model dilatih menggunakan data akademik, dengan pendekatan **feature engineering** untuk memastikan input yang digunakan relevan dan berkualitas.

- Pada proses modelling digunakan **hyperparameter tuning** dilakukan guna meningkatkan performa model. Pada proses evaluasi dilakukan dengan menggunakan metrik **accuracy**, **precision**, **recall**, dan **F2 score**, agar hasil prediksi tidak hanya akurat, tetapi juga sensitif terhadap kasus dropout.



## Data Understanding

Proyek ini menggunakan dataset publik dari UCI Machine Learning Repository berjudul **“Predict Students Dropout and Academic Success”**, yang tersedia di:  
[https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

Dataset ini dikumpulkan dari institusi pendidikan tinggi dengan total data 3630 data,  bertujuan untuk mendeteksi potensi dropout mahasiswa berdasarkan informasi akademik dan demografis. Target dari dataset ini adalah **status mahasiswa**: `Graduate` dan `Dropout`

### Fitur-Fitur dalam Dataset

| Fitur | Deskripsi | Tipe Data |
|------|-----------|-----------|
| **Curricular units 1st/2nd sem (approved, grade, enrolled, credited, evaluations)** | Statistik akademik berdasarkan semester | Numerik |
| **Tuition fees up to date** | Status pembayaran biaya kuliah (`yes` / `no`) | Kategorikal |
| **Scholarship holder** | Status beasiswa (`yes` / `no`) | Kategorikal |
| **Admission grade** | Nilai masuk (0–200) | Numerik |
| **Displaced** | Apakah mahasiswa berasal dari luar kota (`yes` / `no`) | Kategorikal |
| **Previous qualification (grade)** | Nilai kualifikasi sebelumnya | Numerik |
| **Application order** | Urutan pilihan jurusan saat mendaftar (0 = pilihan pertama, hingga 9) | Numerik |
| **Daytime/evening attendance** | Tipe kuliah (1 = siang, 0 = malam) | Numerik |
| **GDP** | GDP (Pertumbuhan ekonomi tahunan (%) | Numerik |
| **Course** | Jenis program studi yang diambil | Kategorikal |
| **Target** | Kategori hasil akhir mahasiswa (`Graduate`, `Dropout`) | Kategorikal |

### Ringkasan Pemahaman

- **Fitur akademik** seperti jumlah mata kuliah yang disetujui, nilai, dan jumlah evaluasi memberikan gambaran performa akademik mahasiswa.
- **Fitur administratif** seperti status beasiswa dan pembayaran memberikan informasi tentang kondisi finansial mahasiswa.
- **Fitur demografis** seperti status displaced dan attendance mode memberikan wawasan terkait konteks sosial mahasiswa.
- Target klasifikasi akan difokuskan pada dua kelas utama: `Graduate` dan `Dropout`.

### Visualisasi Distribusi Data Numerik
![Distribusi Numerik](https://github.com/zakialfadilah/Predict-Students-Dropout-and-Academic-Success/blob/main/assets/distribusi_numerik_new.png?raw=true)

Dalam dataset ini terdapat sejumlah fitur numerik yang merepresentasikan aspek kuantitatif dari performa akademik dan atribut lain mahasiswa. Contohnya termasuk nilai masuk (Admission grade) yang berada pada rentang 0 hingga 200, jumlah mata kuliah yang disetujui dan dinilai pada semester pertama dan kedua (Curricular units approved, grade, credited, evaluations), serta urutan pilihan jurusan saat mendaftar (Application order). Fitur numerik ini penting karena memberikan gambaran yang terukur mengenai kemampuan akademik dan preferensi mahasiswa.


### Visualisasi Distribusi Data Kategorikal
![Distribusi Kategorik](https://github.com/zakialfadilah/Predict-Students-Dropout-and-Academic-Success/blob/main/assets/distribusi_kategorikal.png?raw=true)

Selain itu, dataset juga mengandung fitur kategorikal yang memberikan informasi kontekstual dan status administratif mahasiswa, seperti status pembayaran biaya kuliah (Tuition fees up to date: yes/no), status penerima beasiswa (Scholarship holder: yes/no), jenis kehadiran kuliah (Daytime/evening attendance: 1/0 (Daytime/night), dan apakah mahasiswa berasal dari luar kota (Displaced: yes/no). Data kategorikal ini membantu memahami kondisi non-akademik yang dapat mempengaruhi kelulusan atau dropout mahasiswa.

### Visualisasi Confusion Matrix

- Data Encoding
Pada tahap ini sebelum melakukan visualisasi confusion matrix, label target dilakukan encoding terlebih dahulu, agar data dapat digunakan sebagai bahan analysis
```
df_encoded = df.copy()

# Encode Target: Graduate = 1, Dropout = 0
df_encoded['Target'] = df_encoded['Target'].replace({'Graduate': 1, 'Dropout': 0})
```

setelah selesai, data divisualisasikan menggunakan dengan heatmap dengan kode berikut:
```
plt.figure(figsize=(16, 10))
correlation = df_encoded.corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Heatmap Korelasi Antar Fitur (Termasuk Target Ter-Encode)")
plt.show()
```


![Confusion Matrix](https://github.com/zakialfadilah/Predict-Students-Dropout-and-Academic-Success/blob/main/assets/confusionmatrix.png?raw=true)

Untuk memahami hubungan antar fitur numerik dalam dataset, dilakukan analisis matriks korelasi. Matriks ini menunjukkan sejauh mana setiap pasangan fitur saling berkorelasi, baik positif maupun negatif, dengan nilai berkisar antara -1 hingga 1. Korelasi positif mendekati 1 menunjukkan bahwa kedua fitur cenderung meningkat bersama-sama, sementara korelasi negatif mendekati -1 menunjukkan hubungan berlawanan.




## Data Preparation

- ### Data Encoding
```
df_encoded = df.copy()

# Encode Target: Graduate = 1, Dropout = 0
df_encoded['Target'] = df_encoded['Target'].replace({'Graduate': 1, 'Dropout': 0})
```

Agar data bisa diproses oleh model, data kategorikal perlu diencoding menjadi data numerik, pada label target diubah kolom Graduate menjadi 1, dan Dropout menjadi 0. Langkah ini sudah dilakukan pada tahap visualisasi confusion matrix sebelumnya, yang berarti variable df_encoded (berisikan dataset, dengan target label yang sudah di encode) sudah dapat digunakan.


- ### Cek data null
```
df.isnull().sum() 
```
Fungsi dilakukan cek data null adalah untuk memastikan tidak ada nilai kosong (missing values) dalam dataset yang bisa mengganggu proses analisis atau pelatihan model machine learning. Nilai null bisa menyebabkan error . Berdasarkan pengecekan, dataset tidak memiliki data null
```
Curricular units 2nd sem (approved)	0
Curricular units 2nd sem (grade)	0
Curricular units 1st sem (approved)	0
Curricular units 1st sem (grade)	0
Tuition fees up to date	0
Scholarship holder	0
Curricular units 2nd sem (enrolled)	0
Curricular units 1st sem (enrolled)	0
Admission grade	0
Displaced	0
Previous qualification (grade)	0
Curricular units 2nd sem (evaluations)	0
Application order	0
Daytime/evening attendance\t	0
Curricular units 2nd sem (credited)	0
Curricular units 1st sem (credited)	0
Curricular units 1st sem (evaluations)	0
GDP	0
Course	0
Target	0
```
- ### Cek data duplikat
```
df.duplicated().sum()
```
Output:
```
Total Data Duplikat:
np.int64(0)
```
Tahapan ini dilakukan untuk melakukan cek apakah ada data yang terduplikasi dalam dataset atau tidak. Setelah di cek, dataset tidak memiliki data terduplikasi
- ### Outlier Checking
```
# Pilih fitur numerik
numerical_features = df_encoded.select_dtypes(include=[np.number]).columns.tolist()

# Visualisasi boxplot per fitur numerik
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df_encoded[feature])
    plt.title(f'Boxplot untuk fitur: {feature}')
    plt.xlabel(feature)
    plt.show()
```

Pada tahap ini, ditampilkan seluruh fitur numerik dalam bentuk boxplot, untuk mengetahui distribusi dari dataset. Dan didapatkan bahwa di beberapa fitur memiliki outlier yang cukup signifikan

![outlierchecking](https://github.com/zakialfadilah/Predict-Students-Dropout-and-Academic-Success/blob/main/assets/outlier%20checking.png?raw=true)




- ### Outlier Handling
```
# Hitung IQR untuk fitur Admission grade
Q1 = df_encoded['Admission grade'].quantile(0.25)
Q3 = df_encoded['Admission grade'].quantile(0.75)
IQR = Q3 - Q1

# Batas bawah dan atas
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter: hanya data yang bukan outlier
df_encoded = df_encoded[(df_encoded['Admission grade'] >= lower_bound) & 
                        (df_encoded['Admission grade'] <= upper_bound)]

# Cek hasil
print(f"Jumlah data setelah menghapus outlier: {df_encoded.shape[0]}")
```
Pada dataset, terdeteksi memiliki outlier pada feature Admission grade, beberapa feature lain juga memiliki outlier, namun setelah analisa lebih dalam, salah satu feature yang cukup berpengaruh adalah feature Admission grade. Fungsi outlier handling ini adalah untuk menghindari turunnya akurasi dikarenakan tidak seimbangnya data yang ada.

Berikut merupakan visualisasi distribusi data pada Admission grade
- Sebelum
![before](https://github.com/zakialfadilah/Predict-Students-Dropout-and-Academic-Success/blob/main/assets/before_outlier.png?raw=true)
- Sesudah
![after](https://github.com/zakialfadilah/Predict-Students-Dropout-and-Academic-Success/blob/main/assets/after_outlier.png?raw=true)

- ### Data Splitting
```
from sklearn.model_selection import train_test_split

X = df_encoded.drop(columns=['Target']) 
y = df_encoded['Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Cek hasil
print("Ukuran data latih:", X_train.shape)
print("Ukuran data uji:", X_test.shape)
```

Sebelum membangun model klasifikasi, dataset dibagi menjadi data latih (training set) dan data uji (testing set) untuk memastikan bahwa model dapat dievaluasi secara adil terhadap data yang belum pernah dilihat sebelumnya. Proses pembagian ini dilakukan menggunakan fungsi train_test_split dari pustaka scikit-learn, dengan proporsi 80% data digunakan untuk pelatihan (X_train, y_train) dan 20% sisanya untuk pengujian (X_test, y_test). Selain itu, digunakan parameter stratify=y untuk memastikan distribusi label target pada data latih dan uji tetap proporsional, terutama karena target klasifikasi memiliki kemungkinan distribusi yang tidak seimbang. Nilai random_state=42 digunakan agar proses pembagian data bersifat reproducible, sehingga hasil yang diperoleh dapat direplikasi di masa mendatang.

## Modeling

Dalam projek ini, saya menggunakan lima algoritma klasifikasi yang umum digunakan untuk membandingkan performa mereka terhadap dataset yang telah disiapkan pada tahap awal, yang nantinya akan dipilih model terbaik yang akan digunakan. Kelima model ini dipilih karena keunggulannya masing-masing dalam menangani berbagai jenis data dan kompleksitas.

---

## 1. Logistic Regression
**Logistic Regression** digunakan untuk memodelkan hubungan antara fitur input dan probabilitas dari kelas target. Meskipun disebut "regression", model ini adalah algoritma klasifikasi biner yang cepat dan efektif pada dataset linier atau hampir linier.

- Cocok untuk data linier.
- Cepat dilatih dan diinterpretasi.
- Tidak kompleks secara komputasi.

---

## 2. Decision Tree
**Decision Tree Classifier** membuat model dalam bentuk struktur pohon keputusan, di mana setiap node mewakili fitur, dan cabangnya adalah kondisi logis.

- Mudah dipahami dan divisualisasikan.
- Dapat menangani data kategorikal dan numerik.
- Cenderung overfitting jika tidak di-prune.

---

## 3. Random Forest
**Random Forest** adalah ensemble learning method yang membangun banyak pohon keputusan dan menggabungkan hasilnya untuk meningkatkan akurasi dan mengurangi overfitting.

- Lebih stabil dan akurat dibanding Decision Tree tunggal.
- Bagus untuk menangani data dengan banyak fitur.
- Kurang dapat diinterpretasi dibandingkan model individual.

---

## 4. K-Nearest Neighbors (KNN)
**K-Nearest Neighbors** mengklasifikasikan sebuah titik berdasarkan mayoritas label dari tetangga terdekatnya.

- Tidak memerlukan pelatihan (lazy learner).
- Sensitif terhadap skala data dan jumlah fitur.
- Kinerja bisa menurun pada dataset besar.

---

## 5. Support Vector Machine (SVM)
**Support Vector Machine** mencari hyperplane optimal yang memisahkan kelas dalam ruang fitur dengan margin maksimal.

- Sangat efektif pada dimensi tinggi.
- Memiliki kernel trick untuk data non-linear.
- Relatif mahal secara komputasi.

---

Model-model ini diuji untuk menemukan mana yang paling cocok berdasarkan metrik seperti **Accuracy**, **Precision**, **Recall**, dan **F1 Score**.

---

### Proses Modelling

- ### Import Library
```python
# Instalasi library
!pip install scikit-learn

# Import model dan metrik evaluasi
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
```

- ### Inisialisasi Model
Pada tahap ini, beberapa algoritma klasifikasi diinisialisasi ke dalam sebuah dictionary bernama `models`. Tujuannya adalah untuk mempermudah proses pelatihan dan evaluasi secara iteratif. Setiap model dipilih berdasarkan karakteristiknya yang beragam, mulai dari model yang sederhana dan cepat seperti **Logistic Regression**, hingga model kompleks seperti **Random Forest** dan **SVM** yang mampu menangani data dengan distribusi yang lebih rumit. Parameter awal sebagian besar dibiarkan default, kecuali untuk **Logistic Regression** yang ditambahkan `max_iter=1000` agar proses konvergensi lebih stabil.
```
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "SVM": SVC()
}
```

- ### Evaluasi Model

Setelah seluruh model dilatih menggunakan data pelatihan dan diuji pada data pengujian, diperoleh hasil evaluasi sebagai berikut:

| Model                 | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 0.9087   | 0.8966    | 0.9607 | 0.9275   |
| Random Forest        | 0.9059   | 0.8961    | 0.9561 | 0.9251   |
| Decision Tree        | 0.8497   | 0.8918    | 0.8568 | 0.8740   |
| K-Nearest Neighbors  | 0.8301   | 0.8120    | 0.9376 | 0.8703   |
| Support Vector Machine (SVM) | 0.6081 | 0.6081 | 1.0000 | 0.7563   |

Dari hasil tersebut, dapat disimpulkan bahwa:

- **Logistic Regression** memberikan performa terbaik secara keseluruhan, dengan nilai **F1 Score tertinggi (0.9275)** dan keseimbangan yang sangat baik antara precision dan recall.
- **Random Forest** juga menunjukkan performa yang sangat kompetitif, hanya sedikit di bawah Logistic Regression, menandakan kemampuannya dalam menangkap kompleksitas data.
- **Decision Tree** dan **K-Nearest Neighbors** masih mampu memberikan hasil yang cukup baik, namun menunjukkan penurunan performa dibandingkan dua model sebelumnya.
- **SVM**, meskipun memiliki **recall sempurna (1.0)**, performanya secara keseluruhan tidak optimal, terlihat dari nilai accuracy dan precision yang rendah, yang mengindikasikan banyak prediksi positif palsu (false positives).

Evaluasi ini memberikan gambaran yang jelas mengenai kekuatan dan kelemahan masing-masing model dalam konteks dataset yang digunakan. Dan berdasarkan hasil evaluasi, Logistic Regression menunjukkan akurasi yang paling tinggi, dan dapat ditingkatkan kembali untuk mendapatkan akurasi yang terbaik.

- ### Model Optimation
```
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Parameter grid untuk Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],  
    'penalty': ['l1', 'l2'],             
    'solver': ['liblinear', 'saga'],     
    'max_iter': [100, 200, 300]
}

grid_search = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='f1'
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

# Evaluasi model terbaik
best_logreg = grid_search.best_estimator_
y_pred_best = best_logreg.predict(X_test)

print("Classification Report (Logistic Regression - Best):")
print(classification_report(y_test, y_pred_best))

```
Pada bagian ini bertujuan mencari konfigurasi hyperparameter Logistic Regression terbaik agar model menghasilkan prediksi yang optimal (berdasarkan F1-score) pada data pelatihan, lalu menguji performa model terbaik itu pada data testing untuk memastikan kualitas prediksi.

## Evaluation
Model yang digunakan dalam proyek ini adalah Logistic Regression, dioptimalkan menggunakan GridSearchCV. Untuk mengevaluasi performa model, digunakan metrik-metrik evaluasi yang umum dalam kasus klasifikasi, yaitu:

- Accuracy
- Precision
- Recall
- F1-Score

`Accuracy`
Mengukur seberapa sering model membuat prediksi yang benar dari seluruh data.
Formula:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
`Precision`
Mengukur seberapa banyak prediksi positif yang benar-benar positif.
Formula:
```
Precision = TP / (TP + FP)
```
`Recall`
Mengukur seberapa banyak data positif yang berhasil dikenali model.
Formula:
```
Recall = TP / (TP + FN)
```
`F1-Score`
Rata-rata harmonis dari precision dan recall. Cocok digunakan saat penting untuk mempertimbangkan keduanya secara seimbang.
Formula:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

`Confusion Matrix`
![confusion](https://github.com/zakialfadilah/Predict-Students-Dropout-and-Academic-Success/blob/main/assets/confusionmatrix_evaluasi.png?raw=true)

True Positive (TP) = 417
→ Kasus positif yang berhasil diprediksi dengan benar sebagai positif.

True Negative (TN) = 232
→ Kasus negatif yang berhasil diprediksi dengan benar sebagai negatif.

False Positive (FP) = 47
→ Kasus negatif yang keliru diprediksi sebagai positif (Type I Error).

False Negative (FN) = 16
→ Kasus positif yang keliru diprediksi sebagai negatif (Type II Error).


### Hasil Evaluasi
```
Label	        Precision	Recall	F1-score	Support
0	                0.94	0.83	0.88	    279
1	                0.90	0.96	0.93	    433
Accuracy	                -	    0.91        712
Macro Avg	        0.92	0.90	0.91	    712
Weighted Avg	    0.91	0.91    0.91	    712
```

### Interpretasi
Akurasi model mencapai 91%, menunjukkan bahwa sebagian besar data dapat diklasifikasikan dengan benar.

Model memiliki F1-score tinggi (0.93) pada kelas 1, artinya model sangat baik dalam mengenali kelas ini.

Recall kelas 0 sedikit lebih rendah (0.83) dibanding kelas 1 (0.96), menandakan bahwa model sedikit lebih sulit dalam mengenali semua contoh kelas 0.

Nilai macro average dan weighted average menunjukkan bahwa performa model cukup seimbang di kedua kelas.

---
## Conclusion
Proyek ini berhasil membangun sistem klasifikasi berbasis machine learning untuk memprediksi kemungkinan mahasiswa mengalami dropout menggunakan dataset dari UCI Machine Learning Repository. Dengan mengimplementasikan dua model utama, yaitu Random Forest dan Logistic Regression, serta menerapkan proses feature engineering dan hyperparameter tuning, model mampu memberikan performa prediksi yang baik berdasarkan metrik evaluasi seperti accuracy, precision, recall, dan F2 score.

Melalui pemanfaatan data akademik, administratif, dan demografis, sistem ini diharapkan dapat menjadi alat bantu yang efektif bagi institusi pendidikan dalam mengidentifikasi mahasiswa berisiko tinggi sejak dini dan menyusun strategi intervensi yang lebih tepat sasaran. Hasil proyek ini menunjukkan bahwa pendekatan data-driven memiliki potensi besar dalam menurunkan angka dropout dan meningkatkan kualitas pendidikan tinggi secara keseluruhan.








