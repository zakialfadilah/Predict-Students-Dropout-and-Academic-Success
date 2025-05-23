# -*- coding: utf-8 -*-
"""Student_classification.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1f8yo8XwdpbBvrgkN4EOmu6tdy81INwkg

## Import Library
"""

# Mengimport modul
!pip install scikit-learn
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""Pada tahap awal, diimport seluruh library yang dibutuhkan.

## Load Data
"""

# Load data
df = pd.read_csv('student.csv')
df.head()

"""Pada tahap ini, dataset bernama student.csv diload menggunakan pandas, dan disimpan ke variable df

## Data Understanding
"""

# Menampilkan 5 data teratas
print("Data sample:")
df

# Salin dataframe
df_decode = df.copy()

# Decode fitur kategorikal
df_decode['Daytime/evening attendance\t'] = df_decode['Daytime/evening attendance\t'].replace({1: 'Daytime', 0: 'Evening'})
df_decode['Displaced'] = df_decode['Displaced'].replace({1: 'Yes', 0: 'No'})
df_decode['Scholarship holder'] = df_decode['Scholarship holder'].replace({1: 'Yes', 0: 'No'})
df_decode['Tuition fees up to date'] = df_decode['Tuition fees up to date'].replace({1: 'Yes', 0: 'No'})

# Tampilkan 5 baris awal untuk memastikan
df_decode.head()

"""Pada tahap ini dilakukan decode, untuk menampilkan arti dari setiap data yang tersedia, dikarenakan dataset sudah tersedia dalam bentuk yang sudah diencode, dilakukan decode untuk memahami data secara menyeluruh."""

# Menampilkan informasi umum dataset
print("\nInformasi dataset:")
df.info()

"""Dataset ini memiliki 3.630 baris dan 20 kolom, yang berarti terdiri dari 3.630 entri atau sampel data dan 20 fitur (variabel) yang masing-masing merepresentasikan atribut dari setiap entri. Setiap kolom dalam dataset ini tidak memiliki nilai kosong (null), sehingga data bersih dan siap untuk tahap pemrosesan berikutnya. Tipe data dalam fitur bervariasi, mulai dari int64 untuk data numerik diskret, float64 untuk data numerik kontinu, hingga object untuk kategori target. Fitur-fitur yang tersedia mencakup nilai akademik mahasiswa, status beasiswa, kehadiran, hingga variabel target yang menunjukkan output yang ingin diprediksi oleh model machine learning."""

# Statistik deskriptif
print("\nStatistik deskriptif:")
df.describe()

"""Pada tahap ini dilakukan describe data set, untuk mendapatkan informasi data seperti minimum, maximum, mean, hingga total data."""

# Visualisasi distribusi fitur numerik
numerical_cols = [col for col in df.columns if col not in [
    'Tuition fees up to date',
    'Scholarship holder',
    'Displaced',
    'Application order',
    'Daytime/evening attendance\t',
    'Target'
]]

n = len(numerical_cols)
cols = 3
rows = math.ceil(n / cols)

plt.figure(figsize=(cols * 5, rows * 4))

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.savefig('distribusi_numerik_new.png', dpi=300)
plt.show()

"""Pada tahap ini dilakukan visualisasi data, untuk mendapatkan informasi distribusi data dari setiap fitur numerical."""

# Visualisasi distribusi fitur kategorikal
categorical_cols = [
    'Tuition fees up to date',
    'Scholarship holder',
    'Displaced',
    'Application order',
    'Daytime/evening attendance\t',
    'Target'
]

n = len(categorical_cols)
cols = 3
rows = math.ceil(n / cols)

plt.figure(figsize=(cols * 5, rows * 4))

for i, col in enumerate(categorical_cols, 1):
    plt.subplot(rows, cols, i)
    sns.countplot(data=df, x=col)
    plt.title(col)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig('distribusi_kategorikal.png', dpi=300)
plt.show()

"""Pada tahap ini dilakukan visualisasi distribusi data kategorikal.

### Exploration Data Analysis

Cek Korelasi Antar Fitur
"""

df_encoded = df.copy()

# Encode Target: Graduate = 1, Dropout = 0
df_encoded['Target'] = df_encoded['Target'].replace({'Graduate': 1, 'Dropout': 0})

# Korelasi dan visualisasi heatmap
plt.figure(figsize=(16, 10))
correlation = df_encoded.corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Heatmap Korelasi Antar Fitur (Termasuk Target Ter-Encode)")
plt.show()

"""Pada tahap ini dilakukan visualisasi confusion matrix dari seluruh fitur yang ada, untuk mengetahui hubungan antara satu fitur dengan yang lain, dan mendapatkan insight fitur mana yang terkait cukup kuat satu sama lain.

Distribusi Fitur Numerik Berdasarkan Target
"""

# Visualisasi distribusi data fitur numerik berdasarkan label target
numerical_features = [
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Admission grade",
    "Previous qualification (grade)",
    "GDP"
]

for feature in numerical_features:
    plt.figure(figsize=(6,4))
    sns.kdeplot(data=df_encoded, x=feature, hue="Target", common_norm=False, fill=True)
    plt.title(f"Distribusi {feature} Berdasarkan Target")
    plt.show()

"""Pada tahap ini dilakukan visualisasi distribusi data fitur berdasarkan label target, untuk mengetahui fitur mana yang paling berhubungan paling kuat dengan label target.

Analisi Fitur Kategorikal
"""

# Visualisasi distribusi data fitur kategorikal berdasarkan label target
categorical_features = [
    "Scholarship holder",
    "Tuition fees up to date",
    "Displaced",
    "Daytime/evening attendance\t",
    "Course"
]

for cat in categorical_features:
    if df_encoded[cat].nunique() < 20:
        plt.figure(figsize=(6,4))
        sns.countplot(x=cat, hue="Target", data=df_encoded)
        plt.title(f"Distribusi {cat} berdasarkan Target")
        plt.xticks(rotation=45)
        plt.show()

"""Pada tahap ini dilakukan visualisasi distribusi data fitur kategorikal terhadap kolom target.

## Data Preparation
"""

# Cek data null
print("\nTotal Data Null:")
df.isnull().sum()

# Cek data duplikat
print("\nTotal Data Duplikat:")
df.duplicated().sum()

"""### Berdasarkan pengecekan, didapatkan bahwa dataset tidak memiliki data null dan data duplikat.

### Outlier Checking
"""

# Pilih fitur numerik
numerical_features = df_encoded.select_dtypes(include=[np.number]).columns.tolist()

# Visualisasi boxplot per fitur numerik
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df_encoded[feature])
    plt.title(f'Boxplot untuk fitur: {feature}')
    plt.xlabel(feature)
    plt.show()

"""Berdasarkan visualisasi box plot, didapatkan bahwa di beberapa fitur memiliki outlier, namun setelah dilakukan analisa lebih dalam, ada salah satu fitur yang perlu dilakukan penanganan lebih jauh terhadap outlier, agar tidak mempengaruhi model, yaitu fitur Admission grade.

### Outlier handling
"""

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

"""Setelah penanganan outlier, didapatkan hasil total data yang dapat digunakan sebanyak 3559 data.

Berikut merupakan visualisasi box plot, setelah outlier handling pada fitur admission grade.
"""

# Mengambil fitur numerik dari df_encoded yang sudah bersih
numerical_features = df_encoded.select_dtypes(include=[np.number]).columns.tolist()

# Visualisasi boxplot ulang
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df_encoded[feature])
    plt.title(f'Boxplot untuk fitur: {feature}')
    plt.xlabel(feature)
    plt.show()

"""### Data Splitting"""

# Misal: X adalah fitur, y adalah target
X = df_encoded.drop(columns=['Target'])
y = df_encoded['Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Cek hasil
print("Ukuran data latih:", X_train.shape)
print("Ukuran data uji:", X_test.shape)

"""Pada bagian ini dilakukan data splitting, menjadi 80:20. dengan kolom target "Target" yang berisikan Graduated & Dropout. dan didapatkan hasil Ukuran data latih: (2847, 19)
Ukuran data uji: (712, 19)

## Modelling

### Modelling
"""

# Inisialisasi model
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "SVM": SVC()
}

# Untuk menyimpan hasil evaluasi
results = []

# Train dan evaluasi semua model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

# Tampilkan hasil dalam DataFrame
results_df = pd.DataFrame(results)
print("Hasil Evaluasi Model:")
print(results_df.sort_values(by="F1 Score", ascending=False))

"""Tahap ini adalah proses training dan evaluasi model machine learning. Beberapa algoritma yang digunakan seperti Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, dan Support Vector Machine (SVM) diinisialisasi dan dilatih menggunakan data training (X_train, y_train). Setelah model dilatih, prediksi dilakukan terhadap data uji (X_test), lalu hasil prediksi dibandingkan dengan label asli (y_test) menggunakan metrik evaluasi seperti Accuracy, Precision, Recall, dan F1 Score. Seluruh hasil diekstrak ke dalam bentuk DataFrame untuk dianalisis dan dibandingkan, sehingga dapat diketahui model mana yang memiliki performa terbaik berdasarkan nilai F1 Score.

### Model Optimation
"""

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

"""Tahap ini merupakan tuning hyperparameter untuk model Logistic Regression menggunakan teknik GridSearchCV. GridSearchCV melakukan pencarian kombinasi terbaik dari parameter C, penalty, solver, dan max_iter berdasarkan skor F1 dengan validasi silang sebanyak 5 fold. Setelah menemukan parameter terbaik (best_params_), model terbaik (best_estimator_) digunakan untuk melakukan prediksi pada data uji (X_test).

### Evaluation

Confusion Matrix
"""

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)

# Visualisasi confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_logreg.classes_)
disp.plot(cmap=plt.cm.Blues)

plt.title("Confusion Matrix - Logistic Regression (Best Model)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.grid(False)
plt.show()

"""Tahap ini dilakukan untuk menampilkan visualisasi confusion matrix dari model Logistic Regression terbaik yang telah diperoleh melalui GridSearch. Confusion matrix memberikan gambaran detail tentang jumlah prediksi benar dan salah dari masing-masing kelas, sehingga memudahkan dalam mengevaluasi kesalahan model (false positives dan false negatives).

Accuracy
"""

print('Accuracy:', accuracy_score(y_test, y_pred_best))

"""## Conclusion

Melalui serangkaian tahapan seperti eksplorasi data, visualisasi distribusi fitur, pelatihan berbagai algoritma klasifikasi, dan tuning hyperparameter, diperoleh hasil bahwa model Logistic Regression memberikan performa terbaik.

Model Logistic Regression yang telah di-tuning berhasil mencapai akurasi sebesar 91%, serta menunjukkan performa yang baik pada metrik lain seperti precision, recall, dan F1-score. Model ini tidak hanya sederhana dan efisien, tetapi juga cukup akurat untuk digunakan sebagai dasar dalam sistem pendukung keputusan (decision support system) bagi pihak kampus untuk mengidentifikasi mahasiswa yang berisiko dropout lebih awal.

Dengan hasil ini, model dapat menjadi alat bantu strategis dalam meningkatkan angka kelulusan dan merancang intervensi yang tepat sasaran bagi mahasiswa yang membutuhkan perhatian lebih.
"""