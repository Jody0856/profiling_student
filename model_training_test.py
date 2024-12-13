import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, mean_squared_error, silhouette_score
import joblib
import numpy as np
from sqlalchemy.exc import SQLAlchemyError

# Koneksi ke database
DATABASE_URL = "mysql+pymysql://root:@localhost:3306/profiling_students"  # Ganti dengan kredensial MySQL Anda
engine = create_engine(DATABASE_URL)

# Fungsi untuk membaca, membersihkan, dan menggabungkan data
def load_and_clean_data():
    print("Memulai proses pembersihan data...")
    try:
        data_mahasiswa = pd.read_sql("""
            SELECT 
                npm_mahasiswa,	nama_mahasiswa,	prodi_mahasiswa,	angkatan_mahasiswa,	ipk_mahasiswa,	status_mahasiswa,	pembimbing_tugas_akhir
            FROM data_mahasiswa
            """, engine)
        data_krs_mahasiswa = pd.read_sql(""" 
            SELECT npm_mahasiswa,	jenis_semester,	tahun_semester,	kode_kelas,	kode_matkul,	nama_matkul,	sks_matakuliah,	total_hadir,	total_pertemuan,	total_terlaksana,	total_tidak_hadir,	kode_nilai,	kategori_matakuliah,
                CASE 
                    WHEN kode_nilai = 'A' THEN 4.0
                    WHEN kode_nilai = 'B' THEN 3.0
                    WHEN kode_nilai = 'C' THEN 2.0
                    WHEN kode_nilai = 'D' THEN 1.0
                    WHEN kode_nilai = 'E' THEN 0.0
                END AS nilai 
            FROM data_krs_mahasiswa a
            
            
            WHERE kode_nilai IN ('A','B','C','D','E')
        """, engine)
        data_kegiatan_mahasiswa = pd.read_sql("""SELECT 
             npm_mahasiswa,
             nama_kegiatan,
             tingkat_kegiatan,
             tanggal_kegiatan,
             CASE 
                    WHEN tingkat_kegiatan = 'international' THEN 10
                    WHEN tingkat_kegiatan = 'national' THEN 8
                    WHEN tingkat_kegiatan = 'provinsi' THEN 5
                    WHEN tingkat_kegiatan = 'lokal' THEN 5
                END AS bobot_kegiatan 
        FROM data_kegiatan_mahasiswa a""", engine)
        # Drop duplicate records
        data_mahasiswa.drop_duplicates(subset=['npm_mahasiswa'], inplace=True)
        data_krs_mahasiswa.drop_duplicates(subset=['npm_mahasiswa', 'kode_matkul'], inplace=True)
        data_kegiatan_mahasiswa.drop_duplicates(subset=['npm_mahasiswa', 'nama_kegiatan'], inplace=True)
        # Hitung keterlibatan kegiatan
        data_kegiatan_mahasiswa['keterlibatan_kegiatan'] = data_kegiatan_mahasiswa.groupby('npm_mahasiswa')['npm_mahasiswa'].transform('count')
        data_kegiatan_mahasiswa['total_bobot'] = data_kegiatan_mahasiswa.groupby('npm_mahasiswa')['bobot_kegiatan'].transform('sum')
        # Keep all relevant columns including 'nama_kegiatan'
        data_kegiatan_mahasiswa = data_kegiatan_mahasiswa[['npm_mahasiswa', 'nama_kegiatan', 'keterlibatan_kegiatan', 'bobot_kegiatan', 'total_bobot']].drop_duplicates()
        # Merge data
        merged_data = pd.merge(data_mahasiswa, data_krs_mahasiswa, on="npm_mahasiswa", how="inner")
        merged_data = pd.merge(merged_data, data_kegiatan_mahasiswa, on="npm_mahasiswa", how="left")
        
        merged_data['keterlibatan_kegiatan'].fillna(0, inplace=True)
        merged_data['total_bobot'].fillna(0, inplace=True)
        # Tambahkan rata-rata nilai dan IPK
        merged_data['nilai_rata_rata'] = merged_data.groupby('npm_mahasiswa')['nilai'].transform('mean')
        merged_data['ipk_mahasiswa'].fillna(0, inplace=True)

        print("Pembersihan data selesai.")
        return merged_data, data_kegiatan_mahasiswa
    except SQLAlchemyError as e:
        print(f"Error:: {e}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise

# Fungsi untuk melatih model dan menyimpannya
def train_and_export_models(merged_data):
    print("Memulai proses pelatihan model...")
    # Fitur dan label
    X = merged_data[['ipk_mahasiswa', 'nilai_rata_rata', 'keterlibatan_kegiatan']].fillna(0)
    # Label kelulusan (tugas regresi)    
    def graduation_probability(row):
        if row['status_mahasiswa'] == 'Lulus':
            return 1.0
        elif row['status_mahasiswa'] == 'Aktif':
            attendance_rate = (row['total_hadir'] / row['total_terlaksana']) if row['total_terlaksana'] > 0 else 0
            weighted_score = (
                0.8 * row['nilai_rata_rata'] +
                0.2 * attendance_rate
            )
            return np.clip(weighted_score / 4.0, 0, 1)
        else:
            return 0.0

    def achievement_label(row):
        score = (row['total_bobot'] > 60) if row['keterlibatan_kegiatan'] > 0 else 0
        return np.clip(score, 0, 1)
    
    merged_data['graduation_probability'] = merged_data.apply(graduation_probability, axis=1)
    y_graduation = merged_data['graduation_probability']
    
    merged_data['achievement'] = merged_data.apply(achievement_label, axis=1)
    y_achievement = merged_data['achievement']

    # Model kelulusan (regresi)
    X_train_grad, X_test_grad, y_train_grad, y_test_grad = train_test_split(
        X, y_graduation, test_size=0.3, random_state=0
    )
    graduation_model = RandomForestRegressor(n_estimators=100, random_state=0)
    graduation_model.fit(X_train_grad, y_train_grad)
    joblib.dump(graduation_model, 'student_graduation_model.pkl')
    y_pred_grad = graduation_model.predict(X_test_grad)
    mse_grad = mean_squared_error(y_test_grad, y_pred_grad)
    print(f'MSE Model Kelulusan: {mse_grad}')

    # Model prestasi (klasifikasi)
    X_train_ach, X_test_ach, y_train_ach, y_test_ach = train_test_split(X, y_achievement, test_size=0.3, random_state=0)
    achievement_model = RandomForestClassifier(n_estimators=100, random_state=0)
    achievement_model.fit(X_train_ach, y_train_ach)
    joblib.dump(achievement_model, 'student_achievement_model.pkl')
    y_pred_ach = achievement_model.predict(X_test_ach)
    print('Evaluasi Model Prestasi:\n', classification_report(y_test_ach, y_pred_ach))
    print("Model berhasil dilatih dan disimpan.")




def train_interest_model_with_kmeans(data_kegiatan_mahasiswa):
    additional_stop_words = [
    'di', 'dan', 'ke', 'dari', 'yang', 'pada', 'untuk', 'dengan', 'sebagai', 'atau', 'ini', 'itu', 'oleh',
    'uib', 'webinar', '2020', '2021', '2022', '2023', '2024', 'bidang', 'seminar', 'lokal', 'kepribadian',
    'nasional', 'national', 'student', 'mbkm', 'pengembangan', 'digital', 'program', 'series', 'mahasiswa',
    'sosialisasi', 'batam', 'club', 'project', 'dalam', 'course', 'fundamentals', 'agile', 'associate'
    ]

    print("Memulai pelatihan model prediksi minat mahasiswa menggunakan K-Means...")
    
    if 'nama_kegiatan' not in data_kegiatan_mahasiswa.columns:
        print("Kolom 'nama_kegiatan' tidak ditemukan!")
        return

    students_with_many_activities = data_kegiatan_mahasiswa[
        data_kegiatan_mahasiswa['keterlibatan_kegiatan'] > 0
    ]

    if students_with_many_activities.empty:
        print("Kegiatan mahasiswa 0.")
        return

    # Combine English and additional Indonesian stop words
    vectorizer = CountVectorizer(max_features=100, stop_words='english')
    vectorizer.set_params(stop_words=list(set(vectorizer.get_stop_words()).union(additional_stop_words)))

    activity_matrix = vectorizer.fit_transform(students_with_many_activities['nama_kegiatan'].fillna(""))

    feature_matrix = pd.DataFrame(
        activity_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    feature_matrix['npm_mahasiswa'] = students_with_many_activities['npm_mahasiswa'].values

    activity_features = pd.merge(
        feature_matrix,
        students_with_many_activities[['npm_mahasiswa', 'total_bobot']],
        on='npm_mahasiswa',
        how='left'
    ).fillna(0)

    kmeans = KMeans(n_clusters=5, random_state=0)
    clusters = kmeans.fit_predict(activity_features.drop(columns=['npm_mahasiswa']))

    activity_features['cluster'] = clusters
    cluster_map = activity_features[['npm_mahasiswa', 'cluster']]

    silhouette_avg = silhouette_score(
        activity_features.drop(columns=['npm_mahasiswa', 'cluster']),
        clusters
    )
    print(f"Silhouette Score: {silhouette_avg}")

    for cluster in sorted(activity_features['cluster'].unique()):
        keywords = activity_features[activity_features['cluster'] == cluster].drop(columns=['npm_mahasiswa', 'cluster', 'total_bobot']).sum().sort_values(ascending=False)
        print(f"Cluster {cluster} Keywords: {keywords.head(10).index.tolist()}")

    joblib.dump(kmeans, 'student_interest_model.pkl')
    cluster_map.to_csv("student_interest_clusters.csv", index=False)

    print("Model K-Means berhasil dilatih dan disimpan.")

# Eksekusi utama
merged_data, data_kegiatan_mahasiswa = load_and_clean_data()
#train_and_export_models(merged_data)
train_interest_model_with_kmeans(data_kegiatan_mahasiswa)
