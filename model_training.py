import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import numpy as np
from sklearn.metrics import classification_report, mean_squared_error
from sqlalchemy.exc import SQLAlchemyError
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
# Download NLTK data (run this once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

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
        # Periksa status mahasiswa
        if row['status_mahasiswa'] == 'Lulus':  # Jika sudah lulus, tingkat kelulusan 100%
            return 1.0
        elif row['status_mahasiswa'] == 'Aktif':  # Jika aktif, kita hitung kelulusan berdasarkan kriteria
            attendance_rate = (row['total_hadir'] / row['total_terlaksana']) if row['total_terlaksana'] > 0 else 0
            weighted_score = (
                0.8 * row['nilai_rata_rata'] +
                0.2 * attendance_rate
            )
            return np.clip(weighted_score / 4.0, 0, 1)  # Skor kelulusan
        else:
            return 0.0  # Mahasiswa yang tidak aktif atau lainnya, 0% kelulusan

    def achievement_label(row):
        score = (row['total_bobot'] > 60) if row['keterlibatan_kegiatan'] > 0 else 0
        return np.clip(score, 0, 1)  # Skor prestasi
    
    merged_data['graduation_probability'] = merged_data.apply(graduation_probability, axis=1)
    y_graduation = merged_data['graduation_probability']

    # Label prestasi (tugas klasifikasi)
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

interest_keywords = {
    # Technology and Programming
    "video game": "Gaming",
    "animation": "Animation",
    "programming": "Programming",
    "web programming": "Web Programming",
    "web development": "Web Programming",
    "frontend development": "Programming",
    "backend development": "Programming",
    "full-stack development": "Programming",
    "software engineering": "Programming",
    "app development": "Programming",
    "mobile development": "Mobile Programming",
    "android": "Mobile Programming",
    "ios development": "Mobile Programming",
    "game development": "Gaming",
    "game design": "Gaming",
    "cloud computing": "Cloud Computing",
    "aws": "Cloud Computing",
    "azure": "Cloud Computing",
    "gcp": "Cloud Computing",
    "blockchain": "IT",
    "iot": "IT",
    "devops": "IT",
    "database management": "IT",
    "sql": "IT",
    "nosql": "IT",
    "cybersecurity": "Cyber Security",
    "ethical hacking": "Cyber Security",
    "pen testing": "Cyber Security",
    "data science": "Data Science",
    "data analysis": "Data Science",
    "data visualization": "Data Science",
    "machine learning": "AI & Machine Learning",
    "deep learning": "AI & Machine Learning",
    "ai": "AI & Machine Learning",
    "robotics": "AI & Machine Learning",
    "natural language processing": "AI & Machine Learning",
    "computer vision": "AI & Machine Learning",
    "virtual reality": "Virtual Reality",
    "augmented reality": "Virtual Reality",

    # Design and Media
    "design": "Design",
    "graphic design": "Design",
    "desain grafis": "Design",
    "adobe": "Design",
    "photoshop": "Design",
    "illustrator": "Design",
    "ux design": "Design",
    "user experience": "Design",
    "ui design": "Design",
    "user interface": "Design",
    "cinematography": "Cinematography",
    "video editing": "Cinematography",
    "motion graphics": "Animation",
    "3d modeling": "Design",
    "3d animation": "Animation",
    "autocad": "Design",
    "solidworks": "Design",
    "blender": "Design",

    # Marketing and Business
    # "digital marketing": "Marketing",
    # "seo": "Marketing",
    # "content marketing": "Marketing",
    # "social media marketing": "Marketing",
    # "email marketing": "Marketing",
    #"e-commerce": "Business & Tech",
    "entrepreneurship": "Business & Tech",
    "startup": "Business & Tech",
    "business analytics": "Data Science",
    "financial modeling": "Business & Tech",
    "project management": "Project Management",
    "agile": "Project Management",
    "scrum": "Project Management",
    "kanban": "Project Management",
    #"business development": "Business & Tech",
    #"product management": "Business & Tech",
    "sales": "Business & Tech",

    # Miscellaneous
    "gaming": "Gaming",
    "photography": "Art & Photography",
    # "writing": "Creative Writing",
    # "public speaking": "Communication",
    # "teaching": "Education",
    # "research": "Education",
    # "data entry": "IT",
    # "event management": "Project Management",
    # "volunteering": "Community Service",
    # "sports": "Sports",
    # "fitness": "Health & Wellness",
    # "healthcare": "Health & Wellness",
    # "psychology": "Health & Wellness",
    # "biology": "STEM",
    # "chemistry": "STEM",
    # "physics": "STEM",
    # "mathematics": "STEM",
    # "statistics": "STEM",
    # "economics": "Business & Tech",
    # "law": "Law",
    # "history": "Arts & Humanities",
    # "philosophy": "Arts & Humanities",
    # "music": "Art & Music",
    # "art": "Art & Music",
    # "drawing": "Art & Music",
    # "painting": "Art & Music",
    # "community service": "Community Service",
    # "volunteering": "Community Service",
    # "leadership": "Leadership & Personal Development",
    # "teamwork": "Leadership & Personal Development"
}

# Text preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    additional_stop_words = [
        'di', 'dan', 'ke', 'dari', 'yang', 'pada', 'untuk', 'dengan', 'sebagai', 
        'atau', 'ini', 'itu', 'oleh', 'uib', 'webinar', 'seminar', 'nasional', 
        'lokal', 'program', 'mahasiswa', 'dalam', 'bidang'
    ]
    stop_words.update(additional_stop_words)
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

def assign_label(text):
    matched_labels = [
        label for keyword, label in interest_keywords.items() if keyword in text.lower()
    ]
    return matched_labels[0] if matched_labels else "Others"

def assign_label_with_fallback(text):
    label = assign_label(text)
    if label == "Others":
        return "Others"
    return label

def train_supervised_model(data_kegiatan_mahasiswa):
    print("Starting supervised interest model training...")
    
    # Preprocess text and assign labels
    data_kegiatan_mahasiswa['cleaned_kegiatan'] = data_kegiatan_mahasiswa['nama_kegiatan'].fillna("").apply(preprocess_text)
    data_kegiatan_mahasiswa['interest_label'] = data_kegiatan_mahasiswa['nama_kegiatan'].fillna("").apply(assign_label_with_fallback)
    
    # Filter labeled data
    labeled_data = data_kegiatan_mahasiswa[data_kegiatan_mahasiswa['interest_label'] != "Others"]
    
    if labeled_data.empty:
        print("No labeled data available for training.")
        return
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(labeled_data['cleaned_kegiatan'])
    y = labeled_data['interest_label']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # Train the classifier
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    
    # Save model and vectorizer
    joblib.dump(model, 'student_interest_model_supervised.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Model Evaluation:\n", classification_report(y_test, y_pred))
    print("Model successfully trained and saved.")

merged_data, data_kegiatan_mahasiswa = load_and_clean_data()
train_and_export_models(merged_data)
train_supervised_model(data_kegiatan_mahasiswa)