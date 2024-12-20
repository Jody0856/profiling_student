from flask import Flask, jsonify, request, abort
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import joblib
import numpy as np
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import os
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://profiler-student-frontend.vercel.app"]}})

DATABASE_LOCAL = "mysql+pymysql://root:@localhost:3306/profiling_students" ##Untuk Lokal
#DATABASE_SERVER = "postgresql://root:lh1wLWLJAMjZKKmQ4iLTyxVdRlEOaLaC@dpg-cte5c5tds78s739j1u30-a.oregon-postgres.render.com/profiling_students" ##untuk server

# Use PostgreSQL for the server and MySQL for local development
# Dynamically select database URL
DATABASE_URL = os.getenv("DATABASE_URL", DATABASE_LOCAL)

engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as connection:
        print("Connected to database successfully!")
except Exception as e:
    print(f"Error: {e}")
# Load models
graduation_model = joblib.load('student_graduation_model.pkl')
achievement_model = joblib.load('student_achievement_model.pkl')
interest_model = joblib.load('student_interest_model_supervised.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

def get_student_data(npm):
    try:
        with engine.connect() as connection:
            student_query = text("""
                SELECT 
                    m.npm_mahasiswa, 
                    m.nama_mahasiswa, 
                    m.status_mahasiswa, 
                    m.prodi_mahasiswa, 
                    m.ipk_mahasiswa, 
                    (SELECT COUNT(*) 
                     FROM data_kegiatan_mahasiswa 
                     WHERE npm_mahasiswa = m.npm_mahasiswa) AS keterlibatan_kegiatan,
                    (SELECT AVG(CASE 
                                 WHEN kode_nilai = 'A' THEN 4.0
                                 WHEN kode_nilai = 'B' THEN 3.0
                                 WHEN kode_nilai = 'C' THEN 2.0
                                 WHEN kode_nilai = 'D' THEN 1.0
                                 WHEN kode_nilai = 'E' THEN 0.0
                             END)
                     FROM data_krs_mahasiswa 
                     WHERE npm_mahasiswa = m.npm_mahasiswa
                     AND kode_nilai IS NOT NULL) AS nilai_rata_rata
                FROM data_mahasiswa m
                WHERE m.npm_mahasiswa = :npm
            """)
            result = connection.execute(student_query, {"npm": npm}).fetchone()
            
            if result:
                return dict(result._mapping)
            return None
    except SQLAlchemyError as e:
        raise RuntimeError(f"Kesalahan SQLAlchemy: {e}")
    except Exception as e:
        raise RuntimeError(f"Kesalahan umum: {e}")

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    additional_stop_words = [
        'di', 'dan', 'ke', 'dari', 'yang', 'pada', 'untuk', 'dengan', 'sebagai',
        'atau', 'ini', 'itu', 'oleh', 'uib', 'webinar', 'seminar', 'nasional',
        'lokal', 'program', 'mahasiswa', 'dalam', 'bidang'
    ]
    stop_words.update(additional_stop_words)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

@app.route("/students", methods=["GET"])
def list_students():
    """
    Mengembalikan daftar semua mahasiswa yang tersimpan di database.
    """
    try:
        # Membuka koneksi
        with engine.connect() as connection:
            query = text("""
                SELECT npm_mahasiswa, nama_mahasiswa, prodi_mahasiswa,status_mahasiswa, ipk_mahasiswa
                FROM data_mahasiswa where status_mahasiswa not in ('Mahasiswa Asing')
            """)
            result = connection.execute(query).mappings()

            # Membentuk daftar mahasiswa dari hasil query
            students = [
                {
                    "npm_mahasiswa": row["npm_mahasiswa"],
                    "nama_mahasiswa": row["nama_mahasiswa"],
                    "prodi_mahasiswa": row["prodi_mahasiswa"],
                    "status_mahasiswa": row["status_mahasiswa"],
                    "ipk_mahasiswa": row["ipk_mahasiswa"]
                }
                for row in result
            ]

        # Jika tidak ada data mahasiswa
        if not students:
            return jsonify({"message": "Tidak ada data mahasiswa"}), 404

        # Mengembalikan hasil dalam format JSON
        return jsonify({"students": students}), 200
    
    except Exception as e:
        # Penanganan error
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500

@app.route("/predict", methods=["POST"])
def predict_student_status():
    try:
        npm = request.form.get("npm_mahasiswa")
        if not npm:
            return jsonify({"error": "npm_mahasiswa wajib diisi"}), 400

        npm = str(npm)
        # Ambil data mahasiswa
        student_data = get_student_data(npm)
        if not student_data:
            abort(404, description="Data mahasiswa tidak ditemukan")

        # Validasi data mahasiswa
        ipk_mahasiswa = student_data['ipk_mahasiswa']
        nilai_rata_rata = student_data['nilai_rata_rata']
        keterlibatan_kegiatan = student_data['keterlibatan_kegiatan']

        if ipk_mahasiswa is None or nilai_rata_rata is None:
            return jsonify({"error": "Data mahasiswa tidak lengkap untuk prediksi"}), 400

        # Prediksi peluang kelulusan
        X_new_graduation = np.array([[ipk_mahasiswa, nilai_rata_rata, keterlibatan_kegiatan]])
        graduation_prob = graduation_model.predict(X_new_graduation)[0] * 100

        # Prediksi peluang berprestasi
        X_new_achievement = np.array([[ipk_mahasiswa, nilai_rata_rata, keterlibatan_kegiatan]])
        achievement_prob = achievement_model.predict_proba(X_new_achievement)[0][1]

        # Mengambil data mata kuliah mahasiswa
        with engine.connect() as connection:
            courses_query = text("""
                SELECT nama_matkul, kategori_matakuliah, tahun_semester, kode_nilai,  
                CASE 
                    WHEN kode_nilai = 'A' THEN 4.0
                    WHEN kode_nilai = 'B' THEN 3.0
                    WHEN kode_nilai = 'C' THEN 2.0
                    WHEN kode_nilai = 'D' THEN 1.0
                    WHEN kode_nilai = 'E' THEN 0.0
                END AS nilai,  jenis_semester, sks_matakuliah,
                total_hadir, total_terlaksana, total_tidak_hadir, total_pertemuan
                FROM data_krs_mahasiswa
                WHERE npm_mahasiswa = :npm
                AND kode_nilai IS NOT NULL
            """)
            course_list = [dict(row._mapping) for row in connection.execute(courses_query, {"npm": npm})]

        # Mengambil data kegiatan mahasiswa
        with engine.connect() as connection:
            activities_query = text("""
                SELECT nama_kegiatan, tingkat_kegiatan, tanggal_kegiatan
                FROM data_kegiatan_mahasiswa
                WHERE npm_mahasiswa = :npm
            """)
            activity_list = [dict(row._mapping) for row in connection.execute(activities_query, {"npm": npm})]

        # Preprocess activities and predict interests
        preprocessed_activities = [preprocess_text(activity['nama_kegiatan']) for activity in activity_list]
        tfidf_features = tfidf_vectorizer.transform(preprocessed_activities)
        interest_predictions = interest_model.predict(tfidf_features)

        # Calculate frequencies of predicted interests
        interest_counts = {}
        for interest in interest_predictions:
            interest_counts[interest] = interest_counts.get(interest, 0) + 1

        # Determine dominant interest with a threshold
        total_predictions = sum(interest_counts.values())
        dominant_interest, count = max(interest_counts.items(), key=lambda x: x[1])

        # Assign "Other" if no interest has significant dominance
        if count / total_predictions < 0.5:
            dominant_interest = "Belum ada"

        # Assign category based on achievement probability
        kategori = 'Memenuhi Standar' if achievement_prob >= 0.5 else 'Butuh Peningkatan'
        
        # Return the combined result
        return jsonify({
            "npm_mahasiswa": npm,
            "nama_mahasiswa": student_data['nama_mahasiswa'],
            "status_mahasiswa": student_data['status_mahasiswa'],
            "prodi_mahasiswa": student_data['prodi_mahasiswa'],
            "nilai_rata_rata": student_data['nilai_rata_rata'],
            "ipk_mahasiswa": float(ipk_mahasiswa),
            "persentase_kelulusan": float(graduation_prob),
            "kategori_mahasiswa": kategori,
            "keterlibatan_kegiatan": int(keterlibatan_kegiatan),
            "daftar_mata_kuliah": course_list,
            "daftar_kegiatan": activity_list,
            "dominant_interest": dominant_interest,
            "detailed_interests": interest_counts
        })

    except Exception as e:
        return jsonify({"error": f"Kesalahan pada server: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)