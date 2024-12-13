from flask import Flask, jsonify, request, abort
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError  # Untuk menangani error SQLAlchemy
import joblib
import numpy as np
from flask_cors import CORS  # Import library flask-cors
from sklearn.feature_extraction.text import CountVectorizer

# Memuat model K-Means dan vektorizer
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Middleware CORS
DATABASE_URL = "mysql+pymysql://root:@localhost:3306/profiling_students"  # Ganti dengan kredensial database Anda
engine = create_engine(DATABASE_URL)

# Memuat model yang diperlukan
graduation_model = joblib.load('student_graduation_model.pkl')
achievement_model = joblib.load('student_achievement_model.pkl')
interest_model = joblib.load('student_interest_model.pkl')

vectorizer = CountVectorizer(max_features=50, stop_words='english')

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



@app.route("/predict", methods=["POST"])
def predict_student_status():
    try:
        npm = request.form.get("npm_mahasiswa")
        if not npm:
            return jsonify({"error": "npm_mahasiswa wajib diisi"}), 400

        # Validasi NPM sebagai angka
        try:
            npm = int(npm)
        except ValueError:
            return jsonify({"error": "npm_mahasiswa harus berupa angka"}), 400

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
        X_new_graduation = np.array([[ipk_mahasiswa, nilai_rata_rata, keterlibatan_kegiatan]])  # Tambahkan keterlibatan_kegiatan
        graduation_prob = graduation_model.predict(X_new_graduation)[0] * 100  # Prediksi langsung

        # Prediksi peluang berprestasi
        X_new_achievement = np.array([[ipk_mahasiswa, nilai_rata_rata, keterlibatan_kegiatan]])  # Sesuaikan jumlah fitur
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

        # Ekstraksi fitur dari kegiatan untuk K-Means
        activity_names = [a['nama_kegiatan'] for a in activity_list if a.get('nama_kegiatan')]
        activity_vector = vectorizer.transform([' '.join(activity_names)]).toarray()  # Gabungkan nama kegiatan

        # Gabungkan dengan bobot kegiatan
        total_bobot = sum(a.get('bobot_kegiatan', 0) for a in activity_list)
        kmeans_features = np.hstack((activity_vector, [[total_bobot]]))  # Fitur harus memiliki dimensi yang sama dengan model

        # Prediksi klaster minat
        minat_cluster = int(interest_model.predict(kmeans_features)[0])

        # Kategori mahasiswa
        kategori = 'Memenuhi Standar' if achievement_prob >= 0.5 else 'Butuh Peningkatan'

        # Kembalikan hasil prediksi
        return jsonify({
            "npm_mahasiswa": npm,
            "nama_mahasiswa": student_data['nama_mahasiswa'],
            "status_mahasiswa": student_data['status_mahasiswa'],
            "prodi_mahasiswa": student_data['prodi_mahasiswa'],
            'nilai_rata_rata': student_data['nilai_rata_rata'],
            "ipk_mahasiswa": float(ipk_mahasiswa),
            "persentase_kelulusan": float(graduation_prob),
            "kategori_mahasiswa": kategori,
            "keterlibatan_kegiatan": int(keterlibatan_kegiatan),
            "minat_mahasiswa_cluster": minat_cluster,
            "daftar_mata_kuliah": course_list,
            "daftar_kegiatan": activity_list,
        })

    except Exception as e:
        return jsonify({"error": f"Kesalahan pada server: {str(e)}"}), 500

@app.route("/predict_interest", methods=["POST"])
def predict_student_interest():
    try:
        npm = request.form.get("npm_mahasiswa")
        if not npm:
            return jsonify({"error": "npm_mahasiswa wajib diisi"}), 400

        # Validasi NPM sebagai angka
        try:
            npm = int(npm)
        except ValueError:
            return jsonify({"error": "npm_mahasiswa harus berupa angka"}), 400

        # Ambil kegiatan mahasiswa berdasarkan NPM
        try:
            with engine.connect() as connection:
                activity_query = text("""
                    SELECT nama_kegiatan
                    FROM data_kegiatan_mahasiswa
                    WHERE npm_mahasiswa = :npm
                """)
                result = connection.execute(activity_query, {"npm": npm}).fetchall()
                
                if not result:
                    return jsonify({"error": "Data kegiatan mahasiswa tidak ditemukan"}), 404

                # Gabungkan semua kegiatan mahasiswa menjadi satu teks
                activities = " ".join([row['nama_kegiatan'] for row in result])
        except SQLAlchemyError as e:
            return jsonify({"error": f"Kesalahan SQL: {e}"}), 500

        # Transformasi kegiatan mahasiswa menjadi matriks fitur
        activity_matrix = vectorizer.transform([activities]).toarray()

        # Prediksi cluster menggunakan model K-Means
        cluster = interest_model.predict(activity_matrix)[0]

        return jsonify({
            "npm_mahasiswa": npm,
            "predicted_interest_cluster": int(cluster),
            "cluster_keywords": [
                keyword
                for keyword, score in sorted(
                    zip(vectorizer.get_feature_names_out(), activity_matrix[0]),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            ]  # Menampilkan kata kunci teratas untuk cluster
        })
    except Exception as e:
        return jsonify({"error": f"Kesalahan umum: {e}"}), 500


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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)