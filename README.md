# profiler_student

Sistem Profiler - kecerdasan buatan

Menjalankan sistem profiler
membuat dan mengimpor database:

- jalankan sql server lewat laragon atau xampp dan jalankan kode script convert_excel_to_sql.py dgn kode python convert_excel_to_sql.py
  jalankan sistem profiler:
- jalankan kode script python train_model_with_sql.py
- jalankan kode script python api_server_sql.py

- hostname adalah: http://localhost:8000/
- endpoint ada 2 yaitu: /students atau http://localhost:8000/students untuk dapatkan list semua mahasiswa
- /predict atau http://localhost:8000/predict untuk profiling mahasiswa, diharapkan agar mengirimkan parameter NPM dengan metode post
- pip install pandas scikit-learn fastapi uvicorn joblib openpyxl flask sqlalchemy

atau dengan ini
- pip install -r requirements.txt
