import sqlite3

def init_db():
    conn = sqlite3.connect("ocular.db")
    cursor = conn.cursor()

    # 1. Users Table (Doctors/Medical Staff)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT,
        specialization TEXT,
        hospital TEXT
    )
    """)

    # 2. Patient Records Table (Diagnosis History)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patient_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_id INTEGER,
        patient_name TEXT NOT NULL,
        patient_age INTEGER,
        patient_gender TEXT,
        left_eye_img TEXT,
        right_eye_img TEXT,
        left_diagnosis TEXT,
        right_diagnosis TEXT,
        overall_findings TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (doctor_id) REFERENCES users (id)
    )
    ''')

    conn.commit()
    conn.close()
    print("Database synchronized: 'users' and 'patient_records' tables are ready.")

if __name__ == "__main__":
    init_db()