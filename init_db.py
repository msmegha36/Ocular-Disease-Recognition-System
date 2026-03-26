import sqlite3

def init_db():
    conn = sqlite3.connect("ocular.db")
    cursor = conn.cursor()

    # 1. Users Table
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

    # 2. Patients Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        phone TEXT,
        doctor_id INTEGER, 
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (doctor_id) REFERENCES users (id)
    )
    """)

    # 3. Patient Records Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patient_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_id INTEGER,
        patient_id INTEGER, 
        left_eye_img TEXT,
        right_eye_img TEXT,
        left_diagnosis TEXT,
        right_diagnosis TEXT,
        overall_findings TEXT,
        left_heatmap TEXT,
        right_heatmap TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (doctor_id) REFERENCES users (id),
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
    )
    ''')

    # --- STARTING ID LOGIC ---
    # We insert a record into SQLite's internal sequence tracker
    cursor.execute("INSERT OR IGNORE INTO sqlite_sequence (name, seq) VALUES ('patients', 1000)")
    # If the table already exists, we force the update to 1000
    cursor.execute("UPDATE sqlite_sequence SET seq = 1000 WHERE name = 'patients'")

    conn.commit()
    conn.close()
    print("Database Initialized. Patient IDs will now start from 1001.")

if __name__ == "__main__":
    init_db()