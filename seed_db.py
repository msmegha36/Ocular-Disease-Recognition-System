import sqlite3

# Make sure this matches the DB_NAME in your app.py
DB_NAME = "ocular.db" 

def seed_data():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Use a list of tuples for multiple patients
    test_patients = [
        ("1001", "John Doe", 45, "Male"),
        ("1002", "Lily James", 35, "Female")
    ]
    
    try:
        # executemany is required to loop through the list above
        cursor.executemany("""
            INSERT INTO patients (patient_id, name, age, gender) 
            VALUES (?, ?, ?, ?)
        """, test_patients)
        conn.commit()
        print(f"Success! Added {cursor.rowcount} patients.")
    except sqlite3.IntegrityError:
        print("Error: One of these IDs (1001 or 1002) already exists in the table.")
    finally:
        conn.close()

if __name__ == "__main__":
    seed_data()