import sqlite3

def seed_data():
    conn = sqlite3.connect("ocular.db")
    cursor = conn.cursor()
    
    # This adds the specific ID your terminal was looking for
    test_patient = ("1001", "John Doe", 45, "Male")
    
    try:
        cursor.execute("""
            INSERT INTO patients (patient_id, name, age, gender) 
            VALUES (?, ?, ?, ?)
        """, test_patient)
        conn.commit()
        print("Success: Patient '1001' is now in the database.")
    except sqlite3.IntegrityError:
        print("Patient '1001' already exists.")
    finally:
        conn.close()

if __name__ == "__main__":
    seed_data()