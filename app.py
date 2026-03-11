from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import numpy as np
import tensorflow as tf
from utils.preprocess import preprocess_image
import sqlite3
import os
import hashlib

app = Flask(__name__)
app.secret_key = "ocular_secret_key"

# --- CONFIGURATION ---
UPLOAD_FOLDER = "static/uploads"
DB_NAME = "ocular.db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model globally (Uncomment when ready)
model = tf.keras.models.load_model("model/best_siamese_model.keras", compile=False)

def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- NEW: AJAX ROUTE FOR PATIENT VERIFICATION ---
@app.route("/get_patient/<p_id>")
def get_patient(p_id):
    if "user" not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    conn = get_db()
    patient = conn.execute("SELECT name, age, gender FROM patients WHERE patient_id = ?", (p_id,)).fetchone()
    conn.close()
    
    if patient:
        return jsonify({
            "success": True,
            "name": patient["name"],
            "age": patient["age"],
            "gender": patient["gender"]
        })
    return jsonify({"success": False, "message": "Patient not found"})

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["username"]
        password = hash_password(request.form["password"])
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password)).fetchone()
        conn.close()
        if user:
            session["user"] = user['id']
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session: return redirect(url_for("login"))
    conn = get_db()
    # Fetch doctor info
    user_data = conn.execute("SELECT name, specialization, hospital FROM users WHERE id=?", (session["user"],)).fetchone()
    # Fetch recent history
    history = conn.execute("""
        SELECT pr.*, p.name as patient_name 
        FROM patient_records pr 
        JOIN patients p ON pr.patient_id = p.patient_id 
        WHERE pr.doctor_id=? 
        ORDER BY pr.timestamp DESC LIMIT 5
    """, (session["user"],)).fetchall()
    conn.close()
    return render_template("dashboard.html", user=user_data, history=history)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        patient_id = request.form.get("patient_id")
        left_eye = request.files.get('left_eye')
        right_eye = request.files.get('right_eye')

        # Check if files were actually uploaded
        if not left_eye or left_eye.filename == '':
            flash("Missing left eye image")
            return redirect(request.url)
        if not right_eye or right_eye.filename == '':
            flash("Missing right eye image")
            return redirect(request.url)

        # Save files and get paths
        left_path = os.path.join(UPLOAD_FOLDER, left_eye.filename)
        right_path = os.path.join(UPLOAD_FOLDER, right_eye.filename)
        left_eye.save(left_path)
        right_eye.save(right_path)

        # 1. Preprocess and Predict
        processed_left, q_left = preprocess_image(left_path)
        processed_right, q_right = preprocess_image(right_path)
        
        # Ensure the input shape matches your Siamese model's requirement
        prediction = model.predict([np.expand_dims(processed_left, 0), np.expand_dims(processed_right, 0)])
        
        # Map the prediction to your disease classes
        classes = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract"]
        class_idx = np.argmax(prediction[0])
        primary_disease = classes[class_idx]
        confidence = round(float(np.max(prediction[0]) * 100), 2)
        
        overall_findings = f"{primary_disease} ({confidence}%)"

        # 2. Save to history (patient_records table)
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO patient_records 
            (doctor_id, patient_id, left_eye_img, right_eye_img, left_diagnosis, right_diagnosis, overall_findings)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session["user"], patient_id, left_eye.filename, right_eye.filename, 
              "Signs of lesion", "Normal", overall_findings))
        conn.commit()
        
        # Get patient details for the result page
        patient = conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
        conn.close()

        # 3. Render the Prediction (result.html)
        return render_template("result.html", 
                               primary_disease=primary_disease, 
                               confidence=confidence,
                               left_img=left_eye.filename,
                               right_img=right_eye.filename,
                               patient=patient,
                               description="Automated analysis detected patterns consistent with " + primary_disease)

    # This handles the initial GET request to show the page
    return render_template("upload.html")

@app.route("/history")
def history():
    if "user" not in session: return redirect(url_for("login"))
    conn = get_db()
    # JOIN is necessary to get the patient name from the patients table
    history = conn.execute('''
        SELECT pr.*, p.name as patient_name 
        FROM patient_records pr 
        JOIN patients p ON pr.patient_id = p.patient_id 
        WHERE pr.doctor_id = ? ORDER BY pr.timestamp DESC
    ''', (session["user"],)).fetchall()
    conn.close()
    return render_template("history.html", history=history)

@app.route("/view_result/<int:record_id>")
def view_result(record_id):
    conn = get_db()
    record = conn.execute('''
        SELECT pr.*, p.name, p.age, p.gender FROM patient_records pr 
        JOIN patients p ON pr.patient_id = p.patient_id WHERE pr.id = ?
    ''', (record_id,)).fetchone()
    conn.close()
    
    findings = record['overall_findings']
    primary = findings.split(' (')[0] if '(' in findings else findings
    conf = findings.split('(')[1].replace('%)', '') if '(' in findings else "0"

    return render_template("result.html", primary_disease=primary, confidence=conf,
                               # Ensure these variable names match what result.html expects
                               left_img=record['left_eye_img'], 
                               right_img=record['right_eye_img'],
                               left_status=record['left_diagnosis'], 
                               right_status=record['right_diagnosis'],
                               patient=record, 
                               description="Analysis based on fundus image feature extraction.")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)