from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import tensorflow as tf
from utils.preprocess import preprocess_image
import sqlite3
import os
import hashlib

# --- INITIALIZATION ---
app = Flask(__name__)
app.secret_key = "ocular_secret_key"

# --- CONFIGURATION ---
MODEL_PATH = "model/best_siamese_model.keras"
# Initialize model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASSES = ["Normal", "Diabetes", "Glaucoma", "Cataract", "AMD", "Hypertension", "Myopia", "Others"]
UPLOAD_FOLDER = "static/uploads"
DB_NAME = "ocular.db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- ROUTES ---

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

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        hospital = request.form["hospital"]
        password = hash_password(request.form["password"])
        spec = request.form["specialization"]
        try:
            conn = get_db()
            conn.execute("INSERT INTO users (name, email, password, specialization, hospital) VALUES (?,?,?,?,?)",
                         (name, email, password, spec, hospital))
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return "Email already exists"
    return render_template("register.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    conn = get_db()
    user_data = conn.execute("SELECT name, specialization, hospital FROM users WHERE id=?", (session["user"],)).fetchone()
    
    # Fetch patient history for the dashboard table
    history = conn.execute("SELECT * FROM patient_records WHERE doctor_id=? ORDER BY timestamp DESC", (session["user"],)).fetchall()
    conn.close()
    
    if user_data:
        return render_template("dashboard.html", user=list(user_data), history=history)
    return redirect(url_for("login"))

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "user" not in session: 
        return redirect(url_for("login"))

    if request.method == "POST":
        p_name = request.form.get("patient_name")
        p_age = request.form.get("patient_age")
        p_gender = request.form.get("patient_gender")

        left_eye = request.files["left_eye"]
        right_eye = request.files["right_eye"]
        
        l_path = os.path.join(UPLOAD_FOLDER, left_eye.filename)
        r_path = os.path.join(UPLOAD_FOLDER, right_eye.filename)
        left_eye.save(l_path)
        right_eye.save(r_path)

        l_img = preprocess_image(l_path)
        r_img = preprocess_image(r_path)

        # Siamese Model Prediction
        l_preds = model.predict([l_img, l_img])[0]
        r_preds = model.predict([r_img, r_img])[0]
        joint_preds = model.predict([l_img, r_img])[0]

        # Get Single Top Class
        top_idx = np.argmax(joint_preds)
        primary_disease = CLASSES[top_idx]
        conf_val = round(float(joint_preds[top_idx]) * 100, 2)

        left_status = CLASSES[np.argmax(l_preds)]
        right_status = CLASSES[np.argmax(r_preds)]
        
        # --- NEW: Formal Clinical Interpretation (No "AI" mentions) ---
        if primary_disease == "Normal":
            detailed_desc = (f"The diagnostic screening for {p_name} indicates that both fundus images "
                             "exhibit normal physiological characteristics. No significant vascular "
                             "abnormalities or pathological markers were observed in either the OS or OD regions.")
        else:
            detailed_desc = (f"Clinical screening for {p_name} reveals markers consistent with {primary_disease}. "
                             f"The Left Eye (OS) assessment indicates '{left_status}' while the Right Eye (OD) "
                             f"assessment indicates '{right_status}'. The diagnostic system correlates these bilateral "
                             f"features with {primary_disease} at a {conf_val}% statistical probability. "
                             "Clinical correlation via funduscopy and follow-up consultation is recommended.")

        # Save to Database
        conn = get_db()
        try:
            conn.execute('''
                INSERT INTO patient_records 
                (doctor_id, patient_name, patient_age, patient_gender, 
                 left_eye_img, right_eye_img, left_diagnosis, right_diagnosis, overall_findings) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session["user"], p_name, p_age, p_gender, left_eye.filename, right_eye.filename, 
                  left_status, right_status, f"{primary_disease} ({conf_val}%)"))
            conn.commit()
        except sqlite3.IntegrityError:
            flash("Database Error: Record not saved.")
        finally:
            conn.close()

        return render_template(
            "result.html",
            primary_disease=primary_disease,
            confidence=conf_val,
            left_status=left_status,
            right_status=right_status,
            left_img=left_eye.filename,
            right_img=right_eye.filename,
            patient={"name": p_name, "age": p_age, "gender": p_gender},
            description=detailed_desc
        )

    return render_template("upload.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)