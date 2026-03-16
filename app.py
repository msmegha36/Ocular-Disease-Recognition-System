from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import numpy as np
import tensorflow as tf
from utils.preprocess import preprocess_image
from utils.heatmap import get_gradcam_heatmap, save_and_display_gradcam
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
        email = request.form.get("username")
        password = request.form.get("password")
        hashed_pw = hash_password(password)
        
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=? AND password=?", (email, hashed_pw)).fetchone()
        conn.close()
        
        if user:
            session["user"] = user['id']
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password.", "danger")
            
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = hash_password(request.form.get("password"))
        spec = request.form.get("specialization")
        hosp = request.form.get("hospital")

        conn = get_db()
        try:
            conn.execute("""
                INSERT INTO users (name, email, password, specialization, hospital) 
                VALUES (?, ?, ?, ?, ?)
            """, (name, email, password, spec, hosp))
            conn.commit()
            flash("Account successfully created! You can now log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Registration failed: This email is already in use.", "danger")
        finally:
            conn.close()
            
    return render_template("register.html")

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
        # 1. Collect Form Data
        p_name = request.form.get("patient_name")
        p_age = request.form.get("patient_age")
        p_gender = request.form.get("patient_gender")
        patient_id = request.form.get("patient_id") # Ensure this is in your form

        left_eye = request.files["left_eye"]
        right_eye = request.files["right_eye"]
        
        # 2. Save Original Images
        l_path = os.path.join(UPLOAD_FOLDER, left_eye.filename)
        r_path = os.path.join(UPLOAD_FOLDER, right_eye.filename)
        left_eye.save(l_path)
        right_eye.save(r_path)

        # 3. Preprocess for Model (512x512, normalized)
        l_img_raw, _ = preprocess_image(l_path)
        r_img_raw, _ = preprocess_image(r_path)
        
        # Add batch dimension: (1, 512, 512, 3)
        l_input = np.expand_dims(l_img_raw, 0)
        r_input = np.expand_dims(r_img_raw, 0)

        # 4. Define 8 ODIR Classes
        CLASSES = ["Normal", "Diabetes", "Glaucoma", "Cataract", "AMD", "Hypertension", "Myopia", "Others"]

        # 5. Three-Pass Prediction (Independent vs. Joint)
        l_preds = model.predict([l_input, l_input])[0]   # Left vs Left
        r_preds = model.predict([r_input, r_input])[0]   # Right vs Right
        joint_preds = model.predict([l_input, r_input])[0] # Joint Analysis

        # Extract Statuses
        left_status = CLASSES[np.argmax(l_preds)]
        right_status = CLASSES[np.argmax(r_preds)]
        
        top_idx = np.argmax(joint_preds)
        primary_disease = CLASSES[top_idx]
        conf_val = round(float(joint_preds[top_idx]) * 100, 2)

        # 6. Generate Heatmaps (Grad-CAM)
        left_heatmap_path = None
        right_heatmap_path = None
        
        try:
            from utils.heatmap import get_gradcam_heatmap, save_and_display_gradcam
            
            # Heatmap for Left Eye (using joint context)
            h_data_l = get_gradcam_heatmap(l_input, r_input, model, target_side='left')
            left_heatmap_path = save_and_display_gradcam(l_path, h_data_l, 'left')

            # Heatmap for Right Eye (using joint context)
            h_data_r = get_gradcam_heatmap(l_input, r_input, model, target_side='right')
            right_heatmap_path = save_and_display_gradcam(r_path, h_data_r, 'right')
        except Exception as e:
            print(f"Heatmap Error: {e}")

        # 7. Clinical Interpretation String
        detailed_desc = (f"Clinical screening for {p_name} reveals markers consistent with {primary_disease}. "
                         f"Left Eye (OS): {left_status}. Right Eye (OD): {right_status}. "
                         f"Bilateral feature correlation probability: {conf_val}%.")

        # 8. Save to Database
        conn = get_db()
        try:
            conn.execute('''
                INSERT INTO patient_records 
                (doctor_id, patient_name, patient_age, patient_gender, 
                 left_eye_img, right_eye_img, left_diagnosis, right_diagnosis, 
                 overall_findings, left_heatmap, right_heatmap) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session["user"], p_name, p_age, p_gender, left_eye.filename, right_eye.filename, 
                  left_status, right_status, f"{primary_disease} ({conf_val}%)", 
                  left_heatmap_path, right_heatmap_path))
            conn.commit()
        except Exception as e:
            print(f"DB Error: {e}")
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
            left_heatmap=left_heatmap_path,
            right_heatmap=right_heatmap_path,
            patient={"name": p_name, "age": p_age, "gender": p_gender},
            description=detailed_desc
        )

    return render_template("upload.html")

@app.route("/history")
def history():
    if "user" not in session: return redirect(url_for("login"))
    conn = get_db()
    
    # Fetch all records joined with patient names
    all_records = conn.execute('''
        SELECT pr.*, p.name as patient_name, p.age, p.gender 
        FROM patient_records pr 
        JOIN patients p ON pr.patient_id = p.patient_id 
        WHERE pr.doctor_id = ? 
        ORDER BY p.name ASC, pr.timestamp DESC
    ''', (session["user"],)).fetchall()
    conn.close()

    # Grouping logic: Create a dictionary where key is patient_id
    grouped_history = {}
    for row in all_records:
        p_id = row['patient_id']
        if p_id not in grouped_history:
            grouped_history[p_id] = {
                'name': row['patient_name'],
                'patient_id': p_id,
                'scans': []
            }
        grouped_history[p_id]['scans'].append(row)

    return render_template("history.html", grouped_history=grouped_history)
    
@app.route("/view_result/<int:record_id>")
def view_result(record_id):
    conn = get_db()
    # 1. Fetch the record and JOIN with patients to get the name
    record = conn.execute('''
        SELECT pr.*, p.name, p.age, p.gender FROM patient_records pr 
        JOIN patients p ON pr.patient_id = p.patient_id WHERE pr.id = ?
    ''', (record_id,)).fetchone()
    conn.close()
    
    if not record:
        return "Record not found", 404

    # 2. Extract values from the 'record' dictionary
    # We must use the 'record' variable because 'primary_disease' doesn't exist here
    findings = record['overall_findings']
    primary = findings.split(' (')[0] if '(' in findings else findings
    conf = findings.split('(')[1].replace('%)', '') if '(' in findings else "0"
    
    # 3. Reconstruct the Professional Paragraph logic
    p_name = record['name']
    l_status = record['left_diagnosis']
    r_status = record['right_diagnosis']
    
    description = (
        f"Clinical screening for {p_name} reveals markers consistent with {primary}. "
        f"The Left Eye (OS) assessment indicates '{l_status}' while the Right Eye (OD) assessment indicates '{r_status}'. "
        f"The diagnostic system correlates these bilateral features with {primary} at a {conf}% statistical probability. "
        f"Clinical correlation via funduscopy and follow-up consultation is recommended."
    )

    # 4. Return the template using the record's data
    return render_template("result.html", 
                           primary_disease=primary, 
                           confidence=conf,
                           left_img=record['left_eye_img'], 
                           right_img=record['right_eye_img'],
                           left_status=l_status, 
                           right_status=r_status,
                           left_heatmap=record['left_heatmap'],
                           right_heatmap=record['right_heatmap'],
                           patient=record, 
                           description=description)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)