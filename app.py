from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import numpy as np
import tensorflow as tf
from utils.preprocess import preprocess_image
from utils.heatmap import generate_siamese_heatmap  # Updated name
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

@app.route("/")
def index():
    # This renders the new homepage code you just pasted
    return render_template('home.html')

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

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("username")
        password = request.form.get("password")
        hashed_pw = hash_password(password)
        
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=? AND password=?", (email, hashed_pw)).fetchone()
        conn.close()
        
        if user:
            # FIX 1: Clear the session queue before redirecting to Dashboard
            # This ensures "Account Created" or "Invalid" messages don't follow you in
            session.pop('_flashes', None) 
            
            session["user"] = user['id']
            return redirect(url_for("dashboard"))
        else:
            # FIX 2: Clear old flashes before adding a new "Invalid" one
            # This prevents multiple "Invalid" messages from stacking up
            session.pop('_flashes', None)
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
            
            # FIX 3: Clear any potential errors before adding the success message
            session.pop('_flashes', None)
            flash("Account successfully created! You can now log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            session.pop('_flashes', None)
            flash("Registration failed: This email is already in use.", "danger")
        finally:
            conn.close()
            
    return render_template("register.html")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
         return redirect(url_for("login"))
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

@app.route("/register-patient", methods=["GET", "POST"])
def register_patient():
    if "user" not in session:
        return redirect(url_for("login"))

    success = False  # Track if registration worked
    if request.method == "POST":
        p_name = request.form.get("name")
        p_age = request.form.get("age")
        p_gender = request.form.get("gender")
        p_phone = request.form.get("phone")

        conn = get_db()
        try:
            conn.execute('''
                INSERT INTO patients (name, age, gender, phone) 
                VALUES (?, ?, ?, ?)
            ''', (p_name, p_age, p_gender, p_phone))
            conn.commit()
            
            flash(f"Patient Record for {p_name} created successfully! Redirecting...", "success")
            success = True 
        except Exception as e:
            print(f"Database Error: {e}")
            flash("Error: Could not register patient.", "danger")
        finally:
            conn.close()

    # Pass the success variable to the template
    return render_template("register_patient.html", success=success)

@app.route("/patients")
def list_patients():
    if "user" not in session: 
        return redirect(url_for("login"))
    
    conn = get_db()
    # Fetching in descending order of patient_id or created_at
    patients = conn.execute('''
        SELECT patient_id,name,age,gender FROM patients 
        ORDER BY created_at DESC
    ''').fetchall()
    conn.close()
    
    return render_template("patients.html", patients=patients)    

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "user" not in session: 
        return redirect(url_for("login"))

    if request.method == "POST":
        # 1. GET DATA FROM FORM FIRST
        patient_id = request.form.get("patient_id") 
        
        # Pull files from the request
        # This defines 'left_eye' and 'right_eye' so the NameError disappears
        left_eye = request.files.get("left_eye")
        right_eye = request.files.get("right_eye")

        # Basic safety check
        if not left_eye or not right_eye:
            return "Missing images", 400

        # 2. RE-FETCH PATIENT DETAILS
        # This prevents the "None" values in the final report
        conn = get_db()
        patient_data = conn.execute(
            "SELECT name, age, gender FROM patients WHERE patient_id = ?", 
            (patient_id,)
        ).fetchone()
        
        # Provide fallbacks if patient not found
        p_name = patient_data["name"] if patient_data else "Unknown"
        p_age = patient_data["age"] if patient_data else "N/A"
        p_gender = patient_data["gender"] if patient_data else "N/A"

        # 3. SAVE ORIGINAL IMAGES
        # Using .filename is safe now because left_eye is defined
        l_path = os.path.join(UPLOAD_FOLDER, left_eye.filename)
        r_path = os.path.join(UPLOAD_FOLDER, right_eye.filename)
        left_eye.save(l_path)
        right_eye.save(r_path)

        # 4. PREPROCESS & PREDICT
        l_img_raw, _ = preprocess_image(l_path)
        r_img_raw, _ = preprocess_image(r_path)
        
        l_input = np.expand_dims(l_img_raw, 0)
        r_input = np.expand_dims(r_img_raw, 0)

        CLASSES = ["Normal", "Diabetes", "Glaucoma", "Cataract", "AMD", "Hypertension", "Myopia", "Others"]

        # Siamese Three-Pass Prediction
        l_preds = model.predict([l_input, l_input])[0]
        r_preds = model.predict([r_input, r_input])[0]
        joint_preds = model.predict([l_input, r_input])[0]

        left_status = CLASSES[np.argmax(l_preds)]
        right_status = CLASSES[np.argmax(r_preds)]
        
        top_idx = np.argmax(joint_preds)
        primary_disease = CLASSES[top_idx]
        conf_val = round(float(joint_preds[top_idx]) * 100, 2)

        # 5. GENERATE HEATMAPS
        left_heatmap_path = None
        right_heatmap_path = None
         # In your app.py upload route:
        try:
             # Pass the left image into BOTH inputs of the Siamese model.
             # This effectively "mutes" the right eye influence for this specific heatmap.
             l_h_raw = generate_siamese_heatmap(l_path, model, target_side='left') 
    
             # Do the same for the right side
             r_h_raw = generate_siamese_heatmap(r_path, model, target_side='right')

             # Clean paths as before
             left_heatmap_path = l_h_raw.replace('static/', '').replace('\\', '/') if l_h_raw else None
             right_heatmap_path = r_h_raw.replace('static/', '').replace('\\', '/') if r_h_raw else None
        except Exception as e:
             print(f"Heatmap Error: {e}")

        # 6. FINAL INTERPRETATION & DB SAVE
        detailed_desc = (f"Clinical screening for {p_name} reveals markers consistent with {primary_disease}. "
                         f"Left Eye (OS): {left_status}. Right Eye (OD): {right_status}. "
                         f"Bilateral feature correlation probability: {conf_val}%.")
        
        patient_id_val = request.form.get("patient_id")

        try:  # Align this try block with patient_id_val
            conn.execute('''
                INSERT INTO patient_records 
                (doctor_id, patient_id, left_eye_img, right_eye_img, left_diagnosis, 
                 right_diagnosis, overall_findings, left_heatmap, right_heatmap) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session["user"], 
                patient_id_val, 
                left_eye.filename, 
                right_eye.filename, 
                left_status, 
                right_status, 
                f"{primary_disease} ({conf_val}%)", 
                left_heatmap_path, 
                right_heatmap_path
            ))
            conn.commit()
        except Exception as e:
            print(f"DB Error: {e}")
        finally:
            conn.close()

        # 7. RENDER RESULTS
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
    if "user" not in session:
        return redirect(url_for("login"))
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


@app.route('/logout')
def logout():
    session.clear()  # Removes everything from the session
    return redirect(url_for('login'))

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, post-check=0, pre-check=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "-1" # Setting to -1 or 0 tells the browser it's already expired
    return response

if __name__ == "__main__":
    app.run(debug=True)