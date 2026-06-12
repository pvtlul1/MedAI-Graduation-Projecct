import pandas as pd
import numpy as np
import gradio as gr
import os
import re
import sqlite3
import shutil 
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


BASE_PATH = "/content/drive/MyDrive/MedAI_Final_Oman/"
os.makedirs(BASE_PATH, exist_ok=True)

DOCS_DIR = os.path.join(BASE_PATH, "doctor_documents")
os.makedirs(DOCS_DIR, exist_ok=True)

DB_PATH = BASE_PATH + "medai_secure_v11.db"
HISTORY_CSV = BASE_PATH + "clinical_history.csv"
APPTS_CSV = BASE_PATH + "clinic_appointments.csv"
ADVICE_CSV = BASE_PATH + "physician_prescriptions.csv"

def init_system_storage():
    if not os.path.exists(HISTORY_CSV):
        pd.DataFrame(columns=["Timestamp", "Patient", "Doctor", "Age", "BMI", "Glucose", "BP", "Risk_Result", "Clinical_Notes"]).to_csv(HISTORY_CSV, index=False)
    if not os.path.exists(APPTS_CSV):
        pd.DataFrame(columns=["Patient", "Doctor", "Date", "Time", "Status"]).to_csv(APPTS_CSV, index=False)
    if not os.path.exists(ADVICE_CSV):
        pd.DataFrame(columns=["Date", "Patient", "Doctor", "Medications", "Dietary_Plan", "FollowUp"]).to_csv(ADVICE_CSV, index=False)

    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT)")

    c.execute("CREATE TABLE IF NOT EXISTS doctor_docs (doctor_username TEXT, file_name TEXT, file_path TEXT, upload_date TEXT)")

    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username, password, role) VALUES ('admin', 'admin123', 'Admin')")
    conn.commit(); conn.close()

init_system_storage()


np.random.seed(42); n = 5000
def generate_data(count, risk_factor):
    age = np.random.normal(40 + risk_factor*10, 10, count).clip(21, 80).astype(int)
    bmi = np.random.normal(25 + risk_factor*5, 4, count).clip(18, 45).round(1)
    glu = np.random.normal(95 + risk_factor*40, 20, count).clip(70, 250).astype(int)
    bp = np.random.normal(115 + risk_factor*20, 15, count).clip(80, 190).astype(int)
    gen = np.random.choice([0, 1], size=count)
    fam = np.random.choice([0, 1], size=count, p=[0.8-risk_factor*0.4, 0.2+risk_factor*0.4])
    smk = np.random.choice([0, 1], size=count, p=[0.7-risk_factor*0.3, 0.3+risk_factor*0.3])
    act = np.random.choice([0, 1], size=count, p=[0.3+risk_factor*0.4, 0.7-risk_factor*0.4])
    return pd.DataFrame({'Age': age, 'BMI': bmi, 'Glucose': glu, 'BloodPressure': bp, 'Gender': gen, 'FamilyHistory': fam, 'Smoking': smk, 'PhysicalActivity': act, 'Outcome': [risk_factor]*count})

data_df = pd.concat([generate_data(100, 0), generate_data(100, 1)]).sample(frac=1).reset_index(drop=True)
X = data_df.drop('Outcome', axis=1); y = data_df['Outcome']
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X.columns)])
X_encoded = preprocessor.fit_transform(X)
model = RandomForestClassifier(random_state=42).fit(X_encoded, y)



def upload_doctor_file(doc_username, files):
    if not doc_username or not files:
        return "⚠️ Please provide doctor username and select files.", None

    results = []
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()

    for file in files:
        file_name = os.path.basename(file.name)
        dest_path = os.path.join(DOCS_DIR, f"{doc_username}_{file_name}")
        shutil.copy(file.name, dest_path)

        c.execute("INSERT INTO doctor_docs (doctor_username, file_name, file_path, upload_date) VALUES (?, ?, ?, ?)",
                  (doc_username.strip().lower(), file_name, dest_path, datetime.now().strftime('%Y-%m-%d %H:%M')))

    conn.commit()
    c.execute("SELECT doctor_username, file_name, upload_date FROM doctor_docs")
    data = c.fetchall()
    conn.close()

    return f"✅ Successfully uploaded {len(files)} files for Dr. {doc_username}", pd.DataFrame(data, columns=["Doctor", "File Name", "Date"])

def get_all_docs():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT doctor_username, file_name, upload_date FROM doctor_docs")
    data = c.fetchall()
    conn.close()
    return pd.DataFrame(data, columns=["Doctor", "File Name", "Date"])
def validate_password_complexity(password):
    if len(password) < 8: return False
    if not re.search(r"[A-Z]", password): return False
    if not re.search(r"[a-z]", password): return False
    if not re.search(r"[0-9]", password): return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password): return False
    return True
def login_logic(u, p, expected_role):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username=? AND password=?", (u.strip().lower(), p.strip()))
    res = c.fetchone(); conn.close()
    if res:
        if res[0] == expected_role or (expected_role == "Admin" and res[0] == "Admin"):
            return "Success", u.strip().lower(), res[0]
        else: return f"Error: You are registered as a {res[0]}.", None, None
    return "Error: Invalid Credentials", None, None

def register_user(u, p):
    if not validate_password_complexity(p):
        return "❌ Password Rejected! Must be at least 8 characters long, containing uppercase & lowercase letters, numbers, and special characters (e.g., @, #, $)."
        
    try:
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, 'Patient')", (u.strip().lower(), p.strip()))
        conn.commit(); conn.close(); return "Registration Successful. Please Login."
    except: return "Username already exists."

def add_doc_user(u, p):
    try:
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, 'Doctor')", (u.strip().lower(), p.strip()))
        conn.commit(); conn.close(); return f"Doctor {u} authorized successfully."
    except: return "Error: Username exists."

def patient_analysis_logic(age, bmi, glu, bp, gen, fam, smk, act, p_name):
    g_val = 1 if gen=="Male" else 0; f_val = 1 if fam=="Yes" else 0
    s_val = 1 if smk=="Yes" else 0; a_val = 1 if act=="Active" else 0
    input_data = pd.DataFrame([[age, bmi, glu, bp, g_val, f_val, s_val, a_val]], columns=X.columns)
    prob = model.predict_proba(preprocessor.transform(input_data))[0][1] * 100
    pd.DataFrame([[datetime.now().strftime('%Y-%m-%d %H:%M'), p_name, "Self", age, bmi, glu, bp, f"{prob:.1f}%", "N/A"]]).to_csv(HISTORY_CSV, mode='a', header=False, index=False)
    reasons = []
    if bmi > 25: reasons.append("🔻 Your BMI indicates you are above the optimal weight range, which affects metabolism.")
    if glu > 100: reasons.append("🔻 Your fasting glucose is elevated, a primary indicator requiring attention.")
    if bp > 130: reasons.append("🔻 Your blood pressure is slightly elevated, adding stress to cardiovascular functions.")
    if smk == "Yes": reasons.append("🔻 Smoking significantly increases metabolic and vascular risks.")
    if fam == "Yes": reasons.append("🔻 A family history indicates a genetic predisposition, requiring proactive care.")
    reason_text = "\n".join(reasons) if reasons else "✨ All your vital parameters are currently within healthy ranges."
    tips = "\n========================================\n🌱 DAILY WELLNESS & LIFESTYLE GUIDE:\n========================================\n1. 🥗 Nutrition: Focus on high-fiber foods, whole grains, and lean proteins.\n2. 🏃‍♂️ Activity: Aim for at least 150 minutes of moderate exercise per week.\n3. 💧 Hydration: Drink 2-3 liters of water daily.\n4. 😴 Sleep: Ensure 7-8 hours of quality sleep."
    if prob >= 66: return f"🚨 STATUS: HIGH RISK ({int(prob)}%)\n\nDear {str(p_name).capitalize()}, please remain calm. This tool is for early awareness, not a final diagnosis. We highly recommend scheduling a consultation with a specialist soon.\n\n🔍 WHY IS YOUR RISK ELEVATED?\n{reason_text}\n{tips}"
    elif prob >= 33: return f"⚠️ STATUS: MODERATE RISK ({int(prob)}%)\n\nDear {str(p_name).capitalize()}, you are in a great position to make small changes that will have a massive positive impact on your future health.\n\n🔍 AREAS FOR IMPROVEMENT:\n{reason_text}\n{tips}"
    else: return f"✅ STATUS: NORMAL/LOW RISK ({int(prob)}%)\n\nExcellent work, {str(p_name).capitalize()}! Your lifestyle choices are reflecting positively on your health indicators. Keep maintaining this healthy balance.\n{tips}"

def doctor_clinical_logic(age, bmi, glu, bp, gen, fam, smk, act, p_name, d_name, clinical_notes):
    g_val = 1 if gen=="Male" else 0; f_val = 1 if fam=="Yes" else 0
    s_val = 1 if smk=="Yes" else 0; a_val = 1 if act=="Active" else 0
    input_data = pd.DataFrame([[age, bmi, glu, bp, g_val, f_val, s_val, a_val]], columns=X.columns)
    prob = model.predict_proba(preprocessor.transform(input_data))[0][1] * 100
    risk_level = "CRITICAL" if prob >= 70 else "ELEVATED" if prob >= 40 else "STABLE"
    med_advice = "1. Initiate Metformin 500mg daily.\n   2. Consider dual therapy if targets are not met.\n   3. Intensive lifestyle intervention." if prob >= 66 else "1. Pre-diabetes protocol: No immediate pharmacological intervention required.\n   2. Prescribe structured dietary plan." if prob >= 33 else "1. Routine preventive care.\n   2. No medication required at this stage."
    smoking_alert = "\n⚠️ URGENT CLINICAL ALERT: Patient is an active smoker. Immediate smoking cessation counseling required." if smk == "Yes" else ""
    pd.DataFrame([[datetime.now().strftime('%Y-%m-%d %H:%M'), p_name, d_name, age, bmi, glu, bp, f"{risk_level} ({prob:.1f}%)", clinical_notes]]).to_csv(HISTORY_CSV, mode='a', header=False, index=False)
    return f"====================================================\n🏥 OFFICIAL CLINICAL DIAGNOSTIC REPORT\n====================================================\nAttending Physician : Dr. {str(d_name).upper()}\nPatient Identifier  : {str(p_name).upper()}\nDate of Assessment  : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n----------------------------------------------------\n🩸 DIAGNOSTIC OUTCOME:\nRisk Level          : {risk_level}\nCalculated Prob.    : {prob:.1f}% {smoking_alert}\n\n💊 SUGGESTED ACTION PLAN & MEDICATIONS:\n{med_advice}\n\n🔬 RECOMMENDED LABORATORY TESTS:\n[ ] HbA1c (Glycated Hemoglobin)\n[ ] Fasting Lipid Panel (Triglycerides, HDL, LDL)\n[ ] Comprehensive Metabolic Panel (CMP)\n\n📝 PHYSICIAN'S OBSERVATIONS:\n{clinical_notes if clinical_notes else 'No additional remarks.'}\n===================================================="

def book_appointment(p_name, d_name, date, time):
    if not date or not time or not d_name: return "⚠️ Error: Please complete all fields."
    pd.DataFrame([[p_name, d_name, date, time, "Confirmed"]]).to_csv(APPTS_CSV, mode='a', header=False, index=False)
    return f"✅ Appointment successfully booked with Dr. {d_name.capitalize()} on {date} at {time}."

upcoming_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 15)]
clinic_times = ["08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM", "01:00 PM", "02:00 PM", "03:00 PM", "04:00 PM"]


interactive_css = """
/* [كود CSS الأصلي نفسه دون تغيير] */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@500;600;700;800&display=swap');
body, html, * { font-family: 'Inter', sans-serif !important; }
h1, h2, h3, h4, h5, h6 { font-family: 'Poppins', sans-serif !important; font-weight: 700 !important; letter-spacing: -0.5px !important; color: #0f172a !important; }
.welcome-title { font-size: 4.5em !important; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0px !important; line-height: 1.2 !important; }
.welcome-subtitle { color: #475569 !important; margin-top: 5px !important; font-weight: 600 !important; font-size: 1.8em !important; }
.panel-title { font-family: 'Poppins', sans-serif !important; font-size: 1.8em !important; font-weight: 700 !important; color: #0f172a !important; margin: 0 0 15px 0 !important; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }
@keyframes fadeSlideUp { 0% { opacity: 0; transform: translateY(25px); } 100% { opacity: 1; transform: translateY(0); } }
.page-transition { animation: fadeSlideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards !important; }
body { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }
.primary-btn { background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important; border: none !important; color: white !important; font-family: 'Poppins', sans-serif !important; font-weight: 600 !important; letter-spacing: 1px !important; box-shadow: 0 4px 15px rgba(30, 58, 138, 0.3) !important; transition: all 0.3s ease-in-out !important; }
.primary-btn:hover { transform: translateY(-3px) scale(1.02) !important; box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5) !important; }
.secondary-btn { background-color: #ffffff !important; color: #1e3a8a !important; font-family: 'Poppins', sans-serif !important; font-weight: 600 !important; border: 2px solid #cbd5e1 !important; transition: all 0.3s ease-in-out !important; }
.secondary-btn:hover { border-color: #3b82f6 !important; color: #3b82f6 !important; transform: translateY(-2px) !important; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15) !important; }
.gr-panel { border-radius: 16px !important; background-color: #ffffff !important; border: 1px solid #e2e8f0 !important; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05) !important; transition: all 0.4s ease !important; }
input, textarea, select { border-radius: 8px !important; transition: all 0.3s ease !important; font-family: 'Inter', sans-serif !important; }
"""

modern_theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo", neutral_hue="slate")

with gr.Blocks(theme=modern_theme, css=interactive_css, title="Diabetes Prediction System") as app:
    user_state = gr.State(None); role_state = gr.State(None)

    with gr.Column(visible=True, elem_classes=["page-transition"]) as landing_sec:
        gr.HTML("<center><h1 class='welcome-title'>MedAI</h1></center>")
        gr.HTML("<center><h3 class='welcome-subtitle'>Diabetes Prediction System</h3></center>")
        gr.HTML("<br><br>")
        with gr.Row():
            btn_go_patient = gr.Button("👤 I AM A PATIENT", elem_classes=["primary-btn"], size="lg")
            btn_go_doctor = gr.Button("🩺 I AM A DOCTOR", elem_classes=["secondary-btn"], size="lg")
        gr.HTML("<br><br><br>")
        with gr.Row():
            btn_go_admin = gr.Button("⚙️ system administration", elem_classes=["secondary-btn"], size="sm")

    with gr.Column(visible=False, elem_classes=["gr-panel", "page-transition"]) as auth_patient_sec:
        gr.HTML("<h2 class='panel-title'>👤 Patient Portal Access</h2>")
        btn_back_p = gr.Button("← Back to Home", size="sm", elem_classes=["secondary-btn"])
        with gr.Tabs():
            with gr.Tab("Login"):
                p_l_u = gr.Textbox(label="Username"); p_l_p = gr.Textbox(label="Password", type="password")
                p_l_btn = gr.Button("Secure Login", elem_classes=["primary-btn"]); p_l_res = gr.Textbox(label="Status")
            with gr.Tab("Register"):
                r_u = gr.Textbox(label="Choose Username"); r_p = gr.Textbox(label="Password", type="password")
                r_btn = gr.Button("Register Account", elem_classes=["primary-btn"]); r_res = gr.Textbox(label="Status")

    with gr.Column(visible=False, elem_classes=["gr-panel", "page-transition"]) as auth_doctor_sec:
        gr.HTML("<h2 class='panel-title'>🩺 Doctor Clinical Access</h2>")
        btn_back_d = gr.Button("← Back to Home", size="sm", elem_classes=["secondary-btn"])
        d_l_u = gr.Textbox(label="Medical Username"); d_l_p = gr.Textbox(label="Password", type="password")
        d_l_btn = gr.Button("Secure Medical Login", elem_classes=["primary-btn"]); d_l_res = gr.Textbox(label="Status")

    with gr.Column(visible=False, elem_classes=["gr-panel", "page-transition"]) as auth_admin_sec:
        gr.HTML("<h2 class='panel-title'>⚙️ Administration Access</h2>")
        btn_back_a = gr.Button("← Back to Home", size="sm", elem_classes=["secondary-btn"])
        a_l_u = gr.Textbox(label="Admin ID"); a_l_p = gr.Textbox(label="Password", type="password")
        a_l_btn = gr.Button("Admin Login", elem_classes=["primary-btn"]); a_l_res = gr.Textbox(label="Status")

    with gr.Column(visible=False, elem_classes=["gr-panel", "page-transition"]) as admin_dashboard:
        gr.HTML("<h2 class='panel-title'>🛠️ Administrative Control Center</h2>")
        logout_a = gr.Button("Secure Logout", size="sm", elem_classes=["secondary-btn"])

        with gr.Tabs():
            with gr.Tab("User Management"):
                with gr.Row():
                    a_doc_u = gr.Textbox(label="New Physician Username"); a_doc_p = gr.Textbox(label="System Password")
                    a_doc_btn = gr.Button("Authorize Physician", elem_classes=["primary-btn"])
                a_res = gr.Textbox(label="Operation Result")

            with gr.Tab("Doctor Verification Documents"): 
                gr.Markdown("### 📂 Upload Doctor IDs & Certificates")
                with gr.Row():
                    doc_id_input = gr.Textbox(label="Target Doctor Username (Must be registered)")
                    doc_files = gr.File(label="Upload Certificates/ID Cards", file_count="multiple")
                upload_btn = gr.Button("Save Documents to System", elem_classes=["primary-btn"])
                upload_status = gr.Textbox(label="Upload Status")
                gr.Markdown("### 📋 Archival Record of Uploaded Documents")
                docs_table = gr.DataFrame(value=get_all_docs(), interactive=False)

    with gr.Column(visible=False, elem_classes=["gr-panel", "page-transition"]) as patient_dashboard:
        gr.HTML("<h2 class='panel-title'>👤 Personal Wellness Portal</h2>")
        logout_p = gr.Button("Secure Logout", size="sm", variant="stop")
        with gr.Tabs():
            with gr.Tab("AI Health Check"):
                with gr.Row():
                    p_age = gr.Slider(21, 80, label="Age"); p_bmi = gr.Slider(15, 50, label="BMI")
                    p_glu = gr.Slider(70, 250, label="Fasting Glucose Level"); p_bp = gr.Slider(80, 180, label="Systolic Blood Pressure")
                with gr.Row():
                    p_gen = gr.Radio(["Male", "Female"], label="Gender", value="Female"); p_fam = gr.Radio(["Yes", "No"], label="Family History of Diabetes", value="No")
                    p_smk = gr.Radio(["Yes", "No"], label="Active Smoker", value="No"); p_act = gr.Radio(["Active", "Inactive"], label="Physical Activity Status", value="Active")
                p_btn = gr.Button("Analyze My Health", elem_classes=["primary-btn"])
                p_predict_res = gr.Textbox(label="Comprehensive Health Report", lines=15)
            with gr.Tab("Book Medical Appointment"):
                with gr.Row():
                    ap_doc = gr.Textbox(label="Target Doctor Name")
                    ap_date = gr.Dropdown(choices=upcoming_dates, label="Select Available Date")
                    ap_time = gr.Dropdown(choices=clinic_times, label="Select Time Slot")
                ap_btn = gr.Button("Confirm Appointment", elem_classes=["primary-btn"])
                ap_res = gr.Textbox(label="Booking Status")

    with gr.Column(visible=False, elem_classes=["gr-panel", "page-transition"]) as doctor_dashboard:
        gr.HTML("<h2 class='panel-title'>🩺 Clinical Diagnostic Station</h2>")
        logout_d = gr.Button("Secure Logout", size="sm", variant="stop")
        with gr.Tabs():
            with gr.Tab("Professional Diagnosis"):
                d_pat_id = gr.Textbox(label="Patient ID / Name")
                with gr.Row():
                    d_age = gr.Number(label="Patient Age"); d_bmi = gr.Number(label="Patient BMI")
                    d_glu = gr.Number(label="Glucose (Fasting)"); d_bp = gr.Number(label="Systolic BP")
                with gr.Row():
                    d_gen = gr.Radio(["Male", "Female"], label="Gender", value="Male"); d_fam = gr.Radio(["Yes", "No"], label="Family History", value="No")
                    d_smk = gr.Radio(["Yes", "No"], label="Smoking Status", value="No"); d_act = gr.Radio(["Active", "Inactive"], label="Activity", value="Active")
                d_notes = gr.Textbox(label="Clinical Observations (Optional)", lines=2)
                d_btn = gr.Button("Generate & Save Official Report", elem_classes=["primary-btn"])
                d_predict_res = gr.Textbox(label="Advanced Clinical Report Output", lines=20)

    btn_go_patient.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None, [landing_sec, auth_patient_sec])
    btn_go_doctor.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None, [landing_sec, auth_doctor_sec])
    btn_go_admin.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None, [landing_sec, auth_admin_sec])

    back_to_home = lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    btn_back_p.click(back_to_home, None, [landing_sec, auth_patient_sec, auth_doctor_sec, auth_admin_sec])
    btn_back_d.click(back_to_home, None, [landing_sec, auth_patient_sec, auth_doctor_sec, auth_admin_sec])
    btn_back_a.click(back_to_home, None, [landing_sec, auth_patient_sec, auth_doctor_sec, auth_admin_sec])

    def route_after_login(role):
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=role=="Admin"), gr.update(visible=role=="Patient"), gr.update(visible=role=="Doctor"))

    p_l_btn.click(lambda u, p: login_logic(u, p, "Patient"), [p_l_u, p_l_p], [p_l_res, user_state, role_state]).then(route_after_login, role_state, [auth_patient_sec, auth_doctor_sec, auth_admin_sec, admin_dashboard, patient_dashboard, doctor_dashboard])
    d_l_btn.click(lambda u, p: login_logic(u, p, "Doctor"), [d_l_u, d_l_p], [d_l_res, user_state, role_state]).then(route_after_login, role_state, [auth_patient_sec, auth_doctor_sec, auth_admin_sec, admin_dashboard, patient_dashboard, doctor_dashboard])
    a_l_btn.click(lambda u, p: login_logic(u, p, "Admin"), [a_l_u, a_l_p], [a_l_res, user_state, role_state]).then(route_after_login, role_state, [auth_patient_sec, auth_doctor_sec, auth_admin_sec, admin_dashboard, patient_dashboard, doctor_dashboard])

    r_btn.click(register_user, [r_u, r_p], r_res)
    a_doc_btn.click(add_doc_user, [a_doc_u, a_doc_p], a_res)

    upload_btn.click(upload_doctor_file, [doc_id_input, doc_files], [upload_status, docs_table])

    p_btn.click(patient_analysis_logic, [p_age, p_bmi, p_glu, p_bp, p_gen, p_fam, p_smk, p_act, user_state], p_predict_res)
    d_btn.click(doctor_clinical_logic, [d_age, d_bmi, d_glu, d_bp, d_gen, d_fam, d_smk, d_act, d_pat_id, user_state, d_notes], d_predict_res)
    ap_btn.click(book_appointment, [user_state, ap_doc, ap_date, ap_time], ap_res)

    logout_func = lambda: (None, None, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    logout_a.click(logout_func, None, [user_state, role_state, landing_sec, admin_dashboard, patient_dashboard, doctor_dashboard])
    logout_p.click(logout_func, None, [user_state, role_state, landing_sec, admin_dashboard, patient_dashboard, doctor_dashboard])
    logout_d.click(logout_func, None, [user_state, role_state, landing_sec, admin_dashboard, patient_dashboard, doctor_dashboard])

app.launch(debug=True, share=True)