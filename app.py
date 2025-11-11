# app.py — NeuroVR Lab v3.5 (full)
# AI + VR + Drug Discovery + Research Dashboard + Patient Recorder (session-only)
# System by Simon | allinmer57@gmail.com

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib, shap, base64, io, os, matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from streamlit.components.v1 import html

# --------------------------- CONFIG ---------------------------
st.set_page_config(
    page_title=" NeuroVR Lab — VR Stroke Research & Drug Discovery",
    page_icon="🧠",
    layout="wide"
)

# --------------------------- SESSION STATE SETUP ---------------------------
if "patient_records" not in st.session_state:
    st.session_state.patient_records = pd.DataFrame(
        columns=["id", "age", "bp", "cholesterol", "glucose", "bmi", "stroke", "timestamp"]
    )

if "compounds" not in st.session_state:
    st.session_state.compounds = []  # store evaluated compounds in-session

# fixed home notes (your preferred text)
if "notes" not in st.session_state:
    st.session_state.notes = (
        "🧠 **NeuroVR Lab: AI-Powered Stroke & Drug Discovery Platform**\n\n"
        "Condition: Stroke (Cerebrovascular Accident) is a major neurological disorder caused by interruption "
        "of blood flow to the brain, leading to tissue damage.\n\n"
        "Goal: NeuroVR Lab combines AI prediction, patient data simulation, and virtual reality visualization "
        "to enhance understanding of stroke patterns and aid discovery of potential therapeutic compounds.\n\n"
        "Impact: This system offers an experimental framework for early risk detection, drug interaction testing, "
        "and educational visualization."
    )

# --------------------------- HELPERS ---------------------------
def train_model(dataframe):
    # Expect dataframe with columns: age,bp,cholesterol,glucose,bmi,stroke
    X = dataframe[["age", "bp", "cholesterol", "glucose", "bmi"]]
    y = dataframe["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_local = RandomForestClassifier(n_estimators=100, random_state=42)
    model_local.fit(X_train, y_train)
    acc_local = accuracy_score(y_test, model_local.predict(X_test))
    return model_local, acc_local

def safe_shap_plot(model, X):
    """
    Compute SHAP values and return a matplotlib Figure and the shap array
    normalized into a 2D array (samples x features). This avoids shape issues
    that lead to 'Per-column arrays must each be 1-dimensional' errors.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values can be a list [neg, pos]
    shap_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap_to_plot = np.array(shap_to_plot)

    # If shap_to_plot has extra dimensions collapse them
    if shap_to_plot.ndim > 2:
        shap_to_plot = shap_to_plot.reshape(shap_to_plot.shape[0], -1)

    # Make sure shap columns align with X columns
    if shap_to_plot.shape[1] != X.shape[1]:
        # Trim or pad with zeros to match feature count
        if shap_to_plot.shape[1] > X.shape[1]:
            shap_to_plot = shap_to_plot[:, :X.shape[1]]
        else:
            pad_width = X.shape[1] - shap_to_plot.shape[1]
            shap_to_plot = np.pad(shap_to_plot, ((0,0),(0,pad_width)), mode="constant", constant_values=0.0)

    # Draw summary_plot onto a matplotlib figure
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_to_plot, X, show=False)
    fig = plt.gcf()
    plt.close(fig)
    return fig, shap_to_plot

def add_patient_record(age, bp, chol, glucose, bmi, stroke):
    df = st.session_state.patient_records
    new_id = int(df["id"].max()) + 1 if not df.empty else 1
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = {"id": new_id, "age": age, "bp": bp, "cholesterol": chol,
               "glucose": glucose, "bmi": bmi, "stroke": stroke, "timestamp": ts}
    st.session_state.patient_records = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

def delete_patient_record(record_id):
    st.session_state.patient_records = st.session_state.patient_records[st.session_state.patient_records["id"] != record_id].reset_index(drop=True)

# --------------------------- LOAD SYNTHETIC DATA & INITIAL MODEL ---------------------------
@st.cache_data
def load_synthetic():
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.randint(30, 85, 200),
        "bp": np.random.randint(110, 190, 200),
        "cholesterol": np.random.randint(100, 300, 200),
        "glucose": np.random.randint(60, 200, 200),
        "bmi": np.random.uniform(18, 35, 200),
        "stroke": np.random.choice([0, 1], 200, p=[0.75, 0.25])
    })

data = load_synthetic()
model, acc = train_model(data)

# --------------------------- SIDEBAR NAVIGATION ---------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", [
    "Home 🏠",
    "Research Dashboard 📊",
    "AI Stroke Predictor 🤖",
    "Patient Data Recorder 🧾",
    "Drug Discovery Lab 💊",
    "VR Headset Mode 🕶️",
    "Upload/Train Dataset 📂",
    "Research Report 🧾"
])

# --------------------------- HOME ---------------------------
if page == "Home 🏠":
    st.markdown("<h2 style='color:#1f77b4;'>🧬 NeuroVR Lab — VR Stroke Research & Drug Discovery</h2>", unsafe_allow_html=True)
    st.info(f"Current AI Model Accuracy: {acc*100:.2f}%")

    st.write("### Overview & Research Focus")
    st.markdown(st.session_state.notes)

    st.write("---")
    st.write("### Sample Dataset Preview")
    st.dataframe(data.head())

    st.write("### 3D Stroke Risk Visualization")
    fig3d = px.scatter_3d(data, x="age", y="bp", z="cholesterol", color="stroke",
                         title="3D Stroke Risk Distribution", opacity=0.7)
    st.plotly_chart(fig3d, use_container_width=True)

# --------------------------- RESEARCH DASHBOARD ---------------------------
elif page == "Research Dashboard 📊":
    st.subheader("Research Dashboard: Data Insights")
    st.markdown("Interactive charts update with uploaded dataset or session patient records.")

    # choose data source: base synthetic, uploaded/train, or session records
    source = st.selectbox("Data source for dashboard:", ["Synthetic dataset", "Patient session records"])
    if source == "Patient session records" and not st.session_state.patient_records.empty:
        ds = st.session_state.patient_records.copy()
        # convert types if needed
        ds[["age","bp","cholesterol","glucose","bmi","stroke"]] = ds[["age","bp","cholesterol","glucose","bmi","stroke"]].apply(pd.to_numeric, errors="coerce")
    else:
        ds = data.copy()

    col1, col2 = st.columns(2)
    with col1:
        # handle when stroke column is missing or non-numeric
        if "stroke" in ds.columns and pd.api.types.is_numeric_dtype(ds["stroke"]):
            color_series = ds["stroke"].map({0: "No", 1: "Yes"})
        else:
            color_series = None
        fig_age = px.histogram(ds, x="age", color=color_series, barmode="overlay", title="Age Distribution vs Stroke")
        st.plotly_chart(fig_age, use_container_width=True)
    with col2:
        fig_bp = px.scatter(ds, x="bp", y="cholesterol", color=color_series, title="BP vs Cholesterol")
        st.plotly_chart(fig_bp, use_container_width=True)

    st.markdown("### Correlation Map")
    numeric = ds.select_dtypes(include=[np.number])
    if not numeric.empty:
        fig_corr = px.imshow(numeric.corr(), text_auto=True, aspect="auto", title="Feature Correlation Map")
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### Quick Insights")
    if "stroke" in ds.columns and ds["stroke"].dtype != object and ds["stroke"].notna().any():
        avg_age_stroke = ds[ds["stroke"] == 1]["age"].mean()
        avg_bp_stroke = ds[ds["stroke"] == 1]["bp"].mean()
        st.write(f"- Average age among stroke cases: **{avg_age_stroke:.1f}**")
        st.write(f"- Average BP among stroke cases: **{avg_bp_stroke:.1f}**")
    else:
        st.info("No stroke-labeled records in current view.")

# --------------------------- AI STROKE PREDICTOR ---------------------------
elif page == "AI Stroke Predictor 🤖":
    st.subheader("AI Stroke Risk Prediction (Manual entry or session record)")

    use_record = st.checkbox("Use a patient from session records", value=False)
    if use_record and not st.session_state.patient_records.empty:
        sel = st.selectbox("Select patient ID", options=list(st.session_state.patient_records["id"]))
        patient_row = st.session_state.patient_records[st.session_state.patient_records["id"] == sel].iloc[0]
        age = int(patient_row["age"]); bp = float(patient_row["bp"]); chol = float(patient_row["cholesterol"])
        glucose = float(patient_row["glucose"]); bmi = float(patient_row["bmi"])
        st.write(f"Using patient ID {sel} — Age {age}, BP {bp}, Chol {chol}")
    else:
        age = st.number_input("Age", 18, 100, 45)
        bp = st.number_input("Blood Pressure", 60, 240, 120)
        chol = st.number_input("Cholesterol", 50, 400, 180)
        glucose = st.number_input("Glucose", 40, 400, 100)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

    if st.button("Predict Stroke Risk"):
        X_input = pd.DataFrame([[age, bp, chol, glucose, bmi]], columns=["age","bp","cholesterol","glucose","bmi"])
        prob = model.predict_proba(X_input)[0][1]
        pred = model.predict(X_input)[0]
        st.success(f"Predicted stroke probability: **{prob*100:.2f}%**")
        if pred == 1:
            st.warning("High risk — recommend clinical follow-up.")
        else:
            st.info("Low risk — maintain healthy lifestyle.")

        # SHAP explainability (safe)
        X_full = data[["age","bp","cholesterol","glucose","bmi"]]
        try:
            fig_shap, shap_to_plot = safe_shap_plot(model, X_full)
            st.pyplot(fig_shap)
            # feature importance
            shap_values_mean = np.abs(shap_to_plot).mean(axis=0)
            shap_values_mean = np.array(shap_values_mean).ravel()
            feature_count = X_full.shape[1]
            if len(shap_values_mean) > feature_count:
                shap_values_mean = shap_values_mean[:feature_count]
            elif len(shap_values_mean) < feature_count:
                shap_values_mean = np.pad(shap_values_mean, (0, feature_count - len(shap_values_mean)), 'constant')
            importance_df = pd.DataFrame({"Feature": X_full.columns, "Importance": shap_values_mean}).sort_values("Importance", ascending=False)
            st.plotly_chart(px.bar(importance_df, x="Feature", y="Importance", title="Feature importance (SHAP)"), use_container_width=True)
        except Exception as e:
            st.error(f"SHAP visualization failed: {e}")

# --------------------------- PATIENT DATA RECORDER ---------------------------
elif page == "Patient Data Recorder 🧾":
    st.subheader("Patient Data Recorder (session-only)")

    with st.form("add_patient", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 0, 120, 45, key="rec_age")
            bp = st.number_input("Blood Pressure", 40, 300, 120, key="rec_bp")
            chol = st.number_input("Cholesterol", 0, 600, 180, key="rec_chol")
        with col2:
            glucose = st.number_input("Glucose", 0, 600, 100, key="rec_gluc")
            bmi = st.number_input("BMI", 5.0, 80.0, 25.0, key="rec_bmi")
            stroke = st.selectbox("Stroke (0=no,1=yes)", [0,1], index=0, key="rec_stroke")
        with col3:
            submitted = st.form_submit_button("Add Patient Record")
            if submitted:
                add_patient_record(age, bp, chol, glucose, bmi, stroke)
                st.success("Patient record added (session-only).")

    st.write("### Session Patient Records")
    df = st.session_state.patient_records.copy()
    if df.empty:
        st.info("No session records yet — add patients above.")
    else:
        search = st.text_input("Search by ID (leave blank to show all)")
        if search:
            try:
                sid = int(search)
                df = df[df["id"] == sid]
            except:
                st.warning("Enter a numeric ID to search.")
        st.dataframe(df)

        # delete UI
        st.markdown("#### Delete a record")
        if not df.empty:
            delete_id = st.number_input("Enter ID to delete", min_value=int(df["id"].min()), max_value=int(df["id"].max()), step=1)
            if st.button("Delete Record"):
                delete_patient_record(delete_id)
                st.success(f"Record {delete_id} deleted from session.")

# --------------------------- DRUG DISCOVERY LAB ---------------------------
# --------------------------- DRUG DISCOVERY LAB ---------------------------
elif page == "Drug Discovery Lab 💊":
    st.subheader("Drug Discovery Lab — Professional Compound Evaluation & Comparison")

    st.markdown("""
    Use this lab to compare two compounds with a biologically-informed heuristic.
    Outputs:
    - Efficacy Score (0–100) — internal composite metric
    - Predicted Stroke Reduction (%) — realistic, bounded (0–50%)
    - Confidence (0–100) — higher when properties are in drug-like ranges
    """)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Compound A")
        name_a = st.text_input("Name (A)", "Compound A", key="name_a")
        bind_a = st.number_input("Binding energy (A) [kcal/mol, negative stronger]", -20.0, 0.0, -8.0, step=0.1, key="bind_a")
        sol_a = st.slider("Solubility (A) (0-1)", 0.0, 1.0, 0.5, key="sol_a")
        tox_a = st.slider("Toxicity (A) (0-1)", 0.0, 1.0, 0.2, key="tox_a")

    with colB:
        st.markdown("#### Compound B")
        name_b = st.text_input("Name (B)", "Compound B", key="name_b")
        bind_b = st.number_input("Binding energy (B) [kcal/mol, negative stronger]", -20.0, 0.0, -10.0, step=0.1, key="bind_b")
        sol_b = st.slider("Solubility (B) (0-1)", 0.0, 1.0, 0.6, key="sol_b")
        tox_b = st.slider("Toxicity (B) (0-1)", 0.0, 1.0, 0.1, key="tox_b")

    def evaluate_compound(binding, solubility, toxicity):
        """
        Returns:
         - efficacy_score (0-100): composite internal metric
         - pred_reduction (0-50): predicted stroke reduction percent (realistic bound)
         - confidence (0-100): how trustworthy the estimate is (heuristic)
         - reasons: list of strings describing limiting factors
        """

        reasons = []

        # 1) Binding strength normalization:
        # stronger binding = more negative binding energy. We'll treat energies <= -12 as strong,
        # -5..-12 moderate, > -5 weak. Normalize to [0,1].
        binding_strength = ( -binding - 5 ) / (12 - 5)  # maps -5 -> 0, -12 -> 1
        binding_strength = np.clip(binding_strength, 0.0, 1.0)

        if binding > -5:
            reasons.append("Weak binding (binding energy > -5 kcal/mol).")

        # 2) Solubility factor: prefer moderate-high solubility
        sol_factor = np.clip(solubility, 0.0, 1.0)
        if sol_factor < 0.2:
            reasons.append("Low solubility — absorption may be poor.")

        # 3) Toxicity penalty
        tox_penalty = np.clip(toxicity, 0.0, 1.0)
        if tox_penalty > 0.6:
            reasons.append("High predicted toxicity — safety concerns.")

        # 4) Composite ADMET-like score (0..1)
        # Give binding more weight (0.5), solubility moderate (0.3), toxicity penalty reduces result.
        raw_score = (0.5 * binding_strength) + (0.3 * sol_factor) + (0.2 * (1 - tox_penalty))
        raw_score = np.clip(raw_score, 0.0, 1.0)

        # Map to efficacy_score 0-100 (internal)
        efficacy_score = round(raw_score * 100, 2)

        # Predicted stroke reduction: realistic upper bound 50% (set by domain conservatism)
        pred_reduction = round(raw_score * 50, 2)  # 0..50%

        # Confidence heuristic:
        # High confidence if binding_strength moderate+, sol between 0.25-0.9, toxicity low
        conf = 0.0
        conf += binding_strength * 50
        conf += (0.5 if (0.25 <= sol_factor <= 0.9) else 0.2) * 30
        conf += (1 - tox_penalty) * 20
        confidence = int(np.clip(conf, 5, 100))

        # Special-case: near-zero binding_strength and very low toxicity/neutral -> likely inert (e.g., water)
        if binding_strength < 0.05 and sol_factor > 0.4 and tox_penalty < 0.1:
            reasons.append("Profile resembles an inert/benign substance (low binding).")

        return efficacy_score, pred_reduction, confidence, reasons

    if st.button("Evaluate & Compare"):
        a_eff, a_red, a_conf, a_reasons = evaluate_compound(bind_a, sol_a, tox_a)
        b_eff, b_red, b_conf, b_reasons = evaluate_compound(bind_b, sol_b, tox_b)

        # Present summary
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### {name_a}")
            st.metric("Efficacy Score", f"{a_eff}/100")
            st.metric("Predicted Stroke Reduction", f"{a_red}% (bounded ≤50%)")
            st.metric("Confidence", f"{a_conf}/100")
            if a_reasons:
                st.warning(" • " + "\n • ".join(a_reasons))
        with col2:
            st.markdown(f"### {name_b}")
            st.metric("Efficacy Score", f"{b_eff}/100")
            st.metric("Predicted Stroke Reduction", f"{b_red}% (bounded ≤50%)")
            st.metric("Confidence", f"{b_conf}/100")
            if b_reasons:
                st.warning(" • " + "\n • ".join(b_reasons))

        # Comparative 3D scatter
        df_cmp = pd.DataFrame([
            {"Compound": name_a, "Binding": bind_a, "Solubility": sol_a, "Toxicity": tox_a, "Efficacy": a_eff, "StrokeReduction": a_red, "Confidence": a_conf},
            {"Compound": name_b, "Binding": bind_b, "Solubility": sol_b, "Toxicity": tox_b, "Efficacy": b_eff, "StrokeReduction": b_red, "Confidence": b_conf},
        ])

        fig3d = px.scatter_3d(
            df_cmp,
            x="Binding", y="Solubility", z="Toxicity",
            color="StrokeReduction", symbol="Compound",
            size="Efficacy",
            color_continuous_scale="RdYlGn_r",
            title="Compound Comparison — Efficacy vs Properties (Stroke Reduction %)"
        )
        fig3d.update_traces(marker=dict(opacity=0.9, line=dict(width=1, color="DarkSlateGray")))
        st.plotly_chart(fig3d, use_container_width=True)

        # 2D comparisons
        st.write("### 2D Comparison")
        st.plotly_chart(px.bar(df_cmp, x="Compound", y=["Efficacy", "StrokeReduction", "Confidence"], barmode="group"), use_container_width=True)

        # Final recommendation message
        if a_eff == b_eff and a_red == b_red:
            st.info("Both compounds show similar profiles. Consider structural modification or additional in-silico testing.")
        else:
            better = name_a if a_eff > b_eff else name_b
            st.success(f"Recommendation: {better} shows higher internal efficacy score based on the heuristic.")



# --------------------------- VR HEADSET MODE ---------------------------
elif page == "VR Headset Mode 🕶️":
    st.subheader("VR Headset Mode (Experimental)")
    st.markdown("Open via HTTPS in a VR-capable browser for immersive exploration.")
    aframe_html = """
    <a-scene embedded vr-mode-ui="enabled: true">
      <a-sky color="#ECECEC"></a-sky>
      <a-entity light="type: ambient; color: #BBB"></a-entity>
      <a-entity light="type: directional; intensity: 0.6" position="1 1 0"></a-entity>
      <a-camera position="0 1.6 0"></a-camera>
      <a-sphere position="0 1.5 -3" radius="0.5" color="#EF2D5E"></a-sphere>
    </a-scene>
    """
    html(aframe_html, height=480)

# --------------------------- UPLOAD / RETRAIN DATASET ---------------------------
elif page == "Upload/Train Dataset 📂":
    st.subheader("Upload & Retrain Model (CSV must have: age,bp,cholesterol,glucose,bmi,stroke)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        user_df = pd.read_csv(uploaded)
        st.dataframe(user_df.head())
        if st.button("Retrain Model"):
            try:
                model, new_acc = train_model(user_df)
                st.success(f"Retrained model. New accuracy: {new_acc*100:.2f}%")
            except Exception as e:
                st.error(f"Retrain failed: {e}")

# --------------------------- RESEARCH REPORT ---------------------------
elif page == "Research Report 🧾":
    st.subheader("Generate PDF Report")
    name = st.text_input("Researcher Name", "Simon")
    institution = st.text_input("Institution", "NeuroVR Institute")
    if st.button("Generate PDF"):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(160, 800, "NeuroVR Stroke Research Report")
        c.setFont("Helvetica", 11)
        c.drawString(50, 760, f"Researcher: {name}")
        c.drawString(50, 740, f"Institution: {institution}")
        c.drawString(50, 720, f"Model accuracy (current session): {acc*100:.2f}%")
        c.drawString(50, 700, "Summary: AI-assisted stroke prediction and compound simulation.")
        c.showPage()
        c.save()
        st.download_button("Download Report PDF", buffer.getvalue(), file_name="NeuroVR_Report.pdf")

# --------------------------- FOOTER ---------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color: gray; font-size: 13px;'>
        🧠 <b>System by Simon</b> | 📧 <a href='allinmer57@gmail.com'>allinmer57@gmail.com</a><br>
        © 2025 NeuroVR Lab — Experimental stroke research environment.
    </div>
    """,
    unsafe_allow_html=True
)
