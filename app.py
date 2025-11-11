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
elif page == "Drug Discovery Lab 💊":
    st.subheader("Drug Discovery Lab — AI-Guided Compound Efficacy & Risk Visualizer")

    st.markdown("""
    This lab uses a **machine learning surrogate model** trained on drug-like features  
    to simulate how compounds may affect **stroke recovery and reduction risk**.  
    Use manual mode for two-compound comparison or upload a CSV for batch screening.
    """)

    import io, numpy as np, pandas as pd
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill
    from sklearn.ensemble import RandomForestRegressor
    import plotly.express as px

    # --- Surrogate ML model (simple RandomForest) ---
    np.random.seed(42)
    X = np.random.rand(500, 3)
    y = 0.55*(1 - X[:,0]) + 0.3*X[:,1] + 0.15*(1 - X[:,2]) + np.random.normal(0, 0.05, 500)
    surrogate = RandomForestRegressor(n_estimators=200, random_state=42)
    surrogate.fit(X, y)

    # --- Known compounds (auto-fill data) ---
    compound_data = {
        "Aspirin":         {"Binding": -8.0,  "Solubility": 0.7,  "Toxicity": 0.2},
        "Clopidogrel":     {"Binding": -9.0,  "Solubility": 0.5,  "Toxicity": 0.25},
        "tPA (Alteplase)": {"Binding": -10.5, "Solubility": 0.6,  "Toxicity": 0.35},
        "Citicoline":      {"Binding": -7.5,  "Solubility": 0.8,  "Toxicity": 0.1},
        "Edaravone":       {"Binding": -9.2,  "Solubility": 0.55, "Toxicity": 0.15},
        "Piracetam":       {"Binding": -6.5,  "Solubility": 0.9,  "Toxicity": 0.05},
        "Memantine":       {"Binding": -8.8,  "Solubility": 0.65, "Toxicity": 0.2},
        "Water (control)": {"Binding": -0.5,  "Solubility": 1.0,  "Toxicity": 0.0},
        "Experimental-X1": {"Binding": -11.0, "Solubility": 0.4,  "Toxicity": 0.3},
    }

    # --- Efficacy predictor ---
    def evaluate_compound(binding, solubility, toxicity):
        X_pred = np.array([[-binding/15, solubility, toxicity]])
        eff_pred = float(np.clip(surrogate.predict(X_pred)[0], 0, 1))
        efficacy = round(eff_pred * 100, 2)
        reduction = round(eff_pred * 50 + np.random.uniform(-2, 2), 2)
        confidence = int(np.clip(70 + eff_pred * 25, 0, 100))
        return efficacy, reduction, confidence

    # --- Excel export ---
    def export_excel(df):
        wb = Workbook()
        ws = wb.active
        ws.title = "Results"
        ws.append(list(df.columns))
        for _, row in df.iterrows():
            ws.append(row.tolist())
            eff, tox = row["Efficacy"], row["Toxicity"]
            color = "90EE90" if eff > 75 and tox < 0.3 else "FFA07A" if eff < 40 else "FFFACD"
            for cell in ws[ws.max_row]:
                cell.fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
        buf = io.BytesIO(); wb.save(buf); buf.seek(0)
        return buf

    mode = st.radio("Select Mode", ["Manual Comparison", "Batch Upload (CSV)"])

    # ---------------- Manual Comparison ----------------
    if mode == "Manual Comparison":
        colA, colB = st.columns(2)
        with colA:
            st.markdown("##### Compound A")
            name_a = st.selectbox("Compound A", list(compound_data.keys()), index=0, key="cmp_a")
            data_a = compound_data[name_a]
            st.caption(f"Auto-filled: Binding={data_a['Binding']}, Solubility={data_a['Solubility']}, Toxicity={data_a['Toxicity']}")
            bind_a = st.number_input("Binding Energy (A) [kcal/mol]", -20.0, 0.0, data_a["Binding"])
            sol_a = st.slider("Solubility (A)", 0.0, 1.0, data_a["Solubility"])
            tox_a = st.slider("Toxicity (A)", 0.0, 1.0, data_a["Toxicity"])

        with colB:
            st.markdown("##### Compound B")
            name_b = st.selectbox("Compound B", list(compound_data.keys()), index=1, key="cmp_b")
            data_b = compound_data[name_b]
            st.caption(f"Auto-filled: Binding={data_b['Binding']}, Solubility={data_b['Solubility']}, Toxicity={data_b['Toxicity']}")
            bind_b = st.number_input("Binding Energy (B) [kcal/mol]", -20.0, 0.0, data_b["Binding"])
            sol_b = st.slider("Solubility (B)", 0.0, 1.0, data_b["Solubility"])
            tox_b = st.slider("Toxicity (B)", 0.0, 1.0, data_b["Toxicity"])

        if st.button("Run Compound Evaluation"):
            a_eff, a_red, a_conf = evaluate_compound(bind_a, sol_a, tox_a)
            b_eff, b_red, b_conf = evaluate_compound(bind_b, sol_b, tox_b)

            df_cmp = pd.DataFrame([
                {"Compound": name_a, "Binding": bind_a, "Solubility": sol_a, "Toxicity": tox_a,
                 "Efficacy": a_eff, "StrokeReduction": a_red, "Confidence": a_conf},
                {"Compound": name_b, "Binding": bind_b, "Solubility": sol_b, "Toxicity": tox_b,
                 "Efficacy": b_eff, "StrokeReduction": b_red, "Confidence": b_conf},
            ])

            # --- Plotly Charts ---
            fig3d = px.scatter_3d(
                df_cmp, x="Binding", y="Solubility", z="Toxicity",
                color="StrokeReduction", size="Efficacy", symbol="Compound",
                color_continuous_scale="RdYlGn", title="3D View — Compound Properties"
            )
            st.plotly_chart(fig3d, use_container_width=True)

            fig2d = px.bar(df_cmp, x="Compound", y=["Efficacy","StrokeReduction"],
                           barmode="group", title="Efficacy vs Stroke Reduction (%)")
            st.plotly_chart(fig2d, use_container_width=True)

            better = name_a if a_eff > b_eff else name_b
            st.success(f"🏆 Recommended Compound: **{better}** — higher predicted efficacy.")

            # --- FIXED 3D INTERACTIVE VIEW ---
            a_html = """
            <script src="https://aframe.io/releases/1.4.0/aframe.min.js"></script>
            <a-scene background="color: #ECECEC">
              <a-entity light="type: ambient; intensity: 0.7"></a-entity>
              <a-entity light="type: directional; position: 2 4 3; intensity: 0.8"></a-entity>
              <a-plane rotation="-90 0 0" width="10" height="10" color="#f0f0f0"></a-plane>
            """
            for i, p in enumerate(df_cmp.to_dict(orient="records")):
                color = "#4CC3D9" if p["Efficacy"] < 60 else "#7CFC00" if p["Efficacy"] < 80 else "#228B22"
                xpos = i * 1.8 - 1.8
                a_html += f"""
                  <a-box position="{xpos} 0.25 -3" depth="0.5" height="0.5" width="0.5" color="{color}"></a-box>
                  <a-text value="{p['Compound']}" align="center" position="{xpos} 0.9 -3" color="#111"></a-text>
                  <a-text value="Stroke Reduction: {p['StrokeReduction']}%" align="center" position="{xpos} 0.65 -3" color="#333"></a-text>
                """
            a_html += """
              <a-camera position="0 1.6 2"></a-camera>
            </a-scene>
            """
            st.markdown("### 🧠 Interactive 3D Compound Display")
            st.components.v1.html(a_html, height=500)

            # --- Export ---
            csv = df_cmp.to_csv(index=False).encode("utf-8")
            excel = export_excel(df_cmp)
            st.download_button("📥 Download Results (CSV)", csv, "compound_results.csv", "text/csv")
            st.download_button("📘 Download Results (Excel)", excel, "compound_results.xlsx")

    # ---------------- Batch Upload ----------------
    else:
        st.markdown("### Upload Compound Dataset (CSV)")
        uploaded = st.file_uploader("Upload file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            required = {"Compound","Binding","Solubility","Toxicity"}
            if not required.issubset(df.columns):
                st.error(f"CSV must contain: {', '.join(required)}")
            else:
                df[["Efficacy","StrokeReduction","Confidence"]] = df.apply(
                    lambda r: pd.Series(evaluate_compound(r["Binding"], r["Solubility"], r["Toxicity"])), axis=1)
                st.success(f"✅ {len(df)} compounds evaluated successfully.")
                st.dataframe(df.head())

                fig3d = px.scatter_3d(df, x="Binding", y="Solubility", z="Toxicity",
                                      color="StrokeReduction", size="Efficacy",
                                      color_continuous_scale="RdYlGn",
                                      title="3D Screening — Surrogate Model")
                st.plotly_chart(fig3d, use_container_width=True)








# --------------------------- VR HEADSET MODE ---------------------------
elif page == "VR Headset Mode 🕶️":
    import streamlit.components.v1 as components

    st.subheader("🧠 NeuroVR Headset Mode — Immersive Visualization")
    st.markdown("""
    This experimental mode enables **3D and VR visualization** of compound effects and stroke risk patterns.  
    Open the app via **HTTPS** on a **WebXR-compatible headset** (e.g., Oculus Quest, Pico, or HoloLens) to enter full VR.
    """)

    aframe_html = """
    <html>
      <head>
        <script src="https://aframe.io/releases/1.5.0/aframe.min.js"></script>
      </head>
      <body>
        <a-scene vr-mode-ui="enabled: true" embedded>
          <!-- Background -->
          <a-sky color="#ECECEC"></a-sky>

          <!-- Lighting -->
          <a-entity light="type: ambient; color: #BBB"></a-entity>
          <a-entity light="type: directional; color: #FFF; intensity: 0.8" position="1 1 0"></a-entity>

          <!-- Camera -->
          <a-entity position="0 1.6 0">
            <a-camera wasd-controls-enabled="true"></a-camera>
          </a-entity>

          <!-- Sample Drug Compounds (3D spheres) -->
          <a-entity id="compounds">
            <a-sphere position="-1 1.5 -3" radius="0.4" color="#EF2D5E" compound-name="Aspirin"></a-sphere>
            <a-sphere position="0 1.5 -4" radius="0.4" color="#4CC3D9" compound-name="Citicoline"></a-sphere>
            <a-sphere position="1 1.5 -3" radius="0.4" color="#7BC8A4" compound-name="tPA"></a-sphere>
            <a-sphere position="0 0.8 -2.5" radius="0.3" color="#FFC65D" compound-name="Edaravone"></a-sphere>
          </a-entity>

          <!-- Labels -->
          <a-text value="NeuroVR Lab — Compound Risk Space" color="#111" position="-1.2 2.5 -3"></a-text>
          <a-text value="(Look around or use headset controls to explore)" color="#333" position="-1.8 2.2 -3.2" width="4"></a-text>

          <!-- Animation Example -->
          <a-animation attribute="rotation" dur="12000" to="0 360 0" repeat="indefinite"></a-animation>

          <!-- Interaction Script -->
          <script>
            AFRAME.registerComponent('compound-info', {
              init: function () {
                this.el.addEventListener('click', () => {
                  const name = this.el.getAttribute('compound-name');
                  alert(`Compound: ${name}\\nBinding: variable\\nSolubility: variable\\nRisk: dynamic`);
                });
              }
            });

            // Attach info component to all compound spheres
            document.querySelectorAll('[compound-name]').forEach(e => e.setAttribute('compound-info', ''));
          </script>
        </a-scene>
      </body>
    </html>
    """

    components.html(aframe_html, height=600, scrolling=False)




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
