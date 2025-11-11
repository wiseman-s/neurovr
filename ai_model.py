import os, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

MODEL_PATH = "models/treatment_model.joblib"

def make_synthetic_dataset(n=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    # features: severity_level (0-2), treatment_type (0-4), days, random biological variability
    severity = rng.integers(0,3,size=n)
    treatment = rng.integers(0,5,size=n)
    days = rng.integers(5,61,size=n)
    potency = rng.random(n)
    noise = rng.normal(0,5,size=n)
    # target: effectiveness score 0-100 (synthetic formula)
    base = 100 - (severity * 20) + (treatment * 5) + (potency * 10) - (days * 0.2)
    score = np.clip(base + noise, 0, 100)
    df = pd.DataFrame({"severity":severity, "treatment":treatment, "days":days, "potency":potency, "score":score})
    return df

def train_and_save_model(path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = make_synthetic_dataset(2000)
    X = df[["severity","treatment","days","potency"]]
    y = df["score"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, path)
    return rf

def load_or_train_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            pass
    return train_and_save_model(path)

def predict_treatment_effect(model, severity_label, treatment_label, days):
    # map labels to numeric codes used in synthetic dataset
    severity_map = {"Mild":0,"Moderate":1,"Severe":2}
    treatment_map = {"No Treatment":0,"Drug A (NeuroProtect)":1,"Drug B (ClotBuster)":2,"Oxygen Therapy":3,"Neuro-Stimulation":4}
    sev = severity_map.get(severity_label,1)
    tr = treatment_map.get(treatment_label,0)
    # potency unknown -> assume 0.5
    X = [[sev, tr, days, 0.5]]
    pred = model.predict(X)[0]
    return float(pred)
