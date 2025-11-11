# NeuroVR Lab — Streamlit Research Demo (Stroke)
This project is a demonstration prototype that shows how **Virtual Reality-inspired visualization** and **computational simulation** can support stroke research and drug discovery.

## Features included
- 3D virtual brain visualization (Plotly scatter)
- AI prediction module (trained on synthetic data)
- Drug discovery lab (simulate hypothetical compound effects)
- Sample dataset (synthetic placeholder) in `data/`
- PDF report generation (multi-page)
- Embedded A-Frame scene for browser-based VR (WebXR) preview
- Export CSV and PDF

## How to run
1. Create a Python environment (recommended) and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Notes
- The included AI model is trained on **synthetic** data. Replace `/data/sample_real_like.csv` with real research datasets (MRI-derived features, clinical trial CSVs) and re-train the model for real results.
- WebXR/VR headset support requires HTTPS hosting and a WebXR-capable browser/device.
- This demo is for research demonstration and educational use only.
