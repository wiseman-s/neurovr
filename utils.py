import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os

def generate_pdf_report(days_arr, neurons_alive, severity, treatment, outpath='neurovr_report.pdf'):
    # Create a multipage PDF with charts
    os.makedirs('reports', exist_ok=True)
    out = os.path.join('reports', outpath)
    with PdfPages(out) as pdf:
        plt.figure(figsize=(8,4))
        plt.plot(days_arr, neurons_alive, marker='o')
        plt.title(f'Neuron Survival Over Time — {severity} / {treatment}')
        plt.xlabel('Days'); plt.ylabel('Neuron Survival (%)')
        plt.grid(True)
        pdf.savefig(); plt.close()

        # second page: summary text as a figure
        plt.figure(figsize=(8,6))
        summary = f"NeuroVR Lab Report\\nSeverity: {severity}\\nTreatment: {treatment}\\nFinal survival: {neurons_alive[-1]:.1f}%"
        plt.text(0.1, 0.6, summary, fontsize=12)
        plt.axis('off')
        pdf.savefig(); plt.close()
    return out
