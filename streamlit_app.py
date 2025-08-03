import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="PV Lastgang Analyse", layout="wide")
st.title("PV Lastgang Wirtschaftlichkeitsanalyse mit Clipping und EEG")

# --- Datei-Upload ---
st.sidebar.header("Daten eingeben")
pv_file = st.sidebar.file_uploader("PV Lastgang (Viertelstundenwerte in kWh)", type=["csv", "xlsx"])
price_file = st.sidebar.file_uploader("Day-Ahead Preise (€/MWh, Viertelstundenwerte)", type=["csv", "xlsx"])

# --- Parameter-Eingaben ---
max_power_kw = st.sidebar.number_input("Wechselrichter Maximalleistung (kW)", min_value=0.0, step=0.1)
eeg_ct_per_kwh = st.sidebar.number_input("EEG-Vergütung (ct/kWh)", min_value=0.0, step=0.1)

def load_data(uploaded_file):
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
        elif uploaded_file.name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file, parse_dates=True, index_col=0)
    return None

pv_data = load_data(pv_file)
price_data = load_data(price_file)

if pv_data is not None and price_data is not None:

    st.subheader("Datenübersicht")
    st.write("PV Lastgang (kWh pro 15 Minuten):", pv_data.head())
    st.write("Day-Ahead Preise (€/MWh):", price_data.head())

    # Umrechnung: 15-Minuten kWh * 4 = kW Leistung
    pv_power_kw = pv_data.iloc[:, 0] * 4
    clipped_power_kw = np.minimum(pv_power_kw, max_power_kw)
    clipped_energy_kwh = clipped_power_kw / 4
    lost_energy_kwh = pv_data.iloc[:, 0] - clipped_energy_kwh

    price_ct_per_kwh = price_data.iloc[:, 0] / 10

    eeg_paid = np.where(price_ct_per_kwh > 0, clipped_energy_kwh * eeg_ct_per_kwh, 0)
    total_eeg_revenue = np.sum(eeg_paid)
    lost_eeg_revenue = np.sum(lost_energy_kwh * eeg_ct_per_kwh)
    curtailed_hours = np.sum((pv_data.iloc[:, 0] > 0) & (price_ct_per_kwh < 0).values) / 4

    total_pv_energy = np.sum(pv_data.iloc[:, 0])
    total_lost_energy = np.sum(lost_energy_kwh)
    total_generated_energy = np.sum(clipped_energy_kwh)
    lost_energy_pct = (total_lost_energy / total_pv_energy * 100) if total_pv_energy > 0 else 0

    # --- Wirtschaftlichkeitsanalyse ---
    st.subheader("Wirtschaftlichkeitsanalyse")

    st.markdown("#### Monetäre Auswertung")
    col1, col2, col3 = st.columns(3)
    eeg_text = f"{total_eeg_revenue / 100:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    clipping_loss_text = f"{lost_eeg_revenue / 100:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    curtailed_hours_text = f"{curtailed_hours:,.1f} h".replace(",", "X").replace(".", ",").replace("X", ".")
    col1.metric("Gesamtertrag EEG", eeg_text)
    col2.metric("Verlust durch Clipping", clipping_loss_text)
    col3.metric("Abregelung wegen negativer Preise", curtailed_hours_text)

    st.markdown("###            # Energetische Auswertung
            fig_energy, axe = plt.subplots(figsize=(8.27, 5))
            axe.axis('off')
            text2 = f"""Verlust durch Clipping: {energy_loss_text}
Verlust in Prozent: {energy_pct_text}
Gesamtertrag: {energy_generated_text}"""
            axe.text(0.1, 0.5, text2, fontsize=12, va='center')
            pdf.savefig(fig_energy)
            plt.close(fig_energy)

            # Charts
            pdf.savefig(fig)  # Clipping Zeitverlauf
            pdf.savefig(fig3) # Clipping Verluste
            pdf.savefig(fig2) # Day-Ahead Preise

        pdf_buffer.seek(0)
        st.download_button("Download PDF", data=pdf_buffer, file_name="PV_Wirtschaftlichkeitsanalyse.pdf", mime='application/pdf')
