import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    st.markdown("#### Energetische Auswertung")
    col4, col5, col6 = st.columns(3)
    energy_loss_text = f"{total_lost_energy:,.2f} kWh".replace(",", "X").replace(".", ",").replace("X", ".")
    energy_pct_text = f"{lost_energy_pct:,.2f} %".replace(",", "X").replace(".", ",").replace("X", ".")
    energy_generated_text = f"{total_generated_energy:,.2f} kWh".replace(",", "X").replace(".", ",").replace("X", ".")
    col4.metric("Verlust durch Clipping", energy_loss_text)
    col5.metric("Verlust in Prozent", energy_pct_text)
    col6.metric("Gesamtertrag", energy_generated_text)

        # --- Clipping-Analyse Visualisierung ---
    # (Charts erzeugen)
    fig, ax = plt.subplots(figsize=(12, 4))
    clipping_mask = pv_power_kw > max_power_kw
    ax.bar(pv_power_kw.index, clipped_power_kw, label="Nach Clipping", color="darkorange", alpha=0.6)
    ax.bar(pv_power_kw.index[clipping_mask], (pv_power_kw - max_power_kw)[clipping_mask], bottom=max_power_kw, label="Über Clipping-Grenze", color="red")
    ax.axhline(max_power_kw, color="red", linestyle=":", label="Max. WR-Leistung")
    ax.set_ylabel("Leistung in kW")
    ax.set_title("Clipping im Zeitverlauf")
    ax.legend()
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    monthly_losses = lost_energy_kwh.resample("M").sum()
    ax3.bar(monthly_losses.index, monthly_losses.values, width=20, color="salmon")
    ax3.set_ylabel("Verlust in kWh")
    ax3.set_title("Clipping-Verluste pro Monat")
    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(price_ct_per_kwh.index, price_ct_per_kwh.where(price_ct_per_kwh >= 0), color="orange", label="Preis ≥ 0 ct/kWh")
    ax2.plot(price_ct_per_kwh.index, price_ct_per_kwh.where(price_ct_per_kwh < 0), color="red", label="Preis < 0 ct/kWh")
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, label="Null-Linie")
    ax2.set_ylabel("Preis in ct/kWh")
    ax2.set_title("Day-Ahead Preise")
    ax2.legend()
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))

    # Charts in Übersicht anzeigen
    st.subheader("Clipping-Analyse Visualisierung")
    st.pyplot(fig)
    st.subheader("Verlorene Energie durch Clipping")
    st.pyplot(fig3)
    st.subheader("Day-Ahead Preisverlauf")
    st.pyplot(fig2)

    # --- PDF Export ---
    if st.button("📄 PDF-Bericht exportieren"):("📄 PDF-Bericht exportieren"):
        from matplotlib.backends.backend_pdf import PdfPages
        from io import BytesIO
        pdf_buffer = BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            # Deckblatt
            fig_cover = plt.figure(figsize=(8.27, 11.69))
            fig_cover.clf()
            fig_cover.text(0.5, 0.75, 'Wirtschaftlichkeitsanalyse – PV Clipping', ha='center', va='center', fontsize=20)
            fig_cover.text(0.5, 0.7, 'Erstellt: ' + pd.Timestamp.now().strftime('%d.%m.%Y'), ha='center', va='center', fontsize=12)
            pdf.savefig(fig_cover)
            plt.close(fig_cover)

            # Monetäre Auswertung
            fig_monetary, axm = plt.subplots(figsize=(8.27, 5))
            axm.axis('off')
            text = f"""Gesamtertrag EEG: {eeg_text}
Verlust durch Clipping: {clipping_loss_text}
Abregelung bei negativen Preisen: {curtailed_hours_text}"""
            axm.text(0.1, 0.5, text, fontsize=12, va='center')
            pdf.savefig(fig_monetary)
            plt.close(fig_monetary)

            # Energetische Auswertung
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

else:
    st.info("Bitte lade beide Dateien hoch, um die Analyse zu starten.")
