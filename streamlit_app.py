import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="PV Lastgang Analyse", layout="wide")
st.title("PV Lastgang Wirtschaftlichkeitsanalyse mit Clipping und EEG")

# --- Datei-Upload ---
st.sidebar.header("Daten eingeben")
pv_file = st.sidebar.file_uploader("PV Lastgang (Viertelstundenwerte in kWh)", type=["csv", "xlsx"])
price_file = st.sidebar.file_uploader("Day-Ahead Preise (€/MWh, Viertelstundenwerte)", type=["csv", "xlsx"])

# --- Parameter-Eingaben ---
max_power_kw = st.sidebar.number_input("Wechselrichter Maximalleistung (kW)", min_value=0.0, step=0.1)
eeg_ct_per_kwh = st.sidebar.number_input("EEG-Vergütung (ct/kWh)", min_value=0.0, step=0.1)  # feste Vergütung

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
    clipped_energy_kwh = clipped_power_kw / 4  # zurück in kWh
    lost_energy_kwh = pv_data.iloc[:, 0] - clipped_energy_kwh

    # Day-Ahead Preis in €/MWh -> ct/kWh
    price_ct_per_kwh = price_data.iloc[:, 0] / 10

    # EEG: Feste Vergütung wenn Preis > 0, sonst 0 (keine Einspeisung)
    eeg_paid = np.where(price_ct_per_kwh > 0, clipped_energy_kwh * eeg_ct_per_kwh, 0)
    total_eeg_revenue = np.sum(eeg_paid)
    lost_eeg_revenue = np.sum(lost_energy_kwh * eeg_ct_per_kwh)

    # Häufigkeit der Abregelung durch negative Preise in Stunden
    curtailed_hours = np.sum((pv_data.iloc[:, 0] > 0) & (price_ct_per_kwh < 0).values) / 4

    # Energieverluste durch Clipping (kWh und %)
    total_pv_energy = np.sum(pv_data.iloc[:, 0])
    total_lost_energy = np.sum(lost_energy_kwh)
    total_generated_energy = np.sum(clipped_energy_kwh)
    lost_energy_pct = (total_lost_energy / total_pv_energy * 100) if total_pv_energy > 0 else 0

    # Ausgabe
    st.subheader("Wirtschaftlichkeitsanalyse")
    col1, col2, col3 = st.columns(3)
    col1.metric("Gesamtertrag (EEG) [€]", f"{total_eeg_revenue / 100:.2f}")
    col2.metric("Verlust durch Clipping [€]", f"{lost_eeg_revenue / 100:.2f}")
    col3.metric("Abregelung wegen negativer Preise [h]", f"{curtailed_hours:.1f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Verlust durch Clipping [kWh]", f"{total_lost_energy:.2f}")
    col5.metric("Verlust in Prozent [%]", f"{lost_energy_pct:.2f}")
    col6.metric("Gesamtertrag (kWh)", f"{total_generated_energy:.2f}")

    # Visualisierung
    st.subheader("Clipping-Analyse")
    fig, ax = plt.subplots(figsize=(12, 4))
    clipping_mask = pv_power_kw > max_power_kw
    ax.bar(pv_power_kw.index[clipping_mask], (pv_power_kw - max_power_kw)[clipping_mask], label="Über Clipping-Grenze (kW)", color="skyblue")
    ax.bar(pv_power_kw.index, clipped_power_kw, label="Nach Clipping (kW)", color="darkorange", alpha=0.6)
    ax.axhline(max_power_kw, color="red", linestyle=":", label="Max. WR-Leistung")
    ax.set_ylabel("Leistung [kW]")
    ax.set_title("Clipping im Zeitverlauf")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Verlorene Energie durch Clipping (monatlich aggregiert)")
    monthly_losses = lost_energy_kwh.resample("M").sum()
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.bar(monthly_losses.index, monthly_losses.values, width=20, color="salmon")
    ax3.set_ylabel("Verlust [kWh]")
    ax3.set_title("Clipping-Verluste pro Monat")
    fig3.autofmt_xdate()
    st.pyplot(fig3)

    st.subheader("Day-Ahead Preisverlauf (ct/kWh, stündlich aggregiert)")
    hourly_prices = price_ct_per_kwh.resample("H").mean()
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(hourly_prices.index, hourly_prices.where(hourly_prices >= 0), color="orange", label="Preis ≥ 0 ct/kWh")
    ax2.plot(hourly_prices.index, hourly_prices.where(hourly_prices < 0), color="red", label="Preis < 0 ct/kWh")
    ax2.set_ylabel("Preis [ct/kWh]")
    ax2.set_title("Day-Ahead Preise (aggregiert nach Stunden, negativ hervorgehoben)")
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, label="Null-Linie")
    ax2.legend()
    st.pyplot(fig2)

else:
    st.info("Bitte lade beide Dateien hoch, um die Analyse zu starten.")
