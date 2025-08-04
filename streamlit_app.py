import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO

# App-Konfiguration
st.set_page_config(page_title="PV Clipping Dashboard", layout="wide")
st.title("PV Lastgang Wirtschaftlichkeitsanalyse mit Clipping und EEG")

# Sidebar: Daten-Upload und Parameter
st.sidebar.header("Daten & Einstellungen")
pv_file = st.sidebar.file_uploader("PV Lastgang (Zeitstempel + kWh)", type=["csv","xlsx"])
price_file = st.sidebar.file_uploader("Day-Ahead Preise (Zeitstempel + €/MWh)", type=["csv","xlsx"])
max_power_kw = st.sidebar.number_input("Wechselrichter Maximalleistung (kW)", min_value=0.0, step=0.1, value=500.0)
eeg_ct = st.sidebar.number_input("EEG-Vergütung (ct/kWh)", min_value=0.0, step=0.1, value=6.3)

# Hilfsfunktion zum Laden der Zeitreihen
def load_series(file):
    """
    Liest Datei mit zwei Spalten: Zeitstempel und Wert.
    Gibt pandas.Series mit DateTimeIndex zurück.
    """
    if not file:
        return None
    # CSV oder Excel automatisch erkennen
    if file.name.lower().endswith('.csv'):
        df = pd.read_csv(file, usecols=[0,1], header=0)
    else:
        df = pd.read_excel(file, usecols=[0,1], header=0)
    # Parsen des Zeitstempels
    df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], errors='coerce')
    df = df.dropna(subset=[df.columns[0]])
    # Numerische Werte sicherstellen
    df.iloc[:,1] = pd.to_numeric(df.iloc[:,1], errors='coerce')
    df = df.dropna(subset=[df.columns[1]])
    series = pd.Series(df.iloc[:,1].values, index=df.iloc[:,0])
    series = series.sort_index()
    return series

# Daten laden
pv_kwh = load_series(pv_file)
price_mwh = load_series(price_file)

if pv_kwh is None or price_mwh is None:
    st.info("Bitte lade beide Dateien hoch, um die Analyse zu starten.")
    st.stop()

# Umrechnung und Clipping
pv_kw = pv_kwh * 4  # kW-Leistung aus kWh/15min
clipped_kw = np.minimum(pv_kw, max_power_kw)
clipped_kwh = clipped_kw / 4  # kWh nach Clipping
lost_kwh = pv_kwh - clipped_kwh

# Day-Ahead Preis synchronisieren
price_ct = price_mwh.reindex(pv_kw.index).fillna(0) / 10  # ct/kWh

# EEG-Einnahmen (bei Preis ≥ 0)
eeg_paid = np.where(price_ct >= 0, clipped_kwh * eeg_ct, 0)

# Kennzahlen berechnen
total_eeg = eeg_paid.sum() / 100  # in €
loss_eeg = (lost_kwh * eeg_ct).sum() / 100  # in €
hours_neg = ((pv_kwh > 0) & (price_ct < 0)).sum() / 4  # in Stunden
total_gen = clipped_kwh.sum()  # kWh
total_lost = lost_kwh.sum()  # kWh
pct_lost = total_lost / pv_kwh.sum() * 100  # %

# Formatierungsfunktion für deutsche Darstellung
def fmt(value, unit=""):
    s = f"{value:,.2f}"
    s = s.replace(",","X").replace(".",",").replace("X",".")
    return s + (f" {unit}" if unit else "")

# Anzeige Kennzahlen
st.subheader("Wirtschaftlichkeitsanalyse")
cols1 = st.columns(3)
cols1[0].metric("Gesamtertrag EEG", fmt(total_eeg, "€"))
cols1[1].metric("Verlust EEG (Clipping)", fmt(loss_eeg, "€"))
cols1[2].metric("Negativpreis-Stunden", fmt(hours_neg, "h"))

cols2 = st.columns(3)
cols2[0].metric("Verlust durch Clipping", fmt(total_lost, "kWh"))
cols2[1].metric("Verlust in %", fmt(pct_lost, "%"))
cols2[2].metric("Gesamtertrag (kWh)", fmt(total_gen, "kWh"))

# Charts erstellen
# 1) Clipping Zeitverlauf
fig1, ax1 = plt.subplots(figsize=(10,4))
mask = pv_kw > max_power_kw
ax1.bar(pv_kw.index, clipped_kw, label='Nach Clipping', color='orange', alpha=0.7)
ax1.bar(pv_kw.index[mask], pv_kw[mask] - max_power_kw,
        bottom=max_power_kw, label='Über Grenze', color='red')
ax1.axhline(max_power_kw, linestyle='--', color='firebrick', label='WR-Max')
ax1.set_title('Clipping im Zeitverlauf')
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
ax1.set_ylabel('Leistung (kW)')
ax1.legend()

# 2) Clipping-Verluste pro Monat
monthly_losses = lost_kwh.resample('M').sum()
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.bar(monthly_losses.index, monthly_losses.values, width=20, color='salmon')
ax2.set_title('Monatliche Clipping-Verluste')
ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
ax2.set_ylabel('Verlust (kWh)')

# 3) Day-Ahead Preisverlauf
fig3, ax3 = plt.subplots(figsize=(10,4))
pos = price_ct.where(price_ct >= 0)
neg = price_ct.where(price_ct < 0)
ax3.plot(pos.index, pos.values, color='green', label='Preis ≥ 0')
ax3.plot(neg.index, neg.values, color='red', label='Preis < 0')
ax3.axhline(0, color='black', linestyle='--')
ax3.set_title('Day-Ahead Preise')
ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
ax3.set_ylabel('Preis (ct/kWh)')
ax3.legend()

# Übersicht anzeigen
st.subheader('Clipping-Analyse Visualisierung')
st.pyplot(fig1)
st.pyplot(fig2)
st.pyplot(fig3)

# PDF-Export
if st.button('PDF-Bericht exportieren'):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        # Deckblatt
        cover = plt.figure(figsize=(8.27,11.69)); cover.clf()
        cover.text(0.5,0.6,'PV Clipping Wirtschaftlichkeitsanalyse', ha='center', va='center', fontsize=24)
        pdf.savefig(cover); plt.close(cover)
        # Kennzahlen Seite
        fig_k, ax_k = plt.subplots(figsize=(8.27,5)); ax_k.axis('off')
        txt = (
            f"Gesamtertrag EEG: {fmt(total_eeg,'€')}\n"
            f"Verlust EEG: {fmt(loss_eeg,'€')}\n"
            f"Negativpreis-Stunden: {fmt(hours_neg,'h')}\n\n"
            f"Verlust Clipping: {fmt(total_lost,'kWh')}\n"
            f"Verlust %: {fmt(pct_lost,'%')}\n"
            f"Gesamt kWh: {fmt(total_gen,'kWh')}"
        )
        ax_k.text(0.1,0.5,txt, fontsize=12)
        pdf.savefig(fig_k); plt.close(fig_k)
        # Charts
        pdf.savefig(fig1); pdf.savefig(fig2); pdf.savefig(fig3)
    buffer.seek(0)
    st.download_button('Download PDF', data=buffer, file_name='PV_Clipping_Analyse.pdf', mime='application/pdf')
