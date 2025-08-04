import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages

# --- Seite konfigurieren ---
st.set_page_config(page_title="PV Lastgang Analyse", layout="wide")
st.title("PV Lastgang Wirtschaftlichkeitsanalyse mit Clipping und EEG")

# --- Datei-Upload ---
st.sidebar.header("Daten & Einstellungen")
pv_file = st.sidebar.file_uploader("PV Lastgang (Viertelstundenwerte in kWh)", type=["csv", "xlsx"])
price_file = st.sidebar.file_uploader("Day-Ahead Preise (€/MWh, Viertelstundenwerte)", type=["csv", "xlsx"])

# --- Parameter ---
max_power_kw = st.sidebar.number_input("Wechselrichter Maximalleistung (kW)", min_value=0.0, step=0.1)
eeg_ct_per_kwh = st.sidebar.number_input("EEG-Vergütung (ct/kWh)", min_value=0.0, step=0.1)

# --- Daten laden (einfach) ---
def load_series(file):
    if file is None:
        return None
    if file.name.endswith('.csv'):
        df = pd.read_csv(file, index_col=0, parse_dates=True)
    else:
        df = pd.read_excel(file, index_col=0, parse_dates=True)
    # Erste Spalte als Zeitstempel, zweite als Werte
    ts = df.iloc[:,0]
    ts.index.name = 'Timestamp'
    return ts

pv_series = load_series(pv_file)
price_series = load_series(price_file)

if pv_series is None or price_series is None:
    st.info("Bitte lade beide Dateien hoch, um die Analyse zu starten.")
    st.stop()

# --- Clipping-Berechnungen ---
pv_kw = pv_series * 4  # kWh -> kW
clipped_kw = np.minimum(pv_kw, max_power_kw)
clipped_kwh = clipped_kw / 4
lost_kwh = pv_series - clipped_kwh

# --- Preise umrechnen ---
price_ct_per_kwh = price_series / 10  # €/MWh -> ct/kWh

# --- Kennzahlen ---
eeg_paid = np.where(price_ct_per_kwh > 0, clipped_kwh * eeg_ct_per_kwh, 0)
total_eeg = eeg_paid.sum() / 100
loss_eeg = (lost_kwh * eeg_ct_per_kwh).sum() / 100
hours_curtailed = ((pv_series > 0) & (price_ct_per_kwh < 0)).sum() / 4
total_generated_kwh = clipped_kwh.sum()
total_lost_kwh = lost_kwh.sum()
lost_pct = total_lost_kwh / pv_series.sum() * 100

# --- Formatierung ---
def fmt(x, unit):
    s = f"{x:,.2f}"
    s = s.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{s} {unit}"

str_total_eeg = fmt(total_eeg, "€")
str_loss_eeg = fmt(loss_eeg, "€")
str_hours = fmt(hours_curtailed, "h")
str_lost_kwh = fmt(total_lost_kwh, "kWh")
str_pct = fmt(lost_pct, "%")
str_generated = fmt(total_generated_kwh, "kWh")

# --- Anzeige Kennzahlen ---
st.subheader("Wirtschaftlichkeitsanalyse")
st.markdown("**Monetäre Auswertung**")
c1, c2, c3 = st.columns(3)
c1.metric("Gesamtertrag EEG", str_total_eeg)
c2.metric("Verlust durch Clipping", str_loss_eeg)
c3.metric("Abregelung (neg. Preise)", str_hours)

st.markdown("**Energetische Auswertung**")
c4, c5, c6 = st.columns(3)
c4.metric("Verlust durch Clipping", str_lost_kwh)
c5.metric("Verlust in %", str_pct)
c6.metric("Gesamtertrag (kWh)", str_generated)

# --- Charts ---
# 1. Clipping im Zeitverlauf
fig1, ax1 = plt.subplots(figsize=(12,4))
mask = pv_kw > max_power_kw
ax1.bar(pv_kw.index, clipped_kw, color='orange', alpha=0.6, label='Nach Clipping')
ax1.bar(pv_kw.index[mask], pv_kw[mask] - max_power_kw, bottom=max_power_kw, color='red', label='Über Grenze')
ax1.axhline(max_power_kw, color='red', linestyle='--', label='WR Max')
ax1.set_ylabel('Leistung (kW)')
ax1.set_title('Clipping im Zeitverlauf')
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
ax1.legend()

# 2. Clipping-Verluste pro Monat
fig2, ax2 = plt.subplots(figsize=(12,4))
monthly_losses = lost_kwh.resample('M').sum()
ax2.bar(monthly_losses.index, monthly_losses.values, width=20, color='salmon')
ax2.set_ylabel('Verlust (kWh)')
ax2.set_title('Clipping-Verluste pro Monat')
ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))

# 3. Day-Ahead Preise
fig3, ax3 = plt.subplots(figsize=(12,4))
ax3.plot(price_ct_per_kwh.index, price_ct_per_kwh.where(price_ct_per_kwh >= 0), color='orange', label='≥ 0 ct/kWh')
ax3.plot(price_ct_per_kwh.index, price_ct_per_kwh.where(price_ct_per_kwh < 0), color='red', label='< 0 ct/kWh')
ax3.axhline(0, color='black', linestyle='--', label='Null-Linie')
ax3.set_ylabel('Preis (ct/kWh)')
ax3.set_title('Day-Ahead Preise')
ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
ax3.legend()

# Charts anzeigen
st.subheader('Clipping-Analyse Visualisierung')
st.pyplot(fig1)
st.subheader('Verlorene Energie durch Clipping')
st.pyplot(fig2)
st.subheader('Day-Ahead Preisverlauf')
st.pyplot(fig3)

# --- PDF-Export ---
if st.button('📄 PDF-Bericht exportieren'):
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)
    buf.seek(0)
    st.download_button('Download PDF', data=buf, file_name='PV_Analyse.pdf', mime='application/pdf')
