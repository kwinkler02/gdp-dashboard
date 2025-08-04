import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages

st.set_page_config(page_title="PV Lastgang Analyse", layout="wide")
st.title("PV Lastgang Wirtschaftlichkeitsanalyse mit Clipping und EEG")

# --- Datei-Upload ---
pv_file = st.sidebar.file_uploader("PV Lastgang (Viertelstundenwerte in kWh)", type=["csv", "xlsx"])
price_file = st.sidebar.file_uploader("Day-Ahead Preise (€/MWh, Viertelstundenwerte)", type=["csv", "xlsx"])

# --- Parameter-Eingaben ---
max_power_kw = st.sidebar.number_input("Wechselrichter Maximalleistung (kW)", min_value=0.0, step=0.1)
eeg_ct_per_kwh = st.sidebar.number_input("EEG-Vergütung (ct/kWh)", min_value=0.0, step=0.1)

# --- Daten laden ---
def load_data(upl):
    if upl is None:
        return None
    if upl.name.endswith('.csv'):
        return pd.read_csv(upl, index_col=0)
    return pd.read_excel(upl, index_col=0)

pv_data = load_data(pv_file)
price_data = load_data(price_file)

# Prüfen
if pv_data is None or price_data is None:
    st.info("Bitte lade beide Dateien hoch, um die Analyse zu starten.")
    st.stop()

# Index als Datetime-Index parsen
pv_data.index = pd.to_datetime(pv_data.index, dayfirst=True, errors='coerce')
price_data.index = pd.to_datetime(price_data.index, dayfirst=True, errors='coerce')

# Ungültige entfernen
pv_data = pv_data[pv_data.index.notna()]
price_data = price_data[price_data.index.notna()]

# Leistung und Clipping
pv_kwh = pd.to_numeric(pv_data.iloc[:,0], errors='coerce')
pv_kw = pv_kwh * 4
clipped_kw = np.minimum(pv_kw, max_power_kw)
clipped_kwh = clipped_kw / 4
lost_kwh = pv_kwh - clipped_kwh

# Preise umrechnen (ct/kWh)
price_mwh = pd.to_numeric(price_data.iloc[:,0], errors='coerce')
price_ct = price_mwh / 10

# Kennzahlen
eeg_paid = np.where(price_ct >= 0, clipped_kwh * eeg_ct_per_kwh, 0)
total_eeg = eeg_paid.sum() / 100
loss_eeg = (lost_kwh * eeg_ct_per_kwh).sum() / 100
neg_hours = ((pv_kwh > 0) & (price_ct < 0)).sum() / 4

total_gen = clipped_kwh.sum()
total_loss = lost_kwh.sum()
loss_pct = total_loss / pv_kwh.sum() * 100 if pv_kwh.sum()>0 else 0

# Format helper (DE)
def fmt(val, unit=""):
    s = f"{val:,.2f} {unit}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s

str_eeg = fmt(total_eeg, '€')
str_loss_eeg = fmt(loss_eeg, '€')
str_neg_hours = fmt(neg_hours, 'h')
str_loss_kwh = fmt(total_loss, 'kWh')
str_loss_pct = fmt(loss_pct, '%')
str_gen = fmt(total_gen, 'kWh')

# Dashboard Kennzahlen
st.subheader("Wirtschaftlichkeitsanalyse")
st.markdown("**Monetäre Auswertung**")
col1, col2, col3 = st.columns(3)
col1.metric("Gesamtertrag EEG", str_eeg)
col2.metric("Verlust EEG durch Clipping", str_loss_eeg)
col3.metric("Abregelung (neg. Preise)", str_neg_hours)

st.markdown("**Energetische Auswertung**")
col4, col5, col6 = st.columns(3)
col4.metric("Verlust durch Clipping", str_loss_kwh)
col5.metric("Verlust in %", str_loss_pct)
col6.metric("Gesamtertrag (kWh)", str_gen)

# Charts
# Clipping Zeitverlauf
fig1, ax1 = plt.subplots(figsize=(10,4))
mask = pv_kw > max_power_kw
ax1.bar(pv_kw.index, clipped_kw, label='Nach Clipping', color='orange', alpha=0.6)
if mask.any():
    ax1.bar(pv_kw.index[mask], pv_kw[mask]-max_power_kw, bottom=max_power_kw, label='Über Grenze', color='red')
ax1.axhline(max_power_kw, linestyle='--', color='red', label='WR Max')
ax1.set_title('Clipping im Zeitverlauf')
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
ax1.legend()
st.pyplot(fig1)

# Verluste pro Monat
fig2, ax2 = plt.subplots(figsize=(10,4))
mon_loss = lost_kwh.rename('loss').resample('M').sum()
ax2.bar(mon_loss.index, mon_loss.values, width=20, color='salmon')
ax2.set_title('Clipping-Verluste pro Monat')
ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
st.pyplot(fig2)

# Day-Ahead Preise
fig3, ax3 = plt.subplots(figsize=(10,4))
ax3.plot(price_ct.index, price_ct.where(price_ct>=0), color='orange', label='Preis ≥ 0')
ax3.plot(price_ct.index, price_ct.where(price_ct<0), color='red', label='Preis < 0')
ax3.axhline(0, color='black', linestyle='--')
ax3.set_title('Day-Ahead Preise')
ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
ax3.legend()
st.pyplot(fig3)

# PDF Export
if st.button('📄 PDF-Bericht exportieren'):
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig1); pdf.savefig(fig2); pdf.savefig(fig3)
        # Kennzahlen-Seite
        fig_k, axk = plt.subplots(figsize=(8.27,5))
        axk.axis('off')
        text = f"Gesamtertrag EEG: {str_eeg}\nVerlust EEG: {str_loss_eeg}\nAbregelung: {str_neg_hours}"
        axk.text(0.1, 0.6, text, fontsize=12)
        text2 = f"Verlust kWh: {str_loss_kwh}\nVerlust %: {str_loss_pct}\nErtrag: {str_gen}"
        axk.text(0.1, 0.3, text2, fontsize=12)
        pdf.savefig(fig_k)
    buf.seek(0)
    st.download_button('Download PDF', data=buf, file_name='PV_Analyse.pdf', mime='application/pdf')
