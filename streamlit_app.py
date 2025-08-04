import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO

# App-Konfiguration
st.set_page_config(page_title="PV Lastgang Analyse", layout="wide")
st.title("PV Lastgang Wirtschaftlichkeitsanalyse mit Clipping und EEG")

# Sidebar: Daten-Upload und Parameter
st.sidebar.header("Daten & Einstellungen")
pv_file = st.sidebar.file_uploader("PV Lastgang (Viertelstundenwerte in kWh)", type=["csv", "xlsx"])
price_file = st.sidebar.file_uploader("Day-Ahead Preise (€/MWh, Viertelstundenwerte)", type=["csv", "xlsx"])

# Parameter-Eingaben
max_power_kw = st.sidebar.number_input("Wechselrichter Maximalleistung (kW)", min_value=0.0, step=0.1)
eeg_ct_per_kwh = st.sidebar.number_input("EEG-Vergütung (ct/kWh)", min_value=0.0, step=0.1)

@st.cache_data
# Lade Zeitreihe und parse Zeitstempel ohne Jahr: füge aktuelles Jahr hinzu
def load_series(file):
    if file is None:
        return None
    # Einlesen der Rohdaten
    if file.name.endswith('.csv'):
        df = pd.read_csv(file, header=0)
    else:
        df = pd.read_excel(file, header=0)
    # Erste Spalte als Timestamp-Strings, zweite Spalte als Werte
    ts = df.iloc[:, 0].astype(str)
    vals = df.iloc[:, 1]
        # Konvertiere Datumsstrings:
    current_year = pd.Timestamp.now().year
    parsed = []
    for s in ts:
        parts = s.split()
        date_str = parts[0]
        time_str = parts[1] if len(parts) > 1 else '00:00'
        d = date_str.split('.')
        # Tag.Monat ohne Jahr
        if len(d) == 2:
            d.append(str(current_year))
        # zweistelliges Jahr -> vierstellig
        elif len(d) == 3 and len(d[2]) == 2:
            d[2] = '20' + d[2]
        # sonst unverändert
        new_date = '.'.join(d) + ' ' + time_str
        parsed.append(new_date)
    # Einheitliches Format: %d.%m.%Y %H:%M
    dates = pd.to_datetime(parsed, format='%d.%m.%Y %H:%M', dayfirst=True)
    series = pd.Series(vals.values, index=dates)
    series.name = vals.name or 'value'
    return series

# Daten laden
pv_series = load_series(pv_file)
price_series = load_series(price_file)
if pv_series is None or price_series is None:
    st.info("Bitte lade beide Dateien hoch, um die Analyse zu starten.")
    st.stop()

# Umrechnung Energie → Leistung
pv_power_kw = pv_series * 4
clipped_power_kw = np.minimum(pv_power_kw, max_power_kw)
clipped_kwh = clipped_power_kw / 4
lost_kwh = pv_series - clipped_kwh

# Preise in ct/kWh
price_ct = price_series / 10

# Kennzahlen berechnen
# EEG-Ertrag
eeg_paid = np.where(price_ct > 0, clipped_kwh * eeg_ct_per_kwh, 0)
total_eeg = eeg_paid.sum() / 100
loss_eeg = (lost_kwh * eeg_ct_per_kwh).sum() / 100
# Abregelungsstunden
curtailed_hours = ((pv_series > 0) & (price_ct < 0)).sum() / 4
# Energetik
total_lost = lost_kwh.sum()
total_gen = clipped_kwh.sum()
perc_lost = (total_lost / pv_series.sum() * 100) if pv_series.sum() > 0 else 0

# Format DE
fmt = lambda val, unit: f"{val:,.2f} {unit}".replace(",", "X").replace(".", ",").replace("X", ".")
str_total_eeg = fmt(total_eeg, '€')
str_loss_eeg = fmt(loss_eeg, '€')
str_curtailed = fmt(curtailed_hours, 'h')
str_lost = fmt(total_lost, 'kWh')
str_perc = fmt(perc_lost, '%')
str_gen = fmt(total_gen, 'kWh')

# Display
st.subheader("Wirtschaftlichkeitsanalyse")
st.markdown("**Monetäre Auswertung**")
col1, col2, col3 = st.columns(3)
col1.metric("Gesamtertrag EEG", str_total_eeg)
col2.metric("Verlust EEG durch Clipping", str_loss_eeg)
col3.metric("Abregelung (neg. Preise)", str_curtailed)

st.markdown("**Energetische Auswertung**")
col4, col5, col6 = st.columns(3)
col4.metric("Verlust durch Clipping", str_lost)
col5.metric("Verlust in %", str_perc)
col6.metric("Gesamtertrag (kWh)", str_gen)

# Charts
# 1) Clipping Zeitverlauf
fig1, ax1 = plt.subplots(figsize=(10, 4))
mask = pv_power_kw > max_power_kw
ax1.bar(pv_power_kw.index, clipped_power_kw, color='orange', alpha=0.6, label='Nach Clipping')
ax1.bar(pv_power_kw.index[mask], pv_power_kw[mask] - max_power_kw,
        bottom=max_power_kw, color='red', label='Über Grenze')
ax1.axhline(max_power_kw, color='red', linestyle='--', label='WR Max')
ax1.set_title('Clipping im Zeitverlauf')
ax1.set_ylabel('Leistung [kW]')
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
ax1.legend()

# 2) Verluste pro Monat
loss_month = lost_kwh.resample('M').sum()
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.bar(loss_month.index, loss_month.values, width=20, color='salmon')
ax2.set_title('Clipping-Verluste pro Monat')
ax2.set_ylabel('Verlust [kWh]')
ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))

# 3) Day-Ahead Preis
pos = price_ct.where(price_ct >= 0)
neg = price_ct.where(price_ct < 0)
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(pos.index, pos, color='orange', label='Preis ≥ 0')
ax3.plot(neg.index, neg, color='red', label='Preis < 0')
ax3.axhline(0, color='black', linestyle='--', label='0-Linie')
ax3.set_title('Day-Ahead Preise')
ax3.set_ylabel('Preis [ct/kWh]')
ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
ax3.legend()

# Ausgabe
st.subheader('Clipping-Analyse Visualisierung')
st.pyplot(fig1)
st.pyplot(fig2)
st.pyplot(fig3)

# PDF Export
def export_pdf():
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Cover
        fig_cover = plt.figure(figsize=(8.27,11.69))
        fig_cover.clf()
        fig_cover.text(0.5,0.75,'PV Wirtschaftlichkeitsanalyse',ha='center',va='center',fontsize=20)
        fig_cover.text(0.5,0.7,'Erstellt: '+pd.Timestamp.now().strftime('%d.%m.%Y'),ha='center',va='center',fontsize=12)
        pdf.savefig(fig_cover); plt.close(fig_cover)
        # Kennzahlen
        fig_k, axk = plt.subplots(figsize=(8.27,5)); axk.axis('off')
        txt = f"Ertrag EEG: {str_total_eeg}\nVerlust EEG: {str_loss_eeg}\nAbregelung: {str_curtailed}"
        axk.text(0.1,0.5,txt,fontsize=12)
        pdf.savefig(fig_k); plt.close(fig_k)
        fig_e, axe = plt.subplots(figsize=(8.27,5)); axe.axis('off')
        txt2 = f"Verlust: {str_lost}\nVerlust %: {str_perc}\nErtrag: {str_gen}"
        axe.text(0.1,0.5,txt2,fontsize=12)
        pdf.savefig(fig_e); plt.close(fig_e)
        # Charts
        pdf.savefig(fig1); pdf.savefig(fig2); pdf.savefig(fig3)
    buf.seek(0)
    return buf

if st.button('📄 PDF exportieren'):
    pdf_buf = export_pdf()
    st.download_button('Download PDF', data=pdf_buf, file_name='PV_Analyse.pdf', mime='application/pdf')
