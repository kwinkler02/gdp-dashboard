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

# Hilfsfunktion zum Laden der Zeitreihen
@st.cache_data
def load_data(file, default_year=None):
    if file is None:
        return None
    # Einlesen und erste Spalte als Index
    if file.name.endswith('.csv'):
        df = pd.read_csv(file, index_col=0)
    else:
        df = pd.read_excel(file, index_col=0)
    # Index: versuchen zu datetime zu konvertieren
    idx = df.index.astype(str)
    # Füge Jahr hinzu, falls nicht vorhanden (z.B. 'DD.MM HH:MM')
    # Ersetze Varianten mit oder ohne Jahr
    sample = idx[0]
    if sample.count('.') == 1:
        year = default_year or pd.Timestamp.now().year
        idx = idx + f'.{year}'
        df.index = pd.to_datetime(idx, dayfirst=True, format='%d.%m.%Y %H:%M')
    else:
        # Standard-Pandasto_datetime mit Tag.Monat.Jahr optional
        df.index = pd.to_datetime(idx, dayfirst=True, infer_datetime_format=True)
    return df

# Daten laden
current_year = pd.Timestamp.now().year
pv_data = load_data(pv_file, default_year=current_year)
price_data = load_data(price_file, default_year=current_year)

# ... verbleibender Code unverändert folgt ab hier # Daten laden
pv_data = load_data(pv_file)
price_data = load_data(price_file)

if pv_data is None or price_data is None:
    st.info("Bitte lade beide Dateien hoch, um die Analyse zu starten.")
    st.stop()

# Kopfzeilen anzeigen
st.subheader("Datenübersicht")
st.write("PV Lastgang (kWh pro 15 Minuten):", pv_data.head())
st.write("Day-Ahead Preise (€/MWh):", price_data.head())

# Umrechnung: 15-Minuten kWh * 4 = kW Leistung
pv_power_kw = pv_data.iloc[:, 0] * 4
clipped_power_kw = np.minimum(pv_power_kw, max_power_kw)
clipped_energy_kwh = clipped_power_kw / 4
lost_energy_kwh = pv_data.iloc[:, 0] - clipped_energy_kwh

# Preise in ct/kWh
price_ct_per_kwh = price_data.iloc[:, 0] / 10

# EEG-Einnahmen und Kennzahlen
eeg_paid = np.where(price_ct_per_kwh > 0, clipped_energy_kwh * eeg_ct_per_kwh, 0)
total_eeg_revenue = eeg_paid.sum() / 100
lost_eeg_revenue = (lost_energy_kwh * eeg_ct_per_kwh).sum() / 100
curtailed_hours = ((pv_data.iloc[:, 0] > 0) & (price_ct_per_kwh < 0)).sum() / 4

total_pv_energy = pv_data.iloc[:, 0].sum()
total_lost_energy = lost_energy_kwh.sum()
total_generated_energy = clipped_energy_kwh.sum()
lost_energy_pct = (total_lost_energy / total_pv_energy * 100) if total_pv_energy > 0 else 0

# Formatierte Strings (DE)
fmt = lambda x, u="": f"{x:,.2f} {u}".replace(",", "X").replace(".", ",").replace("X", ".")
str_eeg = fmt(total_eeg_revenue, "€")
str_loss_eeg = fmt(lost_eeg_revenue, "€")
str_hours = fmt(curtailed_hours, "h")
str_loss_kwh = fmt(total_lost_energy, "kWh")
str_pct = fmt(lost_energy_pct, "%")
str_gen = fmt(total_generated_energy, "kWh")

# Wirtschaftlichkeitsanalyse
st.subheader("Wirtschaftlichkeitsanalyse")
st.markdown("**Monetäre Auswertung**")
c1, c2, c3 = st.columns(3)
c1.metric("Gesamtertrag EEG", str_eeg)
c2.metric("Verlust EEG durch Clipping", str_loss_eeg)
c3.metric("Abregelung (neg. Preise)", str_hours)

st.markdown("**Energetische Auswertung**")
c4, c5, c6 = st.columns(3)
c4.metric("Verlust durch Clipping", str_loss_kwh)
c5.metric("Verlust in %", str_pct)
c6.metric("Gesamtertrag (kWh)", str_gen)

# Charts erzeugen
# 1) Clipping Zeitverlauf
fig1, ax1 = plt.subplots(figsize=(10, 4))
mask = pv_power_kw > max_power_kw
ax1.bar(pv_power_kw.index, clipped_power_kw, label='Nach Clipping', color='orange', alpha=0.6)
ax1.bar(pv_power_kw.index[mask], pv_power_kw[mask] - max_power_kw,
        bottom=max_power_kw, label='Über Grenze', color='red')
ax1.axhline(max_power_kw, linestyle='--', color='red', label='WR Max')
ax1.set_title('Clipping im Zeitverlauf')
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
ax1.set_ylabel('Leistung in kW')
ax1.legend()

# 2) Verluste pro Monat
fig2, ax2 = plt.subplots(figsize=(10, 4))
monthly_losses = lost_energy_kwh.resample('M').sum()
ax2.bar(monthly_losses.index, monthly_losses.values, width=20, color='salmon')
ax2.set_title('Clipping-Verluste pro Monat')
ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
ax2.set_ylabel('Verlust in kWh')

# 3) Day-Ahead Preise
fig3, ax3 = plt.subplots(figsize=(10, 4))
pos = price_ct_per_kwh.where(price_ct_per_kwh >= 0)
neg = price_ct_per_kwh.where(price_ct_per_kwh < 0)
ax3.plot(pos.index, pos, color='orange', label='Preis ≥ 0')
ax3.plot(neg.index, neg, color='red', label='Preis < 0')
ax3.axhline(0, color='black', linestyle='--', linewidth=1, label='Null-Linie')
ax3.set_title('Day-Ahead Preise')
ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
ax3.set_ylabel('Preis in ct/kWh')
ax3.legend()

# Übersicht
st.subheader('Clipping-Analyse Visualisierung')
st.pyplot(fig1)
st.pyplot(fig2)
st.pyplot(fig3)

# PDF Export
if st.button('📄 PDF-Bericht exportieren'):
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Deckblatt
        fig_cover = plt.figure(figsize=(8.27, 11.69))
        fig_cover.clf()
        fig_cover.text(0.5, 0.75, 'Wirtschaftlichkeitsanalyse – PV Clipping', ha='center', va='center', fontsize=20)
        fig_cover.text(0.5, 0.7, 'Erstellt: ' + pd.Timestamp.now().strftime('%d.%m.%Y'), ha='center', va='center', fontsize=12)
        pdf.savefig(fig_cover); plt.close(fig_cover)

        # Monetäre Auswertung
        fig_monetary, axm = plt.subplots(figsize=(8.27, 5))
        axm.axis('off')
        text1 = f"Gesamtertrag EEG: {str_eeg}\nVerlust EEG: {str_loss_eeg}\nAbregelung: {str_hours}"
        axm.text(0.1, 0.5, text1, fontsize=12)
        pdf.savefig(fig_monetary); plt.close(fig_monetary)

        # Energetische Auswertung
        fig_energy, axe = plt.subplots(figsize=(8.27, 5))
        axe.axis('off')
        text2 = f"Verlust kWh: {str_loss_kwh}\nVerlust %: {str_pct}\nErtrag: {str_gen}"
        axe.text(0.1, 0.5, text2, fontsize=12)
        pdf.savefig(fig_energy); plt.close(fig_energy)

        # Charts
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)
    buf.seek(0)
    st.download_button('Download PDF', data=buf, file_name='PV_Wirtschaftlichkeitsanalyse.pdf', mime='application/pdf')
