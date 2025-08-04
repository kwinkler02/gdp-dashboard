import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO

st.set_page_config(page_title="PV Lastgang Analyse", layout="wide")
st.title("PV Lastgang Wirtschaftlichkeitsanalyse mit Clipping und EEG")

# --- Datei-Upload ---
st.sidebar.header("Daten eingeben")
pv_file = st.sidebar.file_uploader("PV Lastgang (Zeitstempel + kWh)", type=["csv", "xlsx"])
price_file = st.sidebar.file_uploader("Day-Ahead Preise (Zeitstempel + €/MWh)", type=["csv", "xlsx"])

# --- Parameter ---
max_power_kw = st.sidebar.number_input("Wechselrichter Maximalleistung (kW)", min_value=0.0, step=0.1)
eeg_ct_per_kwh = st.sidebar.number_input("EEG-Vergütung (ct/kWh)", min_value=0.0, step=0.1)

# --- Daten laden und verarbeiten ---
def load_series(file):
    if file:
        # Einlesen der ersten beiden Spalten
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, usecols=[0,1], header=0)
        else:
            df = pd.read_excel(file, usecols=[0,1], header=0)
        # Zeitstempel parsen
        df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], errors='coerce')
        df = df.dropna(subset=[df.columns[0]])
        series = pd.Series(df.iloc[:,1].values, index=df.iloc[:,0])
        # Sicherstellen datetime Index und sortieren
        series.index = pd.to_datetime(series.index)
        series = series.sort_index()
        series.index.name = None
        return series
    return None
    return None
    return None

pv_kwh = load_series(pv_file)  # kWh pro 15min
price_mwh = load_series(price_file)  # €/MWh pro 15min

if pv_kwh is not None and price_mwh is not None:
    # Leistung in kW
    pv_kw = pv_kwh * 4
    clipped_kw = np.minimum(pv_kw, max_power_kw)
    clipped_kwh = clipped_kw / 4
    lost_kwh = pv_kwh - clipped_kwh

        # Preis in ct/kWh
    # Auf gleichen Index wie PV-KW bringen (fehlende Preise = NaN)
    price_ct = price_mwh.reindex(pv_kw.index)
    # in ct/kWh umrechnen
    price_ct = price_ct / 10
    # NaN-Preise auf 0 setzen
    price_ct = price_ct.fillna(0)

    # EEG-Einnahmen
    eeg_paid = np.where(price_ct > 0, clipped_kwh * eeg_ct_per_kwh, 0)

    # Kennzahlen
    total_eeg = eeg_paid.sum() / 100
    loss_eeg = (lost_kwh * eeg_ct_per_kwh).sum() / 100
    hours_curtailed = ((pv_kwh > 0) & (price_ct < 0)).sum() / 4
    total_generated_kwh = clipped_kwh.sum()
    total_lost_kwh = lost_kwh.sum()
    lost_pct = total_lost_kwh / pv_kwh.sum() * 100

    # Formatierte Strings (deutsch)
    fmt = lambda x,unit="": f"{x:,.2f} {unit}".replace(",","X").replace(".",",").replace("X",".")
    str_eeg = fmt(total_eeg, "€")
    str_loss_eeg = fmt(loss_eeg, "€")
    str_hours = fmt(hours_curtailed, "h")
    str_loss_kwh = fmt(total_lost_kwh, "kWh")
    str_pct = fmt(lost_pct, "%")
    str_gen = fmt(total_generated_kwh, "kWh")

    # Anzeige Kennzahlen
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
    fig1, ax1 = plt.subplots(figsize=(10,4))
    mask = pv_kw > max_power_kw
    ax1.bar(pv_kw.index, clipped_kw, label='Nach Clipping', color='orange', alpha=0.6)
    ax1.bar(pv_kw.index[mask], pv_kw[mask] - max_power_kw, bottom=max_power_kw, label='Über Grenze', color='red')
    ax1.axhline(max_power_kw, linestyle='--', color='red', label='WR Max')
    ax1.set_title('Clipping im Zeitverlauf')
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(10,4))
            # Monatliche Aggregation
        monthly_losses = lost_kwh.resample('M').sum()
    ax2.bar(monthly_losses.index, monthly_losses.values, width=20, color='salmon')
    ax2.set_title('Clipping-Verluste pro Monat')
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))

    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.plot(price_ct.index, price_ct.where(price_ct >= 0), color='orange', label='Preis ≥ 0')
    ax3.plot(price_ct.index, price_ct.where(price_ct < 0), color='red', label='Preis < 0')
    ax3.axhline(0, linestyle='--', color='black')
    ax3.set_title('Day-Ahead Preise')
    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    ax3.legend()

    # Anzeige Charts
    st.subheader('Clipping-Analyse Visualisierung')
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)

    # PDF Export
    if st.button('📄 PDF-Bericht exportieren'):
        buf = BytesIO()
        with PdfPages(buf) as pdf:
            # Deckblatt
            fcover = plt.figure(figsize=(8.27,11.69)); fcover.clf()
            fcover.text(0.5,0.75,'Wirtschaftlichkeitsanalyse – PV Clipping', ha='center', va='center', fontsize=20)
            fcover.text(0.5,0.7,'Erstellt: ' + pd.Timestamp.now().strftime('%d.%m.%Y'), ha='center', va='center', fontsize=12)
            pdf.savefig(fcover); plt.close(fcover)
            # Kennzahlen Monetär
            fmon, axm = plt.subplots(figsize=(8.27,5)); axm.axis('off')
            axm.text(0.1,0.5,f"Gesamtertrag EEG: {str_eeg}\nVerlust EEG: {str_loss_eeg}\nAbregelung: {str_hours}", fontsize=12)
            pdf.savefig(fmon); plt.close(fmon)
            # Kennzahlen Energetisch
            fene, axe = plt.subplots(figsize=(8.27,5)); axe.axis('off')
            axe.text(0.1,0.5,f"Verlust kWh: {str_loss_kwh}\nVerlust %: {str_pct}\nErtrag: {str_gen}", fontsize=12)
            pdf.savefig(fene); plt.close(fene)
            # Charts
            pdf.savefig(fig1); pdf.savefig(fig2); pdf.savefig(fig3)
        buf.seek(0)
        st.download_button('Download PDF', data=buf, file_name='PV_Analyse.pdf', mime='application/pdf')
else:
    st.info('Bitte lade Daten hoch.')
