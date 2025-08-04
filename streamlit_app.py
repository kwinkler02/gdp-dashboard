import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages

# App-Konfiguration
st.set_page_config(page_title="PV Lastgang Analyse", layout="wide")
st.title("PV Lastgang Wirtschaftlichkeitsanalyse mit Clipping und EEG")

# Sidebar: Daten-Upload und Parameter
st.sidebar.header("Daten & Einstellungen")
pv_file = st.sidebar.file_uploader("PV Lastgang (Viertelstundenwerte in kWh)", type=["csv", "xlsx"])
price_file = st.sidebar.file_uploader("Day-Ahead Preise (€/MWh, Viertelstundenwerte)", type=["csv", "xlsx"])

max_power_kw = st.sidebar.number_input("Wechselrichter Maximalleistung (kW)", min_value=0.0, step=0.1)
eeg_ct_per_kwh = st.sidebar.number_input("EEG-Vergütung (ct/kWh)", min_value=0.0, step=0.1)

@st.cache_data
def load_series(file):
    if file is None:
        return None
    # Einlesen ohne index, parse später
    df = pd.read_csv(file, header=0) if file.name.endswith('.csv') else pd.read_excel(file, header=0)
    ts = df.iloc[:,0].astype(str)
    vals = df.iloc[:,1]
    # Ergänze Jahr und parse flexibel
    year = pd.Timestamp.now().year
    parsed = []
    for s in ts:
        parts = s.split()
        d,mY = parts[0], parts[1] if len(parts)>1 else '00:00'
        dparts = d.split('.')
        if len(dparts)==2: dparts.append(str(year))
        if len(dparts)==3 and len(dparts[2])==2: dparts[2]='20'+dparts[2]
        parsed.append('.'.join(dparts)+' '+mY)
    dates = pd.to_datetime(parsed, dayfirst=True, infer_datetime_format=True)
    return pd.Series(vals.values, index=dates)

# Lade Zeitreihen
pv_series = load_series(pv_file)
price_series = load_series(price_file)

if pv_series is not None and price_series is not None:
    # Clipping und EEG-Berechnung
    pv_kw = pv_series * 4
    clipped_kw = pv_kw.clip(upper=max_power_kw)
    clipped_kwh = clipped_kw / 4
    lost_kwh = pv_series - clipped_kwh

    price_ct = price_series / 10
    eeg_paid = np.where(price_ct>0, clipped_kwh * eeg_ct_per_kwh, 0)

    # Kennzahlen (monetär)
    total_eeg = eeg_paid.sum()/100
    loss_eeg = (lost_kwh * eeg_ct_per_kwh).sum()/100
    negative_hours = ((pv_series>0)&(price_ct<0)).sum()/4

    # Kennzahlen (energetisch)
    total_generated = clipped_kwh.sum()
    total_lost = lost_kwh.sum()
    loss_pct = (total_lost/ pv_series.sum()*100) if pv_series.sum()>0 else 0

    # Formatierung deutsch
    fmt = lambda x, u="": f"{x:,.2f} {u}".replace(',','X').replace('.',',').replace('X','.')
    st.subheader("Wirtschaftlichkeitsanalyse")
    st.text("Monetäre Auswertung")
    c1,c2,c3 = st.columns(3)
    c1.metric("Gesamtertrag EEG", fmt(total_eeg,'€'))
    c2.metric("Verlust EEG durch Clipping", fmt(loss_eeg,'€'))
    c3.metric("Abregelung (neg. Preise)", fmt(negative_hours,'h'))

    st.text("Energetische Auswertung")
    c4,c5,c6 = st.columns(3)
    c4.metric("Verlust durch Clipping", fmt(total_lost,'kWh'))
    c5.metric("Verlust in %", fmt(loss_pct,'%'))
    c6.metric("Gesamtertrag (kWh)", fmt(total_generated,'kWh'))

    # Charts
    # 1. Clipping Zeitverlauf
    fig1, ax1 = plt.subplots(figsize=(10,4))
    mask = pv_kw>max_power_kw
    ax1.bar(pv_kw.index, clipped_kw, label='Nach Clipping', color='orange', alpha=0.6)
    ax1.bar(pv_kw.index[mask], pv_kw[mask]-max_power_kw, bottom=max_power_kw,
            label='Über Grenze', color='red')
    ax1.axhline(max_power_kw, linestyle='--', color='red', label='WR Max')
    ax1.set_title('Clipping im Zeitverlauf')
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    ax1.legend()

    # 2. Monatliche Verluste
    fig2, ax2 = plt.subplots(figsize=(10,4))
    monthly = lost_kwh.resample('M').sum()
    ax2.bar(monthly.index, monthly.values, width=20, color='salmon')
    ax2.set_title('Clipping-Verluste pro Monat')
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))

    # 3. Day-Ahead Preis
    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.plot(price_ct.index, price_ct.where(price_ct>=0), color='orange', label='≥0')
    ax3.plot(price_ct.index, price_ct.where(price_ct<0), color='red', label='<0')
    ax3.axhline(0, linestyle='--', color='black')
    ax3.set_title('Day-Ahead Preise')
    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    ax3.legend()

    st.subheader("Visualisierung")
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)

    # PDF Export
    if st.button('PDF-Bericht exportieren'):
        buf=BytesIO()
        with PdfPages(buf) as pdf:
            for fig in [fig1, fig2, fig3]:
                pdf.savefig(fig)
        buf.seek(0)
        st.download_button('Download PDF', data=buf, file_name='PV_Analyse.pdf', mime='application/pdf')
else:
    st.info('Bitte beide Dateien hochladen.')
