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
pv_file = st.sidebar.file_uploader("PV Lastgang (Viertelstundenwerte in kWh)", type=["csv", "xlsx"])
price_file = st.sidebar.file_uploader("Day-Ahead Preise (€/MWh, Viertelstundenwerte)", type=["csv", "xlsx"])

# --- Parameter ---
max_power_kw = st.sidebar.number_input("Wechselrichter Maximalleistung (kW)", min_value=0.0, step=0.1)
eeg_ct_per_kwh = st.sidebar.number_input("EEG-Vergütung (ct/kWh)", min_value=0.0, step=0.1)

# --- Daten laden ---
def load_data(file):
    if file:
        if file.name.endswith('.csv'):
            return pd.read_csv(file, parse_dates=True, index_col=0)
        return pd.read_excel(file, parse_dates=True, index_col=0)
    return None

pv_data = load_data(pv_file)
price_data = load_data(price_file)

if pv_data is not None and price_data is not None:
    # Umrechnung & Clipping
    pv_kw = pv_data.iloc[:,0] * 4
    clipped_kw = np.minimum(pv_kw, max_power_kw)
    clipped_kwh = clipped_kw / 4
    lost_kwh = pv_data.iloc[:,0] - clipped_kwh

    price_ct = price_data.iloc[:,0] / 10
    eeg_paid = np.where(price_ct>0, clipped_kwh*eeg_ct_per_kwh, 0)

    # Kennzahlen
    total_eeg = eeg_paid.sum()/100
    loss_eeg = (lost_kwh*eeg_ct_per_kwh).sum()/100
    hours_curtailed = ((pv_data.iloc[:,0]>0)&(price_ct<0)).sum()/4
    total_generated_kwh = clipped_kwh.sum()
    total_lost_kwh = lost_kwh.sum()
    lost_pct = total_lost_kwh/ pv_data.iloc[:,0].sum()*100

    # Formatierte Strings (DE)
    fmt = lambda x,unit="": f"{x:,.2f} {unit}".replace(",","X").replace(".",",").replace("X",".")
    str_eeg=fmt(total_eeg,"€")
    str_loss_eeg=fmt(loss_eeg,"€")
    str_hours=fmt(hours_curtailed,"h")
    str_loss_kwh=fmt(total_lost_kwh,"kWh")
    str_pct=fmt(lost_pct," %")
    str_gen=fmt(total_generated_kwh,"kWh")

    # Anzeige Kennzahlen
    st.subheader("Wirtschaftlichkeitsanalyse")
    st.markdown("**Monetäre Auswertung**")
    c1,c2,c3 = st.columns(3)
    c1.metric("Gesamtertrag EEG",str_eeg)
    c2.metric("Verlust EEG durch Clipping",str_loss_eeg)
    c3.metric("Abregelung (neg. Preise)",str_hours)

    st.markdown("**Energetische Auswertung**")
    c4,c5,c6 = st.columns(3)
    c4.metric("Verlust durch Clipping",str_loss_kwh)
    c5.metric("Verlust in %",str_pct)
    c6.metric("Gesamtertrag (kWh)",str_gen)

    # Charts erstellen
    # 1 Clipping Zeitverlauf
    fig1,ax1=plt.subplots(figsize=(10,4))
    mask=pv_kw>max_power_kw
    ax1.bar(pv_kw.index,clipped_kw,label='Nach Clipping',color='orange',alpha=0.6)
    ax1.bar(pv_kw.index[mask],pv_kw[mask]-max_power_kw,bottom=max_power_kw,label='Über Grenze',color='red')
    ax1.axhline(max_power_kw,linestyle='--',color='red',label='WR Max')
    ax1.set_title('Clipping im Zeitverlauf')
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    ax1.legend()

    # 2 Verlust pro Monat
    fig2,ax2=plt.subplots(figsize=(10,4))
    losses=lost_kwh.resample('M').sum()
    ax2.bar(losses.index,losses.values,width=20,color='salmon')
    ax2.set_title('Clipping-Verluste pro Monat')
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))

    # 3 Day-Ahead Preis
    fig3,ax3=plt.subplots(figsize=(10,4))
    ax3.plot(price_ct.index,price_ct.where(price_ct>=0),color='orange',label='Preis≥0')
    ax3.plot(price_ct.index,price_ct.where(price_ct<0),color='red',label='Preis<0')
    ax3.axhline(0,linestyle='--',color='black')
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
        buf=BytesIO()
        with PdfPages(buf) as pdf:
            # Deckblatt
            fcover=plt.figure(figsize=(8.27,11.69));fcover.clf()
            fcover.text(0.5,0.75,'Wirtschaftlichkeitsanalyse – PV Clipping',ha='center',va='center',fontsize=20)
            fcover.text(0.5,0.7,'Erstellt: '+pd.Timestamp.now().strftime('%d.%m.%Y'),ha='center',va='center',fontsize=12)
            pdf.savefig(fcover);plt.close(fcover)
            # Kennzahlen
            fmon,axm=plt.subplots(figsize=(8.27,5));axm.axis('off')
            axm.text(0.1,0.5,f"Gesamtertrag EEG: {str_eeg}\nVerlust EEG: {str_loss_eeg}\nAbregelung: {str_hours}",fontsize=12)
            pdf.savefig(fmon);plt.close(fmon)
            fene,axe=plt.subplots(figsize=(8.27,5));axe.axis('off')
            axe.text(0.1,0.5,f"Verlust kWh: {str_loss_kwh}\nVerlust %: {str_pct}\nErtrag: {str_gen}",fontsize=12)
            pdf.savefig(fene);plt.close(fene)
            # Charts
            pdf.savefig(fig1);pdf.savefig(fig2);pdf.savefig(fig3)
        buf.seek(0)
        st.download_button('Download PDF',data=buf,file_name='PV_Analyse.pdf',mime='application/pdf')
else:
    st.info('Bitte lade Daten hoch.')
