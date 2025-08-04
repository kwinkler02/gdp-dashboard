import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO

# --- App Configuration ---
st.set_page_config(page_title="PV Lastgang Analyse", layout="wide")
st.title("PV Lastgang Wirtschaftlichkeitsanalyse mit Clipping und EEG")

# --- Sidebar Inputs ---
pv_file = st.sidebar.file_uploader("PV Lastgang (Viertelstundenwerte in kWh, Format TT.MM HH:MM)", type=["csv", "xlsx"])
price_file = st.sidebar.file_uploader("Day-Ahead Preise (€/MWh, Format TT.MM HH:MM)", type=["csv", "xlsx"])
max_power_kw = st.sidebar.number_input("Wechselrichter Maximalleistung (kW)", min_value=0.0, step=0.1)
eeg_ct_per_kwh = st.sidebar.number_input("EEG-Vergütung (ct/kWh)", min_value=0.0, step=0.1)

# --- Data Loader (einheitliches Format TT.MM HH:MM) ---
@st.cache_data
def load_series(uploaded_file):
    if not uploaded_file:
        return None
    try:
        # Datei einlesen
        df = (pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv')
              else pd.read_excel(uploaded_file))
    except Exception as e:
        st.error(f"Fehler beim Laden der Datei: {e}")
        return None
    # Rohdaten
    raw_ts = df.iloc[:, 0].astype(str).str.strip()
    raw_val = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    # Aktuelles Jahr an Timestamp anhängen
    year = pd.Timestamp.now().year
    full_ts = raw_ts + f" {year}"
    # Datetime parse
    idx = pd.to_datetime(full_ts, format="%d.%m %H:%M %Y", dayfirst=True, errors='coerce')
    series = pd.Series(raw_val.values, index=idx).dropna()
    if series.empty:
        return None
    return series.sort_index()

pv_series = load_series(pv_file)
price_series = load_series(price_file)

if pv_series is None or price_series is None:
    st.info("Bitte lade beide Dateien mit gültigen Zeitstempeln im Format TT.MM HH:MM hoch.")
    st.stop()

# --- Zeitreihen synchronisieren ---
start = max(pv_series.index.min(), price_series.index.min())
end = min(pv_series.index.max(), price_series.index.max())
pv_series = pv_series[start:end]
price_series = price_series[start:end]
st.success(f"Synchronisiert {len(pv_series)} Einträge von {start.date()} bis {end.date()}.")

# --- Berechnungen ---
pv_kw = pv_series * 4
clipped_kw = pv_kw.clip(upper=max_power_kw)
clipped_kwh = clipped_kw / 4
lost_kwh = pv_series - clipped_kwh
price_ct = price_series / 10
eeg_paid = np.where(price_ct >= 0, clipped_kwh * eeg_ct_per_kwh, 0)

total_eeg = eeg_paid.sum() / 100
loss_eeg = (lost_kwh * eeg_ct_per_kwh).sum() / 100
neg_hours = ((pv_series > 0) & (price_ct < 0)).sum() / 4
total_gen = clipped_kwh.sum()
total_loss = lost_kwh.sum()
loss_pct = (total_loss / pv_series.sum() * 100) if pv_series.sum() > 0 else 0

fmt = lambda v, u='': f"{v:,.2f} {u}".replace(',', 'X').replace('.', ',').replace('X', '.')

# --- Kennzahlen ---
st.subheader("Wirtschaftlichkeitsanalyse")
st.markdown("**Monetäre Auswertung**")
c1, c2, c3 = st.columns(3)
c1.metric("Gesamtertrag EEG", fmt(total_eeg, '€'))
c2.metric("Verlust EEG durch Clipping", fmt(loss_eeg, '€'))
c3.metric("Abregelung (neg. Preise)", fmt(neg_hours, 'h'))

st.markdown("**Energetische Auswertung**")
c4, c5, c6 = st.columns(3)
c4.metric("Verlust durch Clipping", fmt(total_loss, 'kWh'))
c5.metric("Verlust in %", fmt(loss_pct, '%'))
c6.metric("Gesamtertrag kWh", fmt(total_gen, 'kWh'))

# --- Helper für Monatsformat auf X-Achse ---
def month_fmt(ax):
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))

# --- Charts ---
# 1) Clipping-Zeitverlauf
fig1, ax1 = plt.subplots(figsize=(10, 4))
mask = pv_kw > max_power_kw
ax1.bar(pv_kw.index, clipped_kw * 4, color='orange', alpha=0.6, label='Nach Clipping')
if mask.any():
    ax1.bar(pv_kw.index[mask], (pv_kw - max_power_kw)[mask], bottom=max_power_kw, color='red', label='Über Grenze')
ax1.axhline(max_power_kw, linestyle='--', color='red', label='WR Max')
ax1.set_title('Clipping im Zeitverlauf')
month_fmt(ax1)
ax1.legend()

# 2) Clipping-Verluste pro Monat
fig2, ax2 = plt.subplots(figsize=(10, 4))
monthly_loss = lost_kwh.resample('M').sum()
ax2.bar(monthly_loss.index, monthly_loss.values, color='salmon', width=20)
ax2.set_title('Clipping-Verluste pro Monat')
month_fmt(ax2)

# 3) Day-Ahead Preise
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(price_ct.index, price_ct.where(price_ct >= 0), color='orange', label='Preis ≥ 0')
ax3.plot(price_ct.index, price_ct.where(price_ct < 0), color='red', label='Preis < 0')
ax3.axhline(0, linestyle='--', color='black', label='Null-Linie')
ax3.set_title('Day-Ahead Preise')
month_fmt(ax3)
ax3.legend()

st.subheader("Visualisierung")
st.pyplot(fig1)
st.pyplot(fig2)
st.pyplot(fig3)

# --- PDF Export ---
if st.button('📄 PDF-Bericht exportieren'):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        # Deckblatt
        fig_cover = plt.figure(figsize=(8, 6)); fig_cover.clf()
        fig_cover.text(0.5, 0.5, 'PV Wirtschaftlichkeitsanalyse', ha='center', va='center', fontsize=18)
        pdf.savefig(fig_cover); plt.close(fig_cover)
        # Kennzahlen
        fig_stats, ax_stats = plt.subplots(figsize=(8, 6)); ax_stats.axis('off')
        stats_text = (
            f"Gesamtertrag EEG: {fmt(total_eeg,'€')}\n"
            f"Verlust EEG Clipping: {fmt(loss_eeg,'€')}\n"
            f"Abregelung neg. Preise: {fmt(neg_hours,'h')}\n"
            f"Verlust kWh: {fmt(total_loss,'kWh')}\n"
            f"Verlust %: {fmt(loss_pct,'%')}\n"
            f"Ertrag kWh: {fmt(total_gen,'kWh')}"
        )
        ax_stats.text(0.1, 0.5, stats_text, fontsize=12)
        pdf.savefig(fig_stats); plt.close(fig_stats)
        # Charts
        for fig in (fig1, fig2, fig3):
            pdf.savefig(fig)
    buffer.seek(0)
    st.download_button('Download PDF', data=buffer, file_name='PV_Analyse.pdf', mime='application/pdf')
