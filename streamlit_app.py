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
        # Parse timestamps flexibel ohne festes Format
    dates = pd.to_datetime(parsed, dayfirst=True, infer_datetime_format=True)
    series = pd.Series(vals.values, index=dates)
    series.name = vals.name or 'value'
    return series
