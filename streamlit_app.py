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
    """FIXED: Robust data loading with proper error handling"""
    if file is None:
        return None
    
    try:
        # Read file
        df = pd.read_csv(file, header=0) if file.name.endswith('.csv') else pd.read_excel(file, header=0)
        
        # Extract timestamps and values
        ts_raw = df.iloc[:, 0]
        vals_raw = df.iloc[:, 1]
        
        # Convert values to numeric, handle errors
        vals = pd.to_numeric(vals_raw, errors='coerce')
        
        # FIXED: Normalize all timestamps to year 2025 for consistency
        target_year = 2025
        parsed_timestamps = []
        valid_values = []
        
        for i, (timestamp, value) in enumerate(zip(ts_raw, vals)):
            # Skip invalid entries
            if pd.isna(timestamp) or pd.isna(value):
                continue
            
            try:
                ts_str = str(timestamp).strip()
                parsed_dt = None
                
                # FIXED: Handle multiple timestamp formats
                # Format 1: "01.01.2025 00:00" (with year)
                if len(ts_str) > 10 and ('2024' in ts_str or '2025' in ts_str):
                    try:
                        parsed_dt = pd.to_datetime(ts_str, dayfirst=True)
                        # Normalize to target year
                        if parsed_dt.year != target_year:
                            parsed_dt = parsed_dt.replace(year=target_year)
                    except:
                        pass
                
                # Format 2: "01.01. 00:00" (without year)
                elif ' ' in ts_str:
                    try:
                        date_part, time_part = ts_str.split(' ', 1)
                        if date_part.endswith('.'):
                            # Remove trailing dot and add year
                            day_month = date_part.rstrip('.')
                            if '.' in day_month:
                                day, month = day_month.split('.')
                                full_ts = f"{day.zfill(2)}.{month.zfill(2)}.{target_year} {time_part}"
                                parsed_dt = pd.to_datetime(full_ts, format='%d.%m.%Y %H:%M')
                    except:
                        pass
                
                # Format 3: Legacy parsing (fallback)
                if parsed_dt is None:
                    try:
                        parts = ts_str.split()
                        d = parts[0] if parts else ts_str
                        time_part = parts[1] if len(parts) > 1 else '00:00'
                        
                        dparts = d.split('.')
                        if len(dparts) == 2:
                            dparts.append(str(target_year))
                        elif len(dparts) == 3 and len(dparts[2]) == 2:
                            dparts[2] = '20' + dparts[2]
                        
                        full_ts = '.'.join(dparts) + ' ' + time_part
                        parsed_dt = pd.to_datetime(full_ts, dayfirst=True, infer_datetime_format=True)
                        
                        # Normalize year
                        if parsed_dt.year != target_year:
                            parsed_dt = parsed_dt.replace(year=target_year)
                    except:
                        pass
                
                if parsed_dt is not None:
                    parsed_timestamps.append(parsed_dt)
                    valid_values.append(value)
                    
            except Exception:
                continue
        
        if len(parsed_timestamps) == 0:
            st.error(f"Keine gültigen Zeitstempel gefunden in {file.name}")
            return None
        
        # Create series
        series = pd.Series(valid_values, index=parsed_timestamps)
        series = series.sort_index()
        
        return series
        
    except Exception as e:
        st.error(f"Fehler beim Laden von {file.name}: {str(e)}")
        return None

# FIXED: Load data with error handling
pv_series = load_series(pv_file)
price_series = load_series(price_file)

# FIXED: Check data compatibility before proceeding
if pv_series is not None and price_series is not None:
    # FIXED: Align time series properly
    # Find common time range
    start_time = max(pv_series.index.min(), price_series.index.min())
    end_time = min(pv_series.index.max(), price_series.index.max())
    
    # Filter both series to common time range
    pv_filtered = pv_series[(pv_series.index >= start_time) & (pv_series.index <= end_time)]
    price_filtered = price_series[(price_series.index >= start_time) & (price_series.index <= end_time)]
    
    # FIXED: Robust reindexing with nearest neighbor matching
    common_index = pv_filtered.index.intersection(price_filtered.index)
    
    if len(common_index) == 0:
        # Try nearest neighbor matching
        try:
            price_aligned = price_filtered.reindex(pv_filtered.index, method='nearest', tolerance=pd.Timedelta('15 minutes'))
            valid_mask = ~price_aligned.isna()
            
            if valid_mask.sum() > 0:
                pv_series = pv_filtered[valid_mask]
                price_series = price_aligned[valid_mask]
                st.success(f"Daten synchronisiert: {len(pv_series)} Datenpunkte")
            else:
                st.error("Keine übereinstimmenden Zeitstempel gefunden!")
                st.stop()
        except:
            st.error("Fehler beim Synchronisieren der Zeitreihen!")
            st.stop()
    else:
        # Use exact matches
        pv_series = pv_filtered.reindex(common_index)
        price_series = price_filtered.reindex(common_index)
        st.success(f"Daten geladen: {len(pv_series)} Datenpunkte")
    
    # Clipping und EEG-Berechnung
    pv_kw = pv_series * 4
    clipped_kw = pv_kw.clip(upper=max_power_kw)
    clipped_kwh = clipped_kw / 4
    lost_kwh = pv_series - clipped_kwh

    price_ct = price_series / 10
    eeg_paid = np.where(price_ct > 0, clipped_kwh * eeg_ct_per_kwh, 0)

    # Kennzahlen (monetär)
    total_eeg = eeg_paid.sum() / 100
    loss_eeg = (lost_kwh * eeg_ct_per_kwh).sum() / 100
    negative_hours = ((pv_series > 0) & (price_ct < 0)).sum() / 4

    # Kennzahlen (energetisch)
    total_generated = clipped_kwh.sum()
    total_lost = lost_kwh.sum()
    loss_pct = (total_lost / pv_series.sum() * 100) if pv_series.sum() > 0 else 0

    # Formatierung deutsch
    fmt = lambda x, u="": f"{x:,.2f} {u}".replace(',', 'X').replace('.', ',').replace('X', '.')
    
    st.subheader("Wirtschaftlichkeitsanalyse")
    st.text("Monetäre Auswertung")
    c1, c2, c3 = st.columns(3)
    c1.metric("Gesamtertrag EEG", fmt(total_eeg, '€'))
    c2.metric("Verlust EEG durch Clipping", fmt(loss_eeg, '€'))
    c3.metric("Abregelung (neg. Preise)", fmt(negative_hours, 'h'))

    st.text("Energetische Auswertung")
    c4, c5, c6 = st.columns(3)
    c4.metric("Verlust durch Clipping", fmt(total_lost, 'kWh'))
    c5.metric("Verlust in %", fmt(loss_pct, '%'))
    c6.metric("Gesamtertrag (kWh)", fmt(total_generated, 'kWh'))

    # Charts
    # 1. Clipping Zeitverlauf
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    mask = pv_kw > max_power_kw
    ax1.bar(pv_kw.index, clipped_kw, label='Nach Clipping', color='orange', alpha=0.6)
    if mask.any():  # FIXED: Only plot if there's actual clipping
        ax1.bar(pv_kw.index[mask], pv_kw[mask] - max_power_kw, bottom=max_power_kw,
                label='Über Grenze', color='red')
    ax1.axhline(max_power_kw, linestyle='--', color='red', label='WR Max')
    ax1.set_title('Clipping im Zeitverlauf')
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    ax1.legend()

    # 2. Monatliche Verluste
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    monthly = lost_kwh.resample('M').sum()
    ax2.bar(monthly.index, monthly.values, width=20, color='salmon')
    ax2.set_title('Clipping-Verluste pro Monat')
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))

    # 3. Day-Ahead Preis
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(price_ct.index, price_ct.where(price_ct >= 0), color='orange', label='≥0')
    ax3.plot(price_ct.index, price_ct.where(price_ct < 0), color='red', label='<0')
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
        buf = BytesIO()
        with PdfPages(buf) as pdf:
            for fig in [fig1, fig2, fig3]:
                pdf.savefig(fig, bbox_inches='tight')
        buf.seek(0)
        st.download_button('Download PDF', data=buf, file_name='PV_Analyse.pdf', mime='application/pdf')

elif pv_file is not None or price_file is not None:
    # FIXED: Show which file is missing
    missing = []
    if pv_file is None:
        missing.append("PV-Daten")
    if price_file is None:
        missing.append("Preisdaten")
    st.warning(f"Noch fehlend: {', '.join(missing)}")
else:
    st.info('Bitte beide Dateien hochladen.')
