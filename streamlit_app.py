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
max_power_kw = st.sidebar.number_input("Wechselrichter Maximalleistung (kW)", min_value=0.0, step=0.1, value=10.0)
eeg_ct_per_kwh = st.sidebar.number_input("EEG-Vergütung (ct/kWh)", min_value=0.0, step=0.1, value=8.2)

@st.cache_data
def load_series(file):
    """Lade Zeitreihe und parse Zeitstempel"""
    if file is None:
        return None
    
    try:
        # Einlesen der Rohdaten
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, header=0)
        else:
            df = pd.read_excel(file, header=0)
        
        # Erste Spalte als Timestamp-Strings, zweite Spalte als Werte
        ts = df.iloc[:, 0].astype(str)
        vals = df.iloc[:, 1]
        
        # Konvertiere Datumsstrings flexibler
        current_year = pd.Timestamp.now().year
        parsed = []
        
        for s in ts:
            try:
                # Verschiedene Formate versuchen
                if 'NaN' in s or 'nan' in s or s == 'nan':
                    continue
                    
                # Format: "01.01. 00:00" oder "01.01.2025 00:00"
                if ' ' in s:
                    date_part, time_part = s.split(' ', 1)
                else:
                    date_part, time_part = s, '00:00'
                
                # Datum parsen
                if date_part.endswith('.'):
                    # Format: "01.01."
                    date_part = date_part[:-1] + f".{current_year}"
                elif date_part.count('.') == 1:
                    # Format: "01.01"
                    date_part = date_part + f".{current_year}"
                
                # Vollständigen Zeitstempel erstellen
                full_timestamp = f"{date_part} {time_part}"
                parsed.append(full_timestamp)
                
            except Exception as e:
                st.warning(f"Fehler beim Parsen von Zeitstempel '{s}': {e}")
                continue
        
        # Parse timestamps
        dates = pd.to_datetime(parsed, format='%d.%m.%Y %H:%M', errors='coerce')
        
        # Entferne NaT Werte
        valid_mask = ~dates.isna()
        dates = dates[valid_mask]
        vals = vals[valid_mask]
        
        series = pd.Series(vals.values, index=dates)
        series.name = vals.name or 'value'
        return series
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Datei: {e}")
        return None

def calculate_clipping_loss(pv_series, max_power_kw):
    """Berechne Clipping-Verluste"""
    if pv_series is None:
        return None, None
    
    # Konvertiere kWh in kW (Viertelstundenwerte * 4)
    power_kw = pv_series * 4
    
    # Clipping anwenden
    clipped_power = power_kw.clip(upper=max_power_kw)
    clipping_loss = power_kw - clipped_power
    
    # Zurück zu kWh
    clipped_energy = clipped_power / 4
    clipping_loss_energy = clipping_loss / 4
    
    return clipped_energy, clipping_loss_energy

def calculate_economics(pv_series, price_series, eeg_ct_per_kwh):
    """Berechne Wirtschaftlichkeit"""
    if pv_series is None or price_series is None:
        return None
    
    # Zeitreihen synchronisieren
    common_index = pv_series.index.intersection(price_series.index)
    if len(common_index) == 0:
        st.error("Keine übereinstimmenden Zeitstempel zwischen PV und Preisdaten gefunden!")
        return None
    
    pv_aligned = pv_series.reindex(common_index, fill_value=0)
    prices_aligned = price_series.reindex(common_index, fill_value=0)
    
    # Preise von €/MWh zu ct/kWh
    prices_ct_per_kwh = prices_aligned / 10
    
    # EEG-Vergütung
    eeg_revenue = pv_aligned * eeg_ct_per_kwh
    
    # Markterlöse
    market_revenue = pv_aligned * prices_ct_per_kwh
    
    # Opportunitätskosten (entgangene Markterlöse durch EEG)
    opportunity_cost = market_revenue - eeg_revenue
    
    results = {
        'pv_energy': pv_aligned,
        'prices': prices_ct_per_kwh,
        'eeg_revenue': eeg_revenue,
        'market_revenue': market_revenue,
        'opportunity_cost': opportunity_cost,
        'total_pv_kwh': pv_aligned.sum(),
        'total_eeg_revenue': eeg_revenue.sum(),
        'total_market_revenue': market_revenue.sum(),
        'total_opportunity_cost': opportunity_cost.sum()
    }
    
    return results

# Hauptprogramm
if pv_file is not None and price_file is not None:
    # Daten laden
    pv_data = load_series(pv_file)
    price_data = load_series(price_file)
    
    if pv_data is not None and price_data is not None:
        st.success(f"PV-Daten geladen: {len(pv_data)} Datenpunkte")
        st.success(f"Preisdaten geladen: {len(price_data)} Datenpunkte")
        
        # Clipping-Analyse
        if max_power_kw > 0:
            clipped_pv, clipping_loss = calculate_clipping_loss(pv_data, max_power_kw)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ursprüngliche PV-Erzeugung", f"{pv_data.sum():.1f} kWh")
                st.metric("Nach Clipping", f"{clipped_pv.sum():.1f} kWh")
            with col2:
                st.metric("Clipping-Verluste", f"{clipping_loss.sum():.1f} kWh")
                st.metric("Verlust in %", f"{(clipping_loss.sum()/pv_data.sum()*100):.1f}%")
            
            # Für Wirtschaftlichkeitsanalyse geclippte Daten verwenden
            analysis_pv = clipped_pv
        else:
            analysis_pv = pv_data
        
        # Wirtschaftlichkeitsanalyse
        economics = calculate_economics(analysis_pv, price_data, eeg_ct_per_kwh)
        
        if economics is not None:
            st.header("Wirtschaftlichkeitsanalyse")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Gesamt PV-Erzeugung", f"{economics['total_pv_kwh']:.1f} kWh")
                st.metric("EEG-Erlöse", f"{economics['total_eeg_revenue']:.2f} €")
            with col2:
                st.metric("Potentielle Markterlöse", f"{economics['total_market_revenue']:.2f} €")
                st.metric("Durchschnittspreis", f"{economics['prices'].mean():.2f} ct/kWh")
            with col3:
                st.metric("Opportunitätskosten", f"{economics['total_opportunity_cost']:.2f} €")
                if economics['total_opportunity_cost'] > 0:
                    st.warning("Markt > EEG: Verlust durch EEG-Vergütung")
                else:
                    st.success("EEG > Markt: Gewinn durch EEG-Vergütung")
            
            # Diagramme
            st.header("Visualisierungen")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # PV-Erzeugung über Zeit
            ax1.plot(analysis_pv.index, analysis_pv.values)
            ax1.set_title('PV-Erzeugung über Zeit')
            ax1.set_ylabel('kWh')
            ax1.tick_params(axis='x', rotation=45)
            
            # Preise über Zeit
            ax2.plot(economics['prices'].index, economics['prices'].values, color='orange')
            ax2.set_title('Day-Ahead Preise')
            ax2.set_ylabel('ct/kWh')
            ax2.tick_params(axis='x', rotation=45)
            
            # Erlösvergleich
            daily_eeg = economics['eeg_revenue'].resample('D').sum()
            daily_market = economics['market_revenue'].resample('D').sum()
            ax3.plot(daily_eeg.index, daily_eeg.values, label='EEG-Erlöse', color='green')
            ax3.plot(daily_market.index, daily_market.values, label='Markterlöse', color='red')
            ax3.set_title('Tägliche Erlöse im Vergleich')
            ax3.set_ylabel('€')
            ax3.legend()
            ax3.tick_params(axis='x', rotation=45)
            
            # Preishistogramm
            ax4.hist(economics['prices'].values, bins=50, alpha=0.7)
            ax4.axvline(eeg_ct_per_kwh, color='red', linestyle='--', label=f'EEG: {eeg_ct_per_kwh} ct/kWh')
            ax4.set_title('Preisverteilung')
            ax4.set_xlabel('ct/kWh')
            ax4.set_ylabel('Häufigkeit')
            ax4.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Zusammenfassung
            st.header("Zusammenfassung")
            if economics['total_opportunity_cost'] > 0:
                st.error(f"Durch die EEG-Vergütung entgehen Ihnen {economics['total_opportunity_cost']:.2f} € an Markterlösen.")
            else:
                st.success(f"Die EEG-Vergütung bringt Ihnen {abs(economics['total_opportunity_cost']):.2f} € mehr als der Markt.")
            
            if max_power_kw > 0 and clipping_loss.sum() > 0:
                st.warning(f"Clipping-Verluste: {clipping_loss.sum():.1f} kWh ({(clipping_loss.sum()/pv_data.sum()*100):.1f}%)")

else:
    st.info("Bitte laden Sie sowohl PV-Daten als auch Preisdaten hoch, um die Analyse zu starten.")
    
    # Beispielformat anzeigen
    st.header("Erwartetes Datenformat")
    st.subheader("PV-Daten:")
    st.text("""
Zeitstempel        | Einspeisung PV (kWh)
01.01. 00:00      | 0
01.01. 00:15      | 0
01.01. 00:30      | 0
...
    """)
    
    st.subheader("Preisdaten:")
    st.text("""
Zeitstempel        | Day Ahead-Preis (€/MWh)
01.01.2025 00:00  | 0.1
01.01.2025 00:15  | 0.1
01.01.2025 00:30  | 0.1
...
    """)
