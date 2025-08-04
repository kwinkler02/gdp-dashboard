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
        
        # Konvertiere Datumsstrings - VEREINHEITLICHE AUF 2025
        target_year = 2025  # Festes Jahr für Konsistenz
        parsed_timestamps = []
        
        for i, s in enumerate(ts):
            try:
                # Skip NaN values
                if pd.isna(s) or 'NaN' in str(s) or 'nan' in str(s):
                    continue
                
                s = str(s).strip()
                
                # Verschiedene Parsing-Strategien
                parsed_dt = None
                
                # Strategie 1: Pandas auto-parsing
                try:
                    parsed_dt = pd.to_datetime(s, dayfirst=True)
                    # Falls Jahr nicht 2025, ersetze es
                    if parsed_dt.year != target_year:
                        parsed_dt = parsed_dt.replace(year=target_year)
                except:
                    pass
                
                # Strategie 2: Manuelles Parsing für "01.01. 00:00" Format
                if parsed_dt is None and ' ' in s:
                    try:
                        date_part, time_part = s.split(' ', 1)
                        if date_part.endswith('.'):
                            # "01.01." -> "01.01.2025"
                            day, month = date_part[:-1].split('.')
                            full_date_str = f"{day}.{month}.{target_year} {time_part}"
                            parsed_dt = pd.to_datetime(full_date_str, format='%d.%m.%Y %H:%M')
                    except:
                        pass
                
                # Strategie 3: Nur Datum ohne Zeit
                if parsed_dt is None:
                    try:
                        if '.' in s and ':' not in s:
                            parts = s.split('.')
                            if len(parts) >= 2:
                                day, month = parts[0], parts[1]
                                full_date_str = f"{day}.{month}.{target_year} 00:00"
                                parsed_dt = pd.to_datetime(full_date_str, format='%d.%m.%Y %H:%M')
                    except:
                        pass
                
                if parsed_dt is not None:
                    parsed_timestamps.append(parsed_dt)
                else:
                    st.warning(f"Konnte Zeitstempel nicht parsen: '{s}' (Zeile {i+2})")
                    
            except Exception as e:
                st.warning(f"Fehler beim Parsen von Zeitstempel '{s}' (Zeile {i+2}): {e}")
                continue
        
        if len(parsed_timestamps) == 0:
            st.error("Keine gültigen Zeitstempel gefunden!")
            return None
        
        # Series erstellen mit gültigen Zeitstempeln
        valid_vals = vals.iloc[:len(parsed_timestamps)]
        series = pd.Series(valid_vals.values, index=parsed_timestamps)
        series.name = vals.name or 'value'
        
        # Sortiere nach Zeitstempel
        series = series.sort_index()
        
        # Debug-Info
        with st.expander(f"Debug: {file.name}", expanded=False):
            st.write(f"Zeitspanne: {series.index.min()} bis {series.index.max()}")
            st.write(f"Erste 3 Zeitstempel: {series.index[:3].tolist()}")
            st.write(f"Letzte 3 Zeitstempel: {series.index[-3:].tolist()}")
            st.write(f"Anzahl Datenpunkte: {len(series)}")
        
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
    
    with st.expander("Debug: Zeitreihen-Synchronisation", expanded=False):
        st.write("PV Zeitspanne:", pv_series.index.min(), "bis", pv_series.index.max())
        st.write("Preis Zeitspanne:", price_series.index.min(), "bis", price_series.index.max())
        st.write("PV Datenpunkte:", len(pv_series))
        st.write("Preis Datenpunkte:", len(price_series))
    
    # Erweiterte Zeitreihen-Synchronisation mit Toleranz
    common_index = pv_series.index.intersection(price_series.index)
    
    st.write(f"✅ Direkte Zeitstempel-Übereinstimmungen: {len(common_index)}")
    
    if len(common_index) == 0:
        # Versuche Nearest-Neighbor Matching mit 15min Toleranz
        st.info("🔄 Keine exakten Übereinstimmungen - versuche intelligentes Matching...")
        
        # Reindex mit nearest method und 15min Toleranz
        try:
            prices_reindexed = price_series.reindex(pv_series.index, method='nearest', tolerance=pd.Timedelta('15 minutes'))
            
            # Entferne NaN Werte
            valid_mask = ~prices_reindexed.isna() & ~pv_series.isna()
            
            if valid_mask.sum() == 0:
                st.error("❌ Auch mit 15-Minuten-Toleranz keine passenden Zeitstempel gefunden!")
                with st.expander("Zeitstempel-Vergleich", expanded=True):
                    st.write("**PV erste 10 Zeitstempel:**")
                    st.write(pv_series.index[:10].tolist())
                    st.write("**Preis erste 10 Zeitstempel:**")
                    st.write(price_series.index[:10].tolist())
                return None
            
            pv_aligned = pv_series[valid_mask]
            prices_aligned = prices_reindexed[valid_mask]
            
            st.success(f"✅ Matching erfolgreich: {len(pv_aligned)} Datenpunkte synchronisiert (von {len(pv_series)} PV-Punkten)")
            
        except Exception as e:
            st.error(f"❌ Fehler beim Nearest-Neighbor Matching: {e}")
            return None
    else:
        pv_aligned = pv_series.reindex(common_index, fill_value=0)
        prices_aligned = price_series.reindex(common_index, fill_value=0)
        st.success(f"✅ Perfekte Zeitstempel-Übereinstimmung: {len(common_index)} Datenpunkte")
    
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
