"""
Plot di tutti i valori presenti nel CSV, mediati, con timestamps percentuale acqua.
Usa i file:
  - Oil Mesaurements.csv
  - Oil % Timestamps.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sqlite3

# --- Parametri di Configurazione ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "Oil Measurements.csv")
TIMESTAMPS_FILE = os.path.join(SCRIPT_DIR, "Oil % Timestamps.csv")
DB_FILE = os.path.join(SCRIPT_DIR, "Oil Temp and Pressure.db")

FINESTRA_MEDIA_MOBILE = 5

# Se la lista è vuota, verranno stampate tutte le colonne.
# Altrimenti verranno stampate solo quelle che contengono queste stringhe (case-insensitive).
COLONNE_DA_PLOTTARE = ["Ampiezza", "Tempo di salita"]

# Timestamp massimo da plottare (es: "2026-02-24 10:30:00"). Lasciare vuoto o None per non applicare alcun limite.
TIMESTAMP_MAX = None

# ==============================================================
# 1. Lettura CSV misure
# ==============================================================
df = pd.read_csv(
    CSV_FILE,
    sep=";",
    decimal=",",
    encoding="utf-8",
    skip_blank_lines=True,
)

# Pulizia nomi colonne
df.columns = df.columns.str.strip()

# Parsing timestamp
time_col = df.columns[0]  # "Tempo (UTC +01:00 yyyy-MM-dd HH:mm:ss)"
df["Timestamp"] = pd.to_datetime(df[time_col].str.replace(r'Tempo \(UTC \+01:00 yyyy-MM-dd HH:mm:ss\)', '', regex=True), format="%Y-%m-%d %H:%M:%S", exact=False) 
# Trova tutte le colonne da plottare (escludendo Timestamp)
if COLONNE_DA_PLOTTARE:
    colonne_da_plottare = [
        c for c in df.columns 
        if c not in [time_col, "Timestamp"] 
        and any(filtro.lower() in c.lower() for filtro in COLONNE_DA_PLOTTARE)
    ]
else:
    colonne_da_plottare = [c for c in df.columns if c not in [time_col, "Timestamp"]]

if not colonne_da_plottare:
    print(f"Attenzione: nessuna colonna corrisponde ai filtri {COLONNE_DA_PLOTTARE}. Plotto tutto.")
    colonne_da_plottare = [c for c in df.columns if c not in [time_col, "Timestamp"]]

# Forza numerico su tutte le colonne e rimuovi eventuale NaN su Timestamp
for col in colonne_da_plottare:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=["Timestamp"], inplace=True)
df.sort_values("Timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)

if TIMESTAMP_MAX:
    try:
        max_ts = pd.to_datetime(TIMESTAMP_MAX)
        df = df[df["Timestamp"] <= max_ts].reset_index(drop=True)
    except Exception as e:
        print(f"Attenzione, formato TIMESTAMP_MAX non valido: {e}")

# ==============================================================
# 2. Media mobile
# ==============================================================
for col in colonne_da_plottare:
    df[f"{col}_Avg"] = df[col].rolling(window=FINESTRA_MEDIA_MOBILE, min_periods=1).mean()

# ==============================================================
# 2.5 Lettura DB Temperatura
# ==============================================================
try:
    with sqlite3.connect(DB_FILE) as conn:
        df_temp = pd.read_sql_query("SELECT time, oil_temp FROM analog_inputs", conn)
    # Parsing timestamp formato "2026-02-24T08:40:48+00:00"
    df_temp["Timestamp"] = pd.to_datetime(df_temp["time"])
    # Converto in UTC+1 (rimuovendo il timezone per allinearlo con l'altro DataFrame)
    df_temp["Timestamp"] = df_temp["Timestamp"].dt.tz_convert('UTC').dt.tz_localize(None) + pd.Timedelta(hours=1)
    df_temp.dropna(subset=["Timestamp", "oil_temp"], inplace=True)
    df_temp.sort_values("Timestamp", inplace=True)
    
    if TIMESTAMP_MAX:
        try:
            max_ts = pd.to_datetime(TIMESTAMP_MAX)
            df_temp = df_temp[df_temp["Timestamp"] <= max_ts].reset_index(drop=True)
        except Exception as e:
            pass # L'errore verrebbe già stampato per il df principale

except Exception as e:
    print(f"Attenzione, errore lettura DB: {e}")
    df_temp = pd.DataFrame()

# ==============================================================
# 3. Lettura timestamps percentuali acqua
# ==============================================================
# La data di riferimento è la data della prima riga nel CSV misure
ref_date = df["Timestamp"].iloc[0].date()

timestamps_df = pd.read_csv(TIMESTAMPS_FILE)
timestamps_pct = []
timestamps_azioni = []
azioni_target = ["motore spento", "motore acceso", "intervento faggiolati"]

for index, row in timestamps_df.iterrows():
    pct_str = str(row.get('% Acqua in Olio', '')).strip()
    time_str = str(row.get('Orario (HH:mm:ss)', '')).strip()
    azione_str = str(row.get('Azione', '')).strip()
    
    try:
        # Check standard format HH.MM.SS
        time_obj = datetime.strptime(time_str, "%H.%M.%S").time()
    except ValueError:
        try:
            # Check HH:MM:SS format fallback
            time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
        except ValueError:
            continue

    ts = datetime.combine(ref_date, time_obj)
    
    if TIMESTAMP_MAX:
        try:
            max_ts = pd.to_datetime(TIMESTAMP_MAX)
            if pd.Timestamp(ts) > max_ts:
                continue
        except Exception:
            pass

    if any(a in azione_str.lower() for a in azioni_target):
        timestamps_azioni.append((ts, azione_str))

    try:
        pct_val = float(pct_str.replace('%', '').replace('"', '').replace(',', '.'))
        timestamps_pct.append((ts, pct_val, pct_str))
    except ValueError:
        continue

# ==============================================================
# 4. Plot
# ==============================================================
fig, axes = plt.subplots(nrows=len(colonne_da_plottare), ncols=1, figsize=(16, 4 * len(colonne_da_plottare)), sharex=True)

if len(colonne_da_plottare) == 1:
    axes = [axes]

# Linee verticali per ogni timestamp di percentuale acqua
colors_vlines = plt.cm.viridis_r(
    [v / 50.0 for (_, v, _) in timestamps_pct]
)  # gradiente colore

for idx, ax in enumerate(axes):
    col = colonne_da_plottare[idx]
    
    # Curva principale
    ax.plot(
        df["Timestamp"],
        df[f"{col}_Avg"],
        color="#1f77b4",
        linewidth=1.5,
        label=f"Media {FINESTRA_MEDIA_MOBILE} punti",
    )
    
    for i, (ts, val, label_str) in enumerate(timestamps_pct):
        ax.axvline(
            x=ts,
            color=colors_vlines[i],
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
        )
        # Etichetta percentuale — posizionata in basso, con rotazione
        y_top = ax.get_ylim()[1]
        y_bottom = ax.get_ylim()[0]
        ax.text(
            ts,
            y_bottom + (y_top - y_bottom) * 0.03,
            f" {label_str}",
            rotation=90,
            verticalalignment="bottom",
            fontsize=9,
            fontweight="bold",
            color=colors_vlines[i],
        )

    for ts, azione in timestamps_azioni:
        ax.axvline(
            x=ts,
            color="black",
            linestyle=":",
            linewidth=1.5,
            alpha=0.85,
        )
        y_top = ax.get_ylim()[1]
        y_bottom = ax.get_ylim()[0]
        ax.text(
            ts,
            y_top - (y_top - y_bottom) * 0.03,
            f"{azione} ",
            rotation=90,
            verticalalignment="top",
            fontsize=9,
            fontweight="bold",
            color="black",
        )

    ax.set_ylabel(col, fontsize=10)
    
    # Sovrapposizione Temperatura
    if not df_temp.empty:
        ax2 = ax.twinx()
        ax2.plot(
            df_temp["Timestamp"],
            df_temp["oil_temp"],
            color="red",
            linewidth=1.2,
            linestyle="-",
            alpha=0.8,
            label="Temperatura Olio (°C)",
        )
        ax2.set_ylabel("Temperatura (°C)", color="red", fontsize=10)
        ax2.tick_params(axis='y', labelcolor="red")
        
        # Unisco le legende
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=9)
    else:
        ax.legend(loc="upper right", fontsize=9)

    ax.grid(True, alpha=0.3, linestyle="-")

# Formattazione assi inferiori
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=60, ha="right", fontsize=8)

axes[-1].set_xlabel("Tempo", fontsize=12)
fig.suptitle(
    f"Valori mediati su {FINESTRA_MEDIA_MOBILE} punti vs Tempo\n"
    "con percentuali di acqua nell'olio",
    fontsize=16,
    fontweight="bold",
)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()
