"""
Plot of all values present in the CSV, averaged, with water percentage timestamps.
Uses the following files:
  - Oil Measurements.csv
  - Oil % Timestamps.csv
"""

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors
from datetime import datetime
import sqlite3
import numpy as np
import streamlit as st

st.set_page_config(page_title="OilSense Plotter", layout="wide")
st.title("OilSense Data Plotter")

# --- Sidebar Configuration Parameters ---
st.sidebar.header("Working Folder")

if "work_dir" not in st.session_state:
    st.session_state.work_dir = os.path.dirname(os.path.abspath(__file__))

if st.sidebar.button("Browse folder..."):
    import subprocess, sys, tempfile, json
    # Run tkinter in a separate process to avoid Tcl_AsyncDelete thread crashes
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_dir = _script_dir
    _script = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "import json, sys\n"
        "root = tk.Tk()\n"
        "root.withdraw()\n"
        "root.wm_attributes('-topmost', 1)\n"
        f"folder = filedialog.askdirectory(initialdir=r'{_default_dir}')\n"
        "root.destroy()\n"
        "print(json.dumps(folder))\n"
    )
    _result = subprocess.run(
        [sys.executable, "-c", _script],
        capture_output=True, text=True
    )
    try:
        _folder = json.loads(_result.stdout.strip())
    except Exception:
        _folder = ""
    if _folder:
        st.session_state.work_dir = _folder

WORK_DIR = st.sidebar.text_input("Selected path", value=st.session_state.work_dir)
st.session_state.work_dir = WORK_DIR

if not WORK_DIR:
    st.info("Waiting for data... Use the 'Browse folder...' button in the sidebar to select the folder containing the log files, or paste the path directly.")
    st.stop()

CSV_FILE = os.path.join(WORK_DIR, "Oil Measurements.csv")
TIMESTAMPS_FILE = os.path.join(WORK_DIR, "Oil % Timestamps.csv")
DB_FILE = os.path.join(WORK_DIR, "Oil Temp and Resistance.db")

missing_files = []
if not os.path.exists(CSV_FILE):
    missing_files.append("- **Oil Measurements.csv**: CSV file (separator ';') containing sensor measurements (timestamps, Amplitude, etc.).")
if not os.path.exists(TIMESTAMPS_FILE):
    missing_files.append("- **Oil % Timestamps.csv**: CSV with logs of actions and water percentage changes (e.g. engine on, engine off, % water in oil and their timestamps).")
if not os.path.exists(DB_FILE):
    missing_files.append("- **Oil Temp and Resistance.db**: SQLite database recording oil temperature and resistance over time.")

if missing_files:
    st.sidebar.error("Warning: incorrect folder, some files are missing.")
    st.error("### Missing files:\nSome required files were not found.\nMake sure the **Working Folder** selected on the left contains the following files:\n\n" + "\n".join(missing_files))
    st.info("Waiting for data... Select the correct folder to continue.")
    st.stop()

# ==============================================================
# CSV Load Function (defined early to compute timestamp default)
# ==============================================================
@st.cache_data
def load_csv_measurements(filepath):
    df = pd.read_csv(
        filepath,
        sep=";",
        decimal=",",
        encoding="utf-8",
        skip_blank_lines=True,
    )
    df.columns = df.columns.str.strip()
    time_col = df.columns[0]
    df["Timestamp"] = pd.to_datetime(df[time_col].str.replace(r'Tempo \(UTC \+01:00 yyyy-MM-dd HH:mm:ss\)', '', regex=True), format="%Y-%m-%d %H:%M:%S", exact=False)
    return df, time_col

# Peek at the CSV to get the max timestamp (cached, costs nothing on re-runs)
_df_peek, _ = load_csv_measurements(CSV_FILE)
_ts_max_default = (
    _df_peek["Timestamp"].dropna().max().strftime("%Y-%m-%d %H:%M:%S")
    if not _df_peek.empty else ""
)

st.sidebar.markdown("---")
st.sidebar.header("Configuration Parameters")

MOVING_AVERAGE_WINDOW = st.sidebar.number_input(
    "Moving Average Window", min_value=1, value=60,
    help="Number of consecutive samples used to compute the rolling mean. Higher values produce a smoother curve but lag more behind sudden changes. Applied to all signals (amplitude, temperature, pressure)."
)
DERIVATIVE_PERIODS = st.sidebar.number_input(
    "Derivative Periods", min_value=1, value=10,
    help="Number of samples over which the finite difference (rate of change) is calculated. A larger value makes the derivative less noisy but less responsive to fast transients."
)
SHOW_DERIVATIVE = st.sidebar.toggle(
    "Show Derivative Plots", value=False,
    help="Show a separate subplot below each signal chart for its derivative."
)

TIMESTAMP_MAX = st.sidebar.text_input(
    "Maximum timestamp to plot", value=_ts_max_default,
    help="Optional cut-off time. Data after this timestamp will be ignored. Format: YYYY-MM-DD HH:MM:SS. Defaults to the last available sample in the dataset."
)

MAX_TS = None
if TIMESTAMP_MAX:
    try:
        ts_str = str(TIMESTAMP_MAX)
        if len(ts_str) > 10:
            ts_str = ts_str[:10] + ts_str[10:].replace('.', ':')
        MAX_TS = pd.to_datetime(ts_str)
    except Exception as e:
        st.sidebar.warning(f"Warning: invalid TIMESTAMP_MAX format: {e}")

st.sidebar.markdown("---")
st.sidebar.header("Extra Sensors")
SHOW_TEMPERATURE = st.sidebar.toggle(
    "Show Temperature", value=False,
    help="Overlay the oil temperature (°C) on each signal chart as a red line on a secondary Y-axis. Data comes from the 'Oil Temp and Resistance.db' database (flowsense.temp_pt100_1)."
)
SHOW_RESISTANCE = st.sidebar.toggle(
    "Show Resistance", value=False,
    help="Overlay the oil resistance (Ω) on each signal chart as a cyan line on a secondary Y-axis. Data comes from the 'Oil Temp and Resistance.db' database (flowsense.oil_resistance)."
)
SHOW_PRESSURE = st.sidebar.toggle(
    "Show Pressure", value=False,
    help="Overlay the oil pressure (bar) on each signal chart as a purple line on a secondary Y-axis. Data comes from the 'Oil Temp and Resistance.db' database (analog_inputs.oil_pressure)."
)

# ==============================================================
# Data Loading Functions (cached for performance)
# ==============================================================
@st.cache_data
def load_db_temperature(db_path):
    try:
        if not os.path.exists(db_path):
            return pd.DataFrame()
        with sqlite3.connect(db_path) as conn:
            # Load temperature (temp_pt100_1) and oil_resistance from flowsense table
            try:
                df_temp = pd.read_sql_query("SELECT time, temp_pt100_1 AS oil_temp, oil_resistance FROM flowsense", conn)
            except Exception:
                try:
                    df_temp = pd.read_sql_query("SELECT time, temp_pt100_1 AS oil_temp FROM flowsense", conn)
                except Exception:
                    df_temp = pd.DataFrame()
            try:
                df_press = pd.read_sql_query("SELECT time, oil_pressure FROM analog_inputs", conn)
                if not df_temp.empty and not df_press.empty:
                    df_temp = pd.merge(df_temp, df_press, on="time", how="outer")
                elif df_temp.empty and not df_press.empty:
                    df_temp = df_press
            except Exception:
                pass

        if df_temp.empty:
            return pd.DataFrame()

        df_temp["Timestamp"] = pd.to_datetime(df_temp["time"])
        df_temp["Timestamp"] = df_temp["Timestamp"].dt.tz_convert('UTC').dt.tz_localize(None) + pd.Timedelta(hours=2)
        df_temp.dropna(subset=["Timestamp"], inplace=True)
        df_temp.sort_values("Timestamp", inplace=True)
        return df_temp
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def load_water_timestamps(ts_filepath):
    if not os.path.exists(ts_filepath):
        return pd.DataFrame()
    return pd.read_csv(ts_filepath)

# ==============================================================
# 1. Read measurements CSV
# ==============================================================
df_raw, time_col = load_csv_measurements(CSV_FILE)
df = df_raw.copy()

available_columns = [c for c in df.columns if c not in [time_col, "Timestamp"]]

# --- Column selector (populated with actual CSV column names) ---
st.sidebar.markdown("---")
_amplitude_cols = [c for c in available_columns if "amplitude" in c.lower() or "ampiezza" in c.lower()]
_default_cols = _amplitude_cols if _amplitude_cols else available_columns
columns_to_plot = st.sidebar.multiselect(
    "Columns to plot",
    options=available_columns,
    default=_default_cols,
    help="Pick the CSV columns to display. Each selected column gets its own signal chart (moving average) and a derivative chart below it."
)

if not columns_to_plot:
    st.warning("No column selected. Select at least one variable from the sidebar.")
    st.stop()

for col in columns_to_plot:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=["Timestamp"], inplace=True)
df.sort_values("Timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)

if MAX_TS is not None:
    df = df[df["Timestamp"] <= MAX_TS].reset_index(drop=True)

# ==============================================================
# 2. Moving average and derivative
# ==============================================================
for col in columns_to_plot:
    df[f"{col}_Avg"] = df[col].rolling(window=MOVING_AVERAGE_WINDOW, min_periods=1).mean()

    dt = df["Timestamp"].diff(periods=DERIVATIVE_PERIODS).dt.total_seconds()
    dt = dt.replace(0, np.nan)
    df[f"{col}_Deriv"] = df[f"{col}_Avg"].diff(periods=DERIVATIVE_PERIODS) / dt

# ==============================================================
# 2.5 Read temperature/pressure database and apply moving average
# ==============================================================
df_temp_raw = load_db_temperature(DB_FILE)
df_temp = df_temp_raw.copy()
if not df_temp.empty and MAX_TS is not None:
    df_temp = df_temp[df_temp["Timestamp"] <= MAX_TS].reset_index(drop=True)
if not df_temp.empty:
    if "oil_temp" in df_temp.columns:
        df_temp["oil_temp_avg"] = df_temp["oil_temp"].rolling(window=MOVING_AVERAGE_WINDOW, min_periods=1).mean()
    if "oil_resistance" in df_temp.columns:
        df_temp["oil_resistance_avg"] = df_temp["oil_resistance"].rolling(window=MOVING_AVERAGE_WINDOW, min_periods=1).mean()
    if "oil_pressure" in df_temp.columns:
        df_temp["oil_pressure_avg"] = df_temp["oil_pressure"].rolling(window=MOVING_AVERAGE_WINDOW, min_periods=1).mean()

# ==============================================================
# 3. Read water percentage timestamps
# ==============================================================
if not df.empty:
    ref_date = df["Timestamp"].iloc[0].date()
else:
    ref_date = datetime.now().date()

timestamps_df = load_water_timestamps(TIMESTAMPS_FILE)
timestamps_pct = []
timestamps_actions = []
target_actions = ["motore spento", "motore acceso", "intervento faggiolati", "reset faggiolati"]
seen_pcts = set()

if not timestamps_df.empty:
    for index, row in timestamps_df.iterrows():
        pct_str = str(row.get('% Acqua in Olio', '')).strip()
        time_str = str(row.get('Orario (HH:mm:ss)', '')).strip()
        action_str = str(row.get('Azione', '')).strip()

        try:
            time_obj = datetime.strptime(time_str, "%H.%M.%S").time()
        except ValueError:
            try:
                time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
            except ValueError:
                continue

        ts = datetime.combine(ref_date, time_obj)

        if MAX_TS is not None:
            if pd.Timestamp(ts) > MAX_TS:
                continue

        if any(a in action_str.lower() for a in target_actions):
            timestamps_actions.append((ts, action_str))

        try:
            pct_val = float(pct_str.replace('%', '').replace('"', '').replace(',', '.'))
            if pct_val not in seen_pcts:
                timestamps_pct.append((ts, pct_val, pct_str))
                seen_pcts.add(pct_val)
        except ValueError:
            continue

# ==============================================================
# 4. Plot
# ==============================================================
if df.empty:
    st.warning("No data to display for this time range.")
else:
    num_rows_per_col = 2 if SHOW_DERIVATIVE else 1
    total_rows = len(columns_to_plot) * num_rows_per_col

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        specs=[[{"secondary_y": True}]] * total_rows
    )

    colors_vlines = []
    if timestamps_pct:
        # Filter out entries with NaN percentage values to avoid plotly color conversion errors
        timestamps_pct = [(ts, v, lbl) for ts, v, lbl in timestamps_pct if not (isinstance(v, float) and np.isnan(v))]
        if timestamps_pct:
            colors_vlines = plotly.colors.sample_colorscale('Bluered', [min(v / 50.0, 1.0) for (_, v, _) in timestamps_pct])

    for idx, col in enumerate(columns_to_plot):
        if SHOW_DERIVATIVE:
            row_ax = idx * 2 + 1
            row_deriv = idx * 2 + 2
        else:
            row_ax = idx + 1
            row_deriv = None

        # Main signal trace (moving average)
        fig.add_trace(
            go.Scatter(
                x=df["Timestamp"],
                y=df[f"{col}_Avg"],
                mode='lines',
                line=dict(color="#1f77b4", width=1.5),
                name=f"{col} - Moving avg {MOVING_AVERAGE_WINDOW} pts",
                legendgroup=f"group{idx}",
            ),
            row=row_ax, col=1, secondary_y=False
        )

        # Derivative trace
        if SHOW_DERIVATIVE:
            fig.add_trace(
                go.Scatter(
                    x=df["Timestamp"],
                    y=df[f"{col}_Deriv"],
                    mode='lines',
                    line=dict(color="darkorange", width=1.5),
                    name=f"Derivative {col}",
                    legendgroup=f"group{idx}_deriv",
                ),
                row=row_deriv, col=1, secondary_y=False
            )

        # Engine on/off colored background regions
        engine_events = sorted(
            [(ts, az.lower()) for ts, az in timestamps_actions if "motore acceso" in az.lower() or "motore spento" in az.lower()],
            key=lambda x: x[0]
        )
        if engine_events:
            t_max_df = df["Timestamp"].max()
            if not df_temp.empty:
                t_max_df = max(t_max_df, df_temp["Timestamp"].max())

            for i in range(len(engine_events)):
                start_time, action = engine_events[i]
                if i < len(engine_events) - 1:
                    end_time = engine_events[i + 1][0]
                else:
                    end_time = t_max_df
                if MAX_TS is not None and pd.Timestamp(end_time) > MAX_TS:
                    end_time = MAX_TS

                # Green = engine on, grey = engine off
                fill_color = "lightgreen" if "acceso" in action else "lightgrey"

                for r in [row_ax]:
                    fig.add_vrect(
                        x0=start_time,
                        x1=end_time,
                        fillcolor=fill_color,
                        opacity=0.3,
                        layer="below",
                        line_width=0,
                        row=r, col=1
                    )

        # Water percentage vertical lines
        for i, (ts, val, label_str) in enumerate(timestamps_pct):
            for r in [row_ax]:
                fig.add_vline(
                    x=ts,
                    line_color=colors_vlines[i],
                    
                    line_width=1.2,
                    line_dash="dash",
                    opacity=0.85,
                    row=r, col=1
                )
                fig.add_annotation(
                    x=ts,
                    y=0.02,
                    yref="y domain",
                    text=f" {label_str}",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    textangle=-90,
                    font=dict(color=colors_vlines[i], size=10),
                    row=r, col=1
                )

        # Action event vertical lines (engine on/off, faggiolati)
        for ts, action in timestamps_actions:
            action_display = action
            if "motore acceso" in action.lower():
                action_display = "engine on"
            elif "motore spento" in action.lower():
                action_display = "engine off"

            for r in [row_ax]:
                fig.add_vline(
                    x=ts,
                    line_color="black",
                    line_width=1.5,
                    line_dash="dot",
                    opacity=0.85,
                    row=r, col=1
                )
                fig.add_annotation(
                    x=ts,
                    y=0.98,
                    yref="y domain",
                    text=f" {action_display}",
                    showarrow=False,
                    xanchor="left",
                    yanchor="top",
                    textangle=-90,
                    font=dict(color="black", size=10),
                    row=r, col=1
                )

        fig.update_yaxes(title_text=col, row=row_ax, col=1, secondary_y=False)
        if SHOW_DERIVATIVE:
            fig.update_yaxes(title_text=f"Derivative {col}", row=row_deriv, col=1, secondary_y=False)

        # Oil temperature overlay on secondary y-axis (moving-averaged)
        if not df_temp.empty and SHOW_TEMPERATURE and "oil_temp_avg" in df_temp.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_temp["Timestamp"],
                    y=df_temp["oil_temp_avg"],
                    mode='lines',
                    line=dict(color="red", width=1.2),
                    opacity=0.8,
                    name=f"Oil Temperature (°C) - avg {MOVING_AVERAGE_WINDOW} pts",
                    legendgroup="overlay_temp",
                    showlegend=(idx == 0),
                ),
                row=row_ax, col=1, secondary_y=True
            )
            fig.update_yaxes(title_text="Temperature (°C)", color="red", row=row_ax, col=1, secondary_y=True)

        # Oil resistance overlay on secondary y-axis (moving-averaged)
        if not df_temp.empty and SHOW_RESISTANCE and "oil_resistance_avg" in df_temp.columns:
            df_res = df_temp.dropna(subset=["oil_resistance_avg"])
            if not df_res.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_res["Timestamp"],
                        y=df_res["oil_resistance_avg"],
                        mode='lines',
                        line=dict(color="deepskyblue", width=1.2),
                        opacity=0.8,
                        name=f"Oil Resistance (Ω) - avg {MOVING_AVERAGE_WINDOW} pts",
                        legendgroup="overlay_res",
                        showlegend=(idx == 0),
                    ),
                    row=row_ax, col=1, secondary_y=True
                )
                fig.update_yaxes(title_text="Resistance (Ω)", color="deepskyblue", row=row_ax, col=1, secondary_y=True)

        # Oil pressure overlay on secondary y-axis (moving-averaged)
        if not df_temp.empty and SHOW_PRESSURE and "oil_pressure_avg" in df_temp.columns:
            df_pres = df_temp.dropna(subset=["oil_pressure_avg"])
            if not df_pres.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_pres["Timestamp"],
                        y=df_pres["oil_pressure_avg"],
                        mode='lines',
                        line=dict(color="purple", width=1.2),
                        opacity=0.8,
                        name=f"Oil Pressure (bar) - avg {MOVING_AVERAGE_WINDOW} pts",
                        legendgroup="overlay_pres",
                        showlegend=(idx == 0),
                    ),
                    row=row_ax, col=1, secondary_y=True
                )
                fig.update_yaxes(title_text="Pressure (bar)", color="purple", row=row_ax, col=1, secondary_y=True)

    fig.update_xaxes(
        title_text="Time",
        tickformat="%H:%M:%S",
        tickangle=60,
        row=total_rows, col=1
    )

    if MAX_TS is not None:
        fig.update_xaxes(range=[df["Timestamp"].min(), MAX_TS])

    fig.update_layout(
        height=max(600, total_rows * 300),
        title_text=f"Values averaged over {MOVING_AVERAGE_WINDOW} points vs Time",
        title_font=dict(size=16),
        hovermode="x unified",
        template="plotly_white",
        showlegend=False
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', griddash='dot', dtick=60000, hoverformat='%H:%M:%S')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', griddash='dot')

    st.plotly_chart(fig, width='stretch')