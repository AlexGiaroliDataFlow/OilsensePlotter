"""
Flowsense V2 plotter — dati 15/07/2026.
Campione non omogeneo: tracce a gradino (hold del valore precedente fino al prossimo aggiornamento).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="OilSense Plotter — Flowsense V2 (15/07/2026)", layout="wide")
st.title("OilSense Data Plotter — Flowsense V2 — 15/07/2026")

WORK_DIR = Path(__file__).resolve().parent

COLUMN_LABELS = {
    "pt100_1_c": "Temperatura PT100 (°C)",
    "olio_resistenza_kohm": "Resistenza olio (kΩ)",
    "olio_capacita_nf": "Capacità olio (nF)",
}

COLUMN_RENAME = {
    "oil_resistance_kohm": "olio_resistenza_kohm",
    "oil_capacitance_nf": "olio_capacita_nf",
    "temperature_c": "pt100_1_c",
}

TARGET_ACTIONS = ("motore spento", "motore acceso", "stato iniziale", "sostituzione")


def discover_tests(base_dir: Path) -> list[int]:
    tests: set[int] = set()
    for path in base_dir.iterdir():
        m = re.match(r"(?i)test\s*(\d+)\s+dati\.csv$", path.name)
        if m:
            tests.add(int(m.group(1)))
    return sorted(tests, reverse=True)


def test_paths(base_dir: Path, test_num: int) -> dict[str, Path | None]:
    description = base_dir / f"test {test_num} descrizione.txt"
    if not description.exists():
        alt = base_dir / f"test {test_num}.txt"
        description = alt if alt.exists() else None

    return {
        "dati": base_dir / f"Test {test_num} dati.csv",
        "eventi": base_dir / f"Test {test_num} orari e percentuali.csv",
        "descrizione": description,
    }


def parse_pct(raw) -> float:
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip().replace("%", "").replace(",", ".")
    if not s or s.lower() == "nan":
        return np.nan
    val = float(s)
    if val > 1:
        val /= 100.0
    return val


@st.cache_data
def load_description(filepath: str) -> str:
    path = Path(filepath)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


@st.cache_data
def load_measurements(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding="utf-8")
    df.columns = df.columns.str.strip()
    df = df.rename(columns=COLUMN_RENAME)
    df["Timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    value_cols = [c for c in df.columns if c not in ("timestamp", "Timestamp")]
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Timestamp"], inplace=True)
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    for col in value_cols:
        df[col] = df[col].ffill()
    return df


@st.cache_data
def load_events(filepath: str, ref_date_str: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8")
    df.columns = df.columns.str.strip()
    ref_date = pd.Timestamp(ref_date_str).date()
    rows = []
    for _, row in df.iterrows():
        time_str = str(row.get("Orario (HH:mm:ss)", "")).strip()
        if not time_str or time_str.lower() == "nan":
            continue
        for fmt in ("%H:%M:%S", "%H.%M.%S"):
            try:
                time_obj = datetime.strptime(time_str, fmt).time()
                break
            except ValueError:
                time_obj = None
        if time_obj is None:
            continue
        ts = datetime.combine(ref_date, time_obj)
        pct_val = parse_pct(row.get("% Acqua in Olio"))
        action = str(row.get("Azione", "")).strip()
        rows.append({"Timestamp": ts, "pct": pct_val, "action": action})
    return pd.DataFrame(rows)


def parse_events(events_df: pd.DataFrame, max_ts: pd.Timestamp | None):
    timestamps_pct: list[tuple[datetime, float, str]] = []
    timestamps_actions: list[tuple[datetime, str]] = []
    seen_pcts: set[float] = set()

    if events_df.empty:
        return timestamps_pct, timestamps_actions

    for _, row in events_df.iterrows():
        ts = row["Timestamp"]
        if max_ts is not None and pd.Timestamp(ts) > max_ts:
            continue

        action = str(row.get("action", "")).strip()
        if action and any(a in action.lower() for a in TARGET_ACTIONS):
            timestamps_actions.append((ts, action))

        pct_val = row.get("pct")
        if pd.notna(pct_val):
            pct_float = float(pct_val)
            if pct_float not in seen_pcts:
                label = f"{pct_float * 100:.1f}%"
                timestamps_pct.append((ts, pct_float * 100.0, label))
                seen_pcts.add(pct_float)

    return timestamps_pct, timestamps_actions


def add_chart_overlays(fig, row_ax, df, timestamps_pct, timestamps_actions, colors_vlines, max_ts):
    engine_events = sorted(
        [
            (ts, az.lower())
            for ts, az in timestamps_actions
            if "motore acceso" in az.lower() or "motore spento" in az.lower()
        ],
        key=lambda x: x[0],
    )
    if engine_events:
        t_max_df = df["Timestamp"].max()
        for i, (start_time, action) in enumerate(engine_events):
            end_time = engine_events[i + 1][0] if i < len(engine_events) - 1 else t_max_df
            if max_ts is not None and pd.Timestamp(end_time) > max_ts:
                end_time = max_ts
            fill_color = "lightgreen" if "acceso" in action else "lightgrey"
            fig.add_vrect(
                x0=start_time,
                x1=end_time,
                fillcolor=fill_color,
                opacity=0.3,
                layer="below",
                line_width=0,
                row=row_ax,
                col=1,
            )

    for i, (ts, _val, label_str) in enumerate(timestamps_pct):
        fig.add_vline(
            x=ts,
            line_color=colors_vlines[i],
            line_width=1.2,
            line_dash="dash",
            opacity=0.85,
            row=row_ax,
            col=1,
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
            row=row_ax,
            col=1,
        )

    for ts, action in timestamps_actions:
        action_display = action
        if "motore acceso" in action.lower():
            action_display = "engine on"
        elif "motore spento" in action.lower():
            action_display = "engine off"
        elif "sostituzione" in action.lower():
            action_display = "replacement"
        elif "stato iniziale" in action.lower():
            action_display = "initial state"

        fig.add_vline(
            x=ts,
            line_color="black",
            line_width=1.5,
            line_dash="dot",
            opacity=0.85,
            row=row_ax,
            col=1,
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
            row=row_ax,
            col=1,
        )


def build_figure(
    df: pd.DataFrame,
    columns_to_plot: list[str],
    timestamps_pct,
    timestamps_actions,
    moving_average_window: int,
    show_derivative: bool,
    max_ts: pd.Timestamp | None,
    title: str,
):
    num_rows_per_col = 2 if show_derivative else 1
    total_rows = len(columns_to_plot) * num_rows_per_col

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    colors_vlines = []
    if timestamps_pct:
        colors_vlines = plotly.colors.sample_colorscale(
            "Bluered",
            [min(v / 50.0, 1.0) for (_, v, _) in timestamps_pct],
        )

    for idx, col in enumerate(columns_to_plot):
        row_ax = idx * 2 + 1 if show_derivative else idx + 1
        row_deriv = idx * 2 + 2 if show_derivative else None
        label = COLUMN_LABELS.get(col, col)

        fig.add_trace(
            go.Scatter(
                x=df["Timestamp"],
                y=df[f"{col}_Avg"],
                mode="lines",
                line=dict(color="#1f77b4", width=1.5, shape="hv"),
                name=f"{label} — media {moving_average_window} pt",
                legendgroup=f"group{idx}",
            ),
            row=row_ax,
            col=1,
        )

        if show_derivative and row_deriv is not None:
            fig.add_trace(
                go.Scatter(
                    x=df["Timestamp"],
                    y=df[f"{col}_Deriv"],
                    mode="lines",
                    line=dict(color="darkorange", width=1.5, shape="hv"),
                    name=f"Derivata {label}",
                    legendgroup=f"group{idx}_deriv",
                ),
                row=row_deriv,
                col=1,
            )
            fig.update_yaxes(title_text=f"Derivata {label}", row=row_deriv, col=1)

        add_chart_overlays(fig, row_ax, df, timestamps_pct, timestamps_actions, colors_vlines, max_ts)
        fig.update_yaxes(title_text=label, row=row_ax, col=1)

    fig.update_xaxes(
        title_text="Tempo",
        tickformat="%H:%M:%S",
        tickangle=60,
        row=total_rows,
        col=1,
    )
    if max_ts is not None:
        fig.update_xaxes(range=[df["Timestamp"].min(), max_ts])

    fig.update_layout(
        height=max(600, total_rows * 300),
        title_text=title,
        title_font=dict(size=16),
        hovermode="x unified",
        template="plotly_white",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray", griddash="dot", dtick=60000, hoverformat="%H:%M:%S")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray", griddash="dot")
    return fig


# --- Sidebar ---
st.sidebar.header("Test")
available_tests = discover_tests(WORK_DIR)
if not available_tests:
    st.error("Nessun test trovato (attesi file 'Test N dati.csv').")
    st.stop()

selected_tests = st.sidebar.multiselect(
    "Seleziona test",
    options=available_tests,
    default=[available_tests[0]],
    format_func=lambda n: f"Test {n}",
    help="I test sono elencati dal numero più alto. Ogni test usa i file con lo stesso numero.",
)

if not selected_tests:
    st.warning("Seleziona almeno un test dalla sidebar.")
    st.stop()

for test_num in selected_tests:
    paths = test_paths(WORK_DIR, test_num)
    missing = [
        name
        for name, path in paths.items()
        if name != "descrizione" and (path is None or not path.exists())
    ]
    if missing:
        st.error(f"File mancanti per Test {test_num}: {', '.join(missing)}")
        st.stop()

if st.sidebar.button("Ricarica dati"):
    st.cache_data.clear()
    st.rerun()

for test_num in selected_tests:
    paths = test_paths(WORK_DIR, test_num)
    st.sidebar.caption(f"Test {test_num}: `{paths['dati'].name}`, `{paths['eventi'].name}`")

st.sidebar.markdown("---")
st.sidebar.header("Parametri")

MOVING_AVERAGE_WINDOW = st.sidebar.number_input(
    "Finestra media mobile",
    min_value=1,
    value=10,
    help="Numero di campioni consecutivi per la media mobile.",
)
DERIVATIVE_PERIODS = st.sidebar.number_input(
    "Periodi derivata",
    min_value=1,
    value=10,
    help="Campioni usati per calcolare la derivata (differenza finita).",
)
SHOW_DERIVATIVE = st.sidebar.toggle(
    "Mostra grafici derivata",
    value=False,
)

TIMESTAMP_MAX = st.sidebar.text_input(
    "Timestamp massimo da plottare",
    value="",
    help="Formato: YYYY-MM-DD HH:MM:SS. Lascia vuoto per usare tutti i dati.",
)

MAX_TS = None
if TIMESTAMP_MAX.strip():
    try:
        ts_str = str(TIMESTAMP_MAX)
        if len(ts_str) > 10:
            ts_str = ts_str[:10] + ts_str[10:].replace(".", ":")
        MAX_TS = pd.to_datetime(ts_str)
    except Exception as exc:
        st.sidebar.warning(f"Formato TIMESTAMP_MAX non valido: {exc}")

test_datasets: list[dict] = []
all_value_columns: list[str] = []

for test_num in selected_tests:
    paths = test_paths(WORK_DIR, test_num)
    df_raw = load_measurements(str(paths["dati"]))
    if df_raw.empty:
        st.warning(f"Nessun dato nel file CSV di Test {test_num}.")
        continue

    ref_date_str = df_raw["Timestamp"].iloc[0].strftime("%Y-%m-%d")
    events_df = load_events(str(paths["eventi"]), ref_date_str)
    value_columns = [c for c in df_raw.columns if c not in ("timestamp", "Timestamp")]
    for col in value_columns:
        if col not in all_value_columns:
            all_value_columns.append(col)

    test_datasets.append(
        {
            "test_num": test_num,
            "paths": paths,
            "df_raw": df_raw,
            "events_df": events_df,
            "value_columns": value_columns,
        }
    )

if not test_datasets:
    st.stop()

st.sidebar.markdown("---")
default_columns = [c for c in all_value_columns if c != "pt100_1_c"]
columns_to_plot = st.sidebar.multiselect(
    "Colonne da plottare",
    options=all_value_columns,
    default=default_columns or all_value_columns,
    format_func=lambda c: COLUMN_LABELS.get(c, c),
)

if not columns_to_plot:
    st.warning("Seleziona almeno una colonna dalla sidebar.")
    st.stop()

for dataset in test_datasets:
    test_num = dataset["test_num"]
    paths = dataset["paths"]
    df_raw = dataset["df_raw"]
    events_df = dataset["events_df"]
    test_columns = [c for c in columns_to_plot if c in dataset["value_columns"]]

    if not test_columns:
        st.warning(f"Test {test_num}: nessuna delle colonne selezionate è disponibile.")
        continue

    st.markdown(f"### Test {test_num}")

    if paths["descrizione"] is not None:
        description = load_description(str(paths["descrizione"]))
        if description:
            st.info(description)

    df = df_raw.copy()
    if MAX_TS is not None:
        df = df[df["Timestamp"] <= MAX_TS].reset_index(drop=True)

    for col in test_columns:
        df[f"{col}_Avg"] = df[col].rolling(window=MOVING_AVERAGE_WINDOW, min_periods=1).mean()
        dt = df["Timestamp"].diff(periods=DERIVATIVE_PERIODS).dt.total_seconds()
        dt = dt.replace(0, np.nan)
        df[f"{col}_Deriv"] = df[f"{col}_Avg"].diff(periods=DERIVATIVE_PERIODS) / dt

    timestamps_pct, timestamps_actions = parse_events(events_df, MAX_TS)

    if df.empty:
        st.warning(f"Test {test_num}: nessun dato da visualizzare per questo intervallo temporale.")
        continue

    _ts_min = df["Timestamp"].min().strftime("%Y-%m-%d %H:%M:%S")
    _ts_max = df["Timestamp"].max().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Intervallo: {_ts_min} → {_ts_max} ({len(df)} punti)")

    fig = build_figure(
        df=df,
        columns_to_plot=test_columns,
        timestamps_pct=timestamps_pct,
        timestamps_actions=timestamps_actions,
        moving_average_window=MOVING_AVERAGE_WINDOW,
        show_derivative=SHOW_DERIVATIVE,
        max_ts=MAX_TS,
        title=(
            f"Test {test_num} — valori mediati su {MOVING_AVERAGE_WINDOW} punti "
            "(interpolazione a gradino)"
        ),
    )
    st.plotly_chart(fig, width="stretch")

    if test_num != selected_tests[-1]:
        st.markdown("---")
