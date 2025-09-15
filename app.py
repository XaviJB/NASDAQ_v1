# Screener NASDAQ Composite + S&P 500 — Streamlit
# Autor: ChatGPT (para Xavi)
# Requisitos: pip install streamlit yfinance pandas numpy requests scikit-learn
# Uso: streamlit run app.py

import os
import io
import time
import json
import math
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Screener NASDAQ + S&P500", layout="wide")

# -----------------------------
# Sidebar — Parámetros
# -----------------------------
st.sidebar.header("Parámetros del filtro")
FINNHUB_API_KEY = st.sidebar.text_input("Finnhub API Key (requerida)", type="password")

min_mcap_input = st.sidebar.number_input(
    "Capitalización mínima (USD)", value=1_500_000_000, min_value=0, step=50_000_000,
    help="Por defecto 1,5B. Si querías 1,5M, pon 1500000."
)

min_employees = st.sidebar.number_input("Empleados mínimos", value=150, min_value=0, step=50)

min_avg_dollar_vol = st.sidebar.number_input(
    "Volumen medio 3 meses (USD por día)", value=800_000, min_value=0, step=50_000,
    help="Media de (Close*Volume) últimos ~63 días hábiles."
)

vol_window = st.sidebar.slider("Ventana volatilidad (días hábiles)", 40, 126, 63)
vol_day_move = st.sidebar.number_input("Umbral movimiento diario (%)", value=2.5, min_value=0.0, step=0.1)
vol_day_ratio = st.sidebar.slider("% de días que deben superar el umbral", 10, 100, 40)

trend_mode = st.sidebar.selectbox(
    "Filtro de tendencia",
    [
        "Cualquiera",
        "Lateral-alcista (suave)",
        "Alcista tras caída brusca (>=10%)"
    ],
    help=(
        "Lateral-alcista: pendiente positiva suave en 90d y 20DMA cerca de 50DMA (±5%).\n"
        "Post-caída: drawdown >=10% en 60d y recuperación > +5% desde el mínimo."
    )
)

universe_opt = st.sidebar.multiselect(
    "Universo de índices",
    ["S&P 500 (^GSPC)", "NASDAQ Composite (^IXIC)"],
    default=["S&P 500 (^GSPC)", "NASDAQ Composite (^IXIC)"]
)

limit_tickers = st.sidebar.number_input(
    "Límite máximo de símbolos a procesar (para pruebas)", value=0, min_value=0,
    help="0 = sin límite. El NASDAQ Composite tiene miles de valores; limita si tu equipo es modesto."
)

run_btn = st.sidebar.button("▶ Ejecutar screener")

# -----------------------------
# Utilidades
# -----------------------------
FINNHUB_BASE = "https://finnhub.io/api/v1"
headers = None
if FINNHUB_API_KEY:
    headers = {"X-Finnhub-Token": FINNHUB_API_KEY}

@st.cache_data(show_spinner=False)
def get_index_constituents(symbol: str) -> list:
    """Obtiene los componentes de un índice desde Finnhub (p.ej. ^GSPC, ^IXIC)."""
    url = f"{FINNHUB_BASE}/index/constituents"
    r = requests.get(url, params={"symbol": symbol}, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    syms = data.get("constituents") or []
    # Finnhub suele devolver tickers US sin sufijos. Filtramos entradas raras.
    return [s for s in syms if isinstance(s, str) and len(s) > 0]

@st.cache_data(show_spinner=False)
def get_profiles_bulk(symbols: list) -> pd.DataFrame:
    """Descarga perfil básico (market cap, empleados) con Finnhub."""
    rows = []
    for s in symbols:
        try:
            url = f"{FINNHUB_BASE}/stock/profile2"
            r = requests.get(url, params={"symbol": s}, headers=headers, timeout=15)
            if r.status_code != 200:
                continue
            j = r.json() or {}
            rows.append({
                "symbol": s,
                "name": j.get("name"),
                "market_cap": j.get("marketCapitalization"),
                "employees": j.get("employeeTotal"),
                "exchange": j.get("exchange"),
                "ticker": j.get("ticker"),
                "ipo": j.get("ipo"),
                "country": j.get("country")
            })
        except Exception:
            continue
        time.sleep(0.05)  # ser amables con la API
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def get_prices(symbols: list, period="1y") -> pd.DataFrame:
    """Descarga OHLCV diario de yfinance (multi-index columns)."""
    data = yf.download(symbols, period=period, interval="1d", auto_adjust=False, progress=False, group_by="ticker")
    return data

def compute_dollar_vol(df_close: pd.Series, df_vol: pd.Series, window: int = 63) -> float:
    x = (df_close * df_vol).dropna()
    if len(x) == 0:
        return np.nan
    return float(x.tail(window).mean())

def ratio_large_moves(df_close: pd.Series, window: int, pct: float) -> float:
    ret = df_close.pct_change().abs().dropna()
    if len(ret) == 0:
        return 0.0
    w = ret.tail(window)
    return 100.0 * (w >= (pct/100.0)).mean()

def lr_slope(series: pd.Series) -> float:
    y = series.dropna().values.reshape(-1, 1)
    if len(y) < 10:
        return np.nan
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    return float(model.coef_[0])

def lateral_up_filter(close: pd.Series) -> bool:
    c = close.dropna()
    if len(c) < 100:
        return False
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    cond_band = (abs(sma20.iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1]) <= 0.05  # ±5%
    slope90 = lr_slope(c.tail(90))
    return bool(cond_band and slope90 > 0)

def post_drop_filter(close: pd.Series) -> bool:
    c = close.dropna()
    if len(c) < 70:
        return False
    # drawdown >=10% en ventana de 60d (mínimo frente a máximo previo cercano)
    last60 = c.tail(60)
    peak = last60.expanding().max()
    dd = (last60/peak - 1.0)
    if (dd.min() <= -0.10):
        trough_idx = dd.idxmin()
        trough_price = last60.loc[trough_idx]
        # recuperación > +5% desde el mínimo
        return bool(last60.iloc[-1] >= trough_price * 1.05)
    return False

# -----------------------------
# Ejecución
# -----------------------------
st.title("Screener NASDAQ Composite + S&P 500")
st.caption("Filtros: mcap, empleados, volumen $ 3m, volatilidad diaria y patrón de tendencia.")

if run_btn:
    if not FINNHUB_API_KEY:
        st.error("Debes introducir tu Finnhub API Key para obtener componentes y fundamentales.")
        st.stop()

    targets = []
    if "S&P 500 (^GSPC)" in universe_opt:
        with st.spinner("Descargando componentes S&P 500..."):
            spx = get_index_constituents("^GSPC")
            targets.extend(spx)
    if "NASDAQ Composite (^IXIC)" in universe_opt:
        with st.spinner("Descargando componentes NASDAQ Composite..."):
            ndx = get_index_constituents("^IXIC")
            targets.extend(ndx)

    symbols = sorted(list({t for t in targets}))
    if limit_tickers and limit_tickers > 0:
        symbols = symbols[:int(limit_tickers)]

    st.write(f"Total símbolos a procesar: {len(symbols)}")

    # Perfiles (mcap / empleados)
    with st.spinner("Descargando fundamentales (market cap, empleados)..."):
        prof = get_profiles_bulk(symbols)
        prof = prof.drop_duplicates(subset=["symbol"]).set_index("symbol")

    # Pre-filtrado por mcap y empleados
    base = prof.copy()
    base = base[(base["market_cap"] >= min_mcap_input) & (base["employees"] >= min_employees)]
    base_symbols = base.index.tolist()

    st.write(f"Símbolos tras mcap/empleados: {len(base_symbols)}")

    if len(base_symbols) == 0:
        st.warning("Ningún símbolo supera mcap/empleados con los parámetros actuales.")
        st.stop()

    # Precios
    with st.spinner("Descargando precios (yfinance, 1y diario)... puede tardar"):
        data = get_prices(base_symbols, period="1y")

    # Estructura de yfinance cambia si hay un solo ticker
    def get_series(tk, field):
        if len(base_symbols) == 1:
            return data[field]
        else:
            return data[(tk, field)]

    rows = []
    for tk in base_symbols:
        try:
            close = get_series(tk, "Close")
            volume = get_series(tk, "Volume")
            if close.isna().all() or volume.isna().all():
                continue

            avg_dollar_vol = compute_dollar_vol(close, volume, window=63)
            vol_ratio = ratio_large_moves(close, window=vol_window, pct=vol_day_move)

            # Tendencia
            pass_trend = True
            if trend_mode == "Lateral-alcista (suave)":
                pass_trend = lateral_up_filter(close)
            elif trend_mode == "Alcista tras caída brusca (>=10%)":
                pass_trend = post_drop_filter(close)

            rows.append({
                "symbol": tk,
                "name": base.loc[tk, "name"],
                "market_cap": base.loc[tk, "market_cap"],
                "employees": base.loc[tk, "employees"],
                "avg_dollar_vol_63d": avg_dollar_vol,
                "pct_days_|ret|>={}%".format(vol_day_move): vol_ratio,
                "trend_ok": pass_trend
            })
        except Exception:
            continue

    out = pd.DataFrame(rows)

    # Filtros finales
    out = out[(out["avg_dollar_vol_63d"] >= min_avg_dollar_vol) & (out["pct_days_|ret|>={}%".format(vol_day_move)] >= vol_day_ratio)]

    if trend_mode != "Cualquiera":
        out = out[out["trend_ok"] == True]

    out = out.sort_values(["trend_ok", "avg_dollar_vol_63d"], ascending=[False, False])

    st.subheader("Resultados")
    st.write(f"Candidatos: {len(out)}")
    st.dataframe(out.reset_index(drop=True))

    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV", data=csv, file_name="screener_resultados.csv", mime="text/csv")

    st.caption("Nota: este screener es heurístico. Valida siempre con tu propia diligencia.")
else:
    st.info("Introduce tu API key de Finnhub, ajusta parámetros y pulsa 'Ejecutar screener'.")
