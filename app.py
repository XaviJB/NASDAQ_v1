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

# Fallback de datos (Stooq) si Yahoo falla
try:
    import pandas_datareader.data as pdr
    HAVE_PDR = True
except Exception:
    HAVE_PDR = False

st.set_page_config(page_title="Screener NASDAQ + S&P500", layout="wide")

# -----------------------------
# Sidebar — Parámetros
# -----------------------------
st.sidebar.header("Parámetros del filtro")
source_mode = st.sidebar.selectbox(
    "Fuente de componentes del índice",
    [
        "Finnhub (requiere acceso al endpoint de índices)",
        "Gratis: SP500 (Wikipedia/yfinance) + Nasdaq (NasdaqTrader)"
    ],
    help="El plan gratuito de Finnhub suele bloquear /index/constituents. Usa el modo Gratis si te da HTTPError."
)
FINNHUB_API_KEY = st.sidebar.text_input("Finnhub API Key (requerida)", type="password")
# También admite Secrets en Streamlit Cloud si dejas el campo vacío
if not FINNHUB_API_KEY:
    FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", "")

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

PUBLIC_WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
PUBLIC_NASDAQ_LIST = "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

@st.cache_data(show_spinner=False)
def get_index_constituents(symbol: str) -> list:
    """Obtiene los componentes de un índice desde Finnhub (p.ej. ^GSPC, ^NDX). Requiere endpoint habilitado."""
    url = f"{FINNHUB_BASE}/index/constituents"
    r = requests.get(url, params={"symbol": symbol}, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    syms = data.get("constituents") or []
    return [s for s in syms if isinstance(s, str) and len(s) > 0]

@st.cache_data(show_spinner=False)
def get_sp500_public() -> list:
    """SP500 desde yfinance / Wikipedia (gratis)."""
    try:
        syms = yf.tickers_sp500()
        if syms: return syms
    except Exception:
        pass
    try:
        tables = pd.read_html(PUBLIC_WIKI_SP500)
        df = tables[0]
        col = [c for c in df.columns if str(c).lower().startswith("symbol")][0]
        return df[col].astype(str).str.replace(".", "-", regex=False).tolist()
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def get_nasdaq_public() -> list:
    """Nasdaq Composite aproximado vía lista oficial de NasdaqTrader (gratis)."""
    try:
        r = requests.get(PUBLIC_NASDAQ_LIST, timeout=30)
        r.raise_for_status()
        lines = r.text.strip().splitlines()
        out = []
        for ln in lines[1:]:
            parts = ln.split("|")
            if len(parts) < 5:
                continue
            sym = parts[0].strip()
            test_flag = parts[-2].strip()  # Test Issue (Y/N)
            if sym and test_flag != "Y":
                out.append(sym)
        return out
    except Exception:
        return []

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
    targets = []
    try:
        if source_mode.startswith("Finnhub"):
            if not FINNHUB_API_KEY:
                st.error("Falta FINNHUB_API_KEY. Cambia a modo Gratis o añade tu clave en Secrets.")
                st.stop()
            if "S&P 500 (^GSPC)" in universe_opt:
                with st.spinner("Descargando componentes S&P 500 (Finnhub)..."):
                    spx = get_index_constituents("^GSPC")
                    targets.extend(spx)
            if "NASDAQ Composite (^IXIC)" in universe_opt:
                with st.spinner("Descargando componentes NASDAQ Composite (Finnhub)..."):
                    ndx = get_index_constituents("^IXIC")
                    targets.extend(ndx)
        else:
            if "S&P 500 (^GSPC)" in universe_opt:
                with st.spinner("Descargando SP500 (Wikipedia/yfinance)..."):
                    targets.extend(get_sp500_public())
            if "NASDAQ Composite (^IXIC)" in universe_opt:
                with st.spinner("Descargando Nasdaq (NasdaqTrader)..."):
                    targets.extend(get_nasdaq_public())
    except requests.HTTPError as e:
        st.error("Finnhub devolvió HTTPError en /index/constituents (probable bloqueo de plan). Cambia a modo Gratis en la barra lateral o usa ^NDX (Nasdaq-100).")
        st.stop()

    symbols = sorted(list({t for t in targets}))
    if limit_tickers and limit_tickers > 0:
        symbols = symbols[:int(limit_tickers)]

    st.write(f"Total símbolos a procesar: {len(symbols)}")

    if len(symbols) == 0:
        st.error("No se han obtenido componentes. Revisa 'Universo de índices' o cambia de fuente (Gratis/Finnhub).")
        st.stop()

    # Perfiles (mcap / empleados)
    with st.spinner("Descargando fundamentales (market cap, empleados)..."):
        prof = None
        if FINNHUB_API_KEY and len(symbols) > 0:
            prof = get_profiles_bulk(symbols)
        if prof is None or prof.empty or ("symbol" not in prof.columns):
            if FINNHUB_API_KEY:
                st.warning("Perfiles vacíos o sin 'symbol'; usando fallback sin mcap/empleados.")
            prof = pd.DataFrame({
                "symbol": symbols,
                "name": [None]*len(symbols),
                "market_cap": [np.nan]*len(symbols),
                "employees": [np.nan]*len(symbols)
            })
        prof = prof.drop_duplicates(subset=["symbol"]).set_index("symbol")

    # Pre-filtrado por mcap y empleados (si hay datos)
    base = prof.copy()
    if prof["market_cap"].notna().any() and prof["employees"].notna().any():
        base = base[(base["market_cap"] >= min_mcap_input) & (base["employees"] >= min_employees)]
    base_symbols = base.index.tolist()

    # Limpieza de tickers problemáticos (warrants, units, rights, etc.) del NASDAQ
    bad_suffixes = ("W", "U", "R")
    base_symbols = [s for s in base_symbols if s.isalpha() and not s.endswith(bad_suffixes)]

    st.write(f"Símbolos tras mcap/empleados (y limpieza): {len(base_symbols)}")

    if len(base_symbols) == 0:
        st.warning("Ningún símbolo supera mcap/empleados con los parámetros actuales.")
        st.stop()

    # Precios
    with st.spinner("Descargando precios (yfinance, 1y diario, secuencial)..."):
        pass

    rows = []
    move_col_name = f"pct_days_|ret|>={vol_day_move:.2f}%"
    for tk in base_symbols:
        try:
            df = None
            # Intento Yahoo Finance (hasta 3 reintentos)
            for _ in range(3):
                try:
                    df = yf.download(tk, period="1y", interval="1d", auto_adjust=False, progress=False, threads=False)
                    if df is not None and not df.empty:
                        break
                except Exception:
                    time.sleep(0.3)
            # Fallback Stooq (si está disponible)
            if (df is None or df.empty) and 'HAVE_PDR' in globals() and HAVE_PDR:
                try:
                    end = datetime.today()
                    start = end - timedelta(days=365)
                    df = pdr.DataReader(f"{tk}.US", "stooq", start, end)
                    if df is not None and not df.empty:
                        df = df.sort_index()
                except Exception:
                    df = None

            if df is None or df.empty:
                continue

            close = df["Close"]
            volume = df["Volume"] if "Volume" in df.columns else pd.Series(index=close.index, data=np.nan)

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
                move_col_name: vol_ratio,
                "trend_ok": pass_trend
            })
        except Exception:
            continue

    out = pd.DataFrame(rows)

    # Filtros finales
    if out.empty:
        st.warning("No se calcularon métricas para ningún símbolo (datos de precios insuficientes o tickers sin volumen). Prueba a bajar filtros o ampliar universo/periodo.")
        st.stop()

    required_cols = ["avg_dollar_vol_63d", move_col_name]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        st.warning(f"Faltan columnas requeridas {missing}. Puede deberse a falta de datos. Muestra preliminar abajo.")
        st.dataframe(out)
        st.stop()

    out = out[(out["avg_dollar_vol_63d"] >= min_avg_dollar_vol) & (out[move_col_name] >= vol_day_ratio)]

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
