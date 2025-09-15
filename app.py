# app.py — Finnhub minimal (test NASDAQ)
# Cómo usar en Streamlit Cloud:
# 1) Añade este archivo como app.py en tu repo
# 2) Crea requirements.txt (contenido en el chat)
# 3) En Settings → Secrets añade: FINNHUB_API_KEY = tu_clave
# 4) Deploy

import time
import math
import json
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="Finnhub · Test NASDAQ", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.header("Conexión")
api_key = st.sidebar.text_input("Finnhub API Key", type="password")
if not api_key:
    api_key = st.secrets.get("FINNHUB_API_KEY", "")
base_url = "https://finnhub.io/api/v1"
headers = {"X-Finnhub-Token": api_key} if api_key else None

st.sidebar.header("Parámetros")
symbol = st.sidebar.text_input("Ticker (NASDAQ)", value="AAPL").strip().upper()
period_days = st.sidebar.slider("Días de histórico (velas D)", min_value=60, max_value=730, value=365, step=5)
run = st.sidebar.button("▶ Cargar")

st.title("Finnhub · Test de conexión y datos de un NASDAQ ticker")
st.caption("Muestra perfil, cotización, métricas y velas diarias. Sirve como base para futuros filtros.")

# ------------- Utils -------------

def fh_get(path: str, params: dict):
    if not api_key:
        st.error("Falta API key de Finnhub.")
        st.stop()
    url = f"{base_url}/{path}"
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code == 429:
            st.error("Finnhub 429 (rate limit). Espera un momento y reintenta.")
            st.stop()
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        # Muestra mensaje corto con código
        st.error(f"HTTPError {r.status_code} en {path}: {r.text[:200]}")
        st.stop()
    except Exception as e:
        st.error(f"Error en petición {path}: {e}")
        st.stop()

@st.cache_data(show_spinner=False)
def fetch_all(symbol: str, period_days: int):
    out = {}
    # Perfil de la empresa
    out["profile2"] = fh_get("stock/profile2", {"symbol": symbol})
    # Cotización actual
    out["quote"] = fh_get("quote", {"symbol": symbol})
    # Métricas (puede estar limitado en plan free)
    try:
        out["metric_all"] = fh_get("stock/metric", {"symbol": symbol, "metric": "all"})
    except Exception:
        out["metric_all"] = {}
    # Velas diarias
    to_ts = int(datetime.utcnow().timestamp())
    from_ts = int((datetime.utcnow() - timedelta(days=period_days)).timestamp())
    candles = fh_get("stock/candle", {
        "symbol": symbol,
        "resolution": "D",
        "from": from_ts,
        "to": to_ts,
    })
    out["candles_raw"] = candles
    # Convertir velas a DataFrame
    if candles and candles.get("s") == "ok":
        df = pd.DataFrame({
            "t": candles.get("t", []),
            "o": candles.get("o", []),
            "h": candles.get("h", []),
            "l": candles.get("l", []),
            "c": candles.get("c", []),
            "v": candles.get("v", []),
        })
        if not df.empty:
            df["date"] = pd.to_datetime(df["t"], unit="s")
            df = df.set_index("date").drop(columns=["t"]).sort_index()
        out["candles_df"] = df
    else:
        out["candles_df"] = pd.DataFrame()
    return out

# ------------- Run -------------
if run:
    st.write(f"**Ticker:** {symbol}")
    data = fetch_all(symbol, period_days)

    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Perfil (profile2)")
        prof = data.get("profile2", {})
        if prof:
            # Muestra campos clave si existen
            campos = {
                "name": prof.get("name"),
                "ticker": prof.get("ticker"),
                "exchange": prof.get("exchange"),
                "marketCapitalization": prof.get("marketCapitalization"),
                "employeeTotal": prof.get("employeeTotal"),
                "ipo": prof.get("ipo"),
                "country": prof.get("country"),
                "finnhubIndustry": prof.get("finnhubIndustry"),
            }
            st.table(pd.DataFrame.from_dict(campos, orient="index", columns=["valor"]))
        with st.expander("JSON completo (profile2)"):
            st.json(prof)

    with col2:
        st.subheader("Cotización (quote)")
        qt = data.get("quote", {})
        if qt:
            # Finnhub devuelve: c(último), d/cambio abs, dp/% cambio, h/l/o pc...
            st.table(pd.DataFrame.from_dict(qt, orient="index", columns=["valor"]))
        with st.expander("JSON completo (quote)"):
            st.json(qt)

    st.subheader("Métricas (stock/metric — puede estar limitado en free)")
    met = data.get("metric_all", {}) or {}
    if met:
        # Keys de primer nivel: metric, metricType, series
        # Mostramos listado de claves disponibles para que sepas qué filtrar
        keys_lv1 = list(met.keys())
        st.write({"claves_nivel1": keys_lv1})
        # Si trae 'metric', lo tabulamos
        if isinstance(met.get("metric"), dict) and met["metric"]:
            dfm = pd.DataFrame.from_dict(met["metric"], orient="index", columns=["valor"]).sort_index()
            st.dataframe(dfm)
        with st.expander("JSON completo (metric)"):
            st.json(met)
    else:
        st.info("Sin métricas (este endpoint puede estar restringido en el plan gratuito).")

    st.subheader("Velas diarias y cálculo básico")
    df = data.get("candles_df", pd.DataFrame())
    if df is not None and not df.empty:
        st.line_chart(df["c"], height=240)
        # Cálculos básicos (para tus futuros filtros)
        try:
            # Volumen en $ ~ cierre * volumen (promedio 63 sesiones)
            df["dollar_vol"] = df["c"] * df["v"].fillna(0)
            avg_dollar_vol_63d = float(df["dollar_vol"].tail(63).mean()) if len(df) >= 10 else float("nan")
            # % de días con |ret| >= 2.5% (últimos 63)
            ret = df["c"].pct_change().abs()
            ratio_25 = 100.0 * (ret.tail(63) >= 0.025).mean()
            st.write({
                "avg_dollar_vol_63d": avg_dollar_vol_63d,
                "pct_days_|ret|>=2.5% (63d)": ratio_25,
                "n_rows": len(df)
            })
        except Exception as e:
            st.warning(f"No se pudieron calcular métricas básicas: {e}")
        with st.expander("DataFrame velas (head)"):
            st.dataframe(df.head())
    else:
        st.warning("Finnhub devolvió sin velas (candles). Prueba con otro símbolo o aumenta el rango de días.")
else:
    st.info("Introduce tu FINNHUB_API_KEY, el símbolo (ej. AAPL) y pulsa 'Cargar'.")
