# app.py — Finnhub + fallback a yfinance (test NASDAQ)
# 1) En Streamlit Cloud → Settings → Secrets:
#    FINNHUB_API_KEY = "tu_clave"
# 2) requirements.txt más abajo
# 3) Deploy (o local: streamlit run app.py)

import os, time
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Finnhub · Test NASDAQ", layout="wide")

# ---------- Sidebar ----------
st.sidebar.header("Conexión")
api_key_input = st.sidebar.text_input("Finnhub API Key", type="password")
api_key = api_key_input or st.secrets.get("FINNHUB_API_KEY", "") or os.getenv("FINNHUB_API_KEY", "")
BASE = "https://finnhub.io/api/v1"
headers = {"X-Finnhub-Token": api_key} if api_key else None

st.sidebar.header("Parámetros")
symbol = st.sidebar.text_input("Ticker (NASDAQ)", value="AAPL").strip().upper()
period_days = st.sidebar.slider("Días de histórico (velas D)", min_value=60, max_value=730, value=365, step=5)
run = st.sidebar.button("▶ Cargar")

st.title("Finnhub · Test de conexión y datos de un ticker NASDAQ")
st.caption("Usa Finnhub para profile/quote/metric y yfinance como fallback para velas diarias.")

# ---------- Utils ----------
def fh_get(path: str, params: dict):
    if not api_key:
        st.error("Falta API key de Finnhub. Ponla en el campo o en Secrets.")
        st.stop()
    url = f"{BASE}/{path}"
    r = requests.get(url, params=params, headers=headers, timeout=20)
    st.write({"endpoint": path, "status": r.status_code, "url": r.url})
    if r.status_code == 429:
        st.error("Finnhub 429 (rate limit). Espera 1–2 min y reintenta.")
        st.stop()
    try:
        r.raise_for_status()
    except requests.HTTPError:
        # Devolvemos el cuerpo para diagnosticar y que el caller decida fallback
        raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
    try:
        return r.json()
    except Exception as e:
        raise RuntimeError(f"JSONDecodeError: {e}")

def get_candles(symbol: str, days: int) -> tuple[pd.DataFrame, str]:
    """Devuelve (df, fuente). Intenta Finnhub; si 403/otro → yfinance."""
    # 1) Finnhub (puede fallar por permisos del plan)
    if api_key:
        try:
            to_ts = int(time.time())
            from_ts = to_ts - days * 86400
            cd = fh_get("stock/candle", {"symbol": symbol, "resolution": "D",
                                         "from": from_ts, "to": to_ts})
            if cd and cd.get("s") == "ok":
                df = pd.DataFrame({
                    "t": cd.get("t", []),
                    "o": cd.get("o", []),
                    "h": cd.get("h", []),
                    "l": cd.get("l", []),
                    "c": cd.get("c", []),
                    "v": cd.get("v", []),
                })
                if not df.empty:
                    df["date"] = pd.to_datetime(df["t"], unit="s")
                    df = df.set_index("date").drop(columns=["t"]).sort_index()
                    return df, "finnhub"
        except Exception as e:
            st.info(f"Finnhub candles no disponible ({e}). Uso fallback a yfinance…")

    # 2) yfinance (histórico diario)
    try:
        # limitar a <= 729 días por si acaso
        dfy = yf.download(symbol, period=f"{min(days, 729)}d",
                          interval="1d", auto_adjust=False,
                          progress=False, threads=False)
        if dfy is not None and not dfy.empty:
            dfy = dfy.rename(columns={"Open":"o","High":"h","Low":"l","Close":"c","Volume":"v"})
            return dfy[["o","h","l","c","v"]].copy(), "yfinance"
    except Exception as e:
        st.error(f"yfinance falló: {e}")

    return pd.DataFrame(), "none"

# ---------- Run ----------
if run:
    st.write(f"**Ticker:** {symbol}")
    st.write(f"**API key presente:** {'sí' if bool(api_key) else 'no'}")

    # QUOTE
    try:
        qt = fh_get("quote", {"symbol": symbol})
        qdf = pd.DataFrame.from_dict(qt, orient="index", columns=["valor"]).astype({"valor":"string"})
        st.subheader("/quote"); st.dataframe(qdf, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudo obtener /quote: {e}")

    # PROFILE2
    try:
        prof = fh_get("stock/profile2", {"symbol": symbol})
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
        pdf = pd.DataFrame.from_dict(campos, orient="index", columns=["valor"]).astype({"valor":"string"})
        st.subheader("/stock/profile2"); st.dataframe(pdf, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudo obtener /stock/profile2: {e}")

    # METRIC (all) (puede no estar en el plan)
    st.subheader("/stock/metric?metric=all")
    met = {}
    try:
        met = fh_get("stock/metric", {"symbol": symbol, "metric": "all"}) or {}
    except Exception as e:
        st.info(f"Sin métricas (posible restricción de plan): {e}")
    if met and isinstance(met.get("metric"), dict):
        dfm = pd.DataFrame.from_dict(met["metric"], orient="index", columns=["valor"]).sort_index()
        st.dataframe(dfm, use_container_width=True)

    # CANDLES (D) con fallback
    st.subheader("Velas diarias")
    df, src = get_candles(symbol, period_days)
    if not df.empty:
        st.caption(f"Fuente histórico: {src}")
        st.line_chart(df["c"], height=240)

        # Métricas básicas para filtros posteriores
        try:
            df["dollar_vol"] = df["c"] * df["v"].fillna(0)
            avg_dollar_vol_63d = float(df["dollar_vol"].tail(63).mean()) if len(df) >= 10 else float("nan")
            ret_abs = df["c"].pct_change().abs()
            ratio_25 = 100.0 * (ret_abs.tail(63) >= 0.025).mean()
            st.write({
                "avg_dollar_vol_63d": avg_dollar_vol_63d,
                "pct_days_|ret|>=2.5%_last63d": ratio_25,
                "rows": int(len(df)),
            })
        except Exception as e:
            st.warning(f"No se pudieron calcular métricas básicas: {e}")
        with st.expander("DataFrame velas (head)"):
            st.dataframe(df.head())
    else:
        st.error("No hay datos históricos ni por Finnhub ni por yfinance.")
else:
    st.info("Pon tu FINNHUB_API_KEY (o en Secrets), el símbolo (ej. AAPL) y pulsa ▶ Cargar.")
