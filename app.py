# app.py — Finnhub test mínimo (Ticker NASDAQ)
# 1) En Streamlit Cloud → Settings → Secrets:
#    FINNHUB_API_KEY = tu_clave
# 2) requirements.txt más abajo
# 3) Deploy (o local: streamlit run app.py)

import os
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Finnhub · Test NASDAQ", layout="wide")

# ---------- Sidebar ----------
st.sidebar.header("Conexión")
api_key_input = st.sidebar.text_input("Finnhub API Key", type="password")
api_key = api_key_input or st.secrets.get("FINNHUB_API_KEY", "") or os.getenv("FINNHUB_API_KEY", "")
base_url = "https://finnhub.io/api/v1"
headers = {"X-Finnhub-Token": api_key} if api_key else None

st.sidebar.header("Parámetros")
symbol = st.sidebar.text_input("Ticker (NASDAQ)", value="AAPL").strip().upper()
period_days = st.sidebar.slider("Días de histórico (velas D)", min_value=60, max_value=730, value=365, step=5)
run = st.sidebar.button("▶ Cargar")

st.title("Finnhub · Test de conexión y datos de un ticker NASDAQ")
st.caption("Visualiza profile2, quote, metric (all) y velas diarias para validar la API y las variables.")

# ---------- Utils ----------
def fh_get(path: str, params: dict):
    if not api_key:
        st.error("Falta API key de Finnhub. Ponla en el campo o en Secrets.")
        st.stop()
    url = f"{base_url}/{path}"
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        st.write({"endpoint": path, "status": r.status_code, "url": r.url})
        if r.status_code == 429:
            st.error("Finnhub 429 (rate limit). Espera 1-2 min y reintenta.")
            st.stop()
        r.raise_for_status()
        try:
            return r.json()
        except Exception as e:
            st.error(f"JSONDecodeError: {e}")
            st.text(r.text[:500])
            st.stop()
    except requests.HTTPError:
        st.error(f"HTTPError {r.status_code}. Body (primeros 500 chars):")
        st.text(r.text[:500])
        st.stop()
    except Exception as e:
        st.error(f"Error en petición {path}: {e}")
        st.stop()

# ---------- Run ----------
if run:
    st.write(f"**Ticker:** {symbol}")
    st.write(f"**API key presente:** {'sí' if bool(api_key) else 'no'}")

    # QUOTE
    st.subheader("/quote")
    qt = fh_get("quote", {"symbol": symbol})  # c (last), d, dp, h, l, o, pc
    st.table(pd.DataFrame.from_dict(qt, orient="index", columns=["valor"]))
    with st.expander("JSON completo (quote)"):
        st.json(qt)

    # PROFILE2
    st.subheader("/stock/profile2")
    prof = fh_get("stock/profile2", {"symbol": symbol})
    # Campos útiles si existen
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

    # METRIC (all)  → puede estar limitado en plan free
    st.subheader("/stock/metric?metric=all")
    met = {}
    try:
        met = fh_get("stock/metric", {"symbol": symbol, "metric": "all"}) or {}
    except Exception:
        met = {}
    if met:
        lvl1 = list(met.keys())
        st.write({"claves_nivel1": lvl1})
        if isinstance(met.get("metric"), dict) and met["metric"]:
            dfm = pd.DataFrame.from_dict(met["metric"], orient="index", columns=["valor"]).sort_index()
            st.dataframe(dfm)
        with st.expander("JSON completo (metric)"):
            st.json(met)
    else:
        st.info("Sin métricas (endpoint puede estar restringido en plan gratuito).")

    # CANDLES (D)
    st.subheader("/stock/candle (diario)")
    to_ts = int(datetime.utcnow().timestamp())
    from_ts = int((datetime.utcnow() - timedelta(days=period_days)).timestamp())
    cd = fh_get("stock/candle", {"symbol": symbol, "resolution": "D", "from": from_ts, "to": to_ts})
    st.write({k: (len(v) if isinstance(v, list) else v) for k, v in cd.items()})

    if cd.get("s") == "ok":
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
            st.line_chart(df["c"], height=240)

            # Cálculos básicos (para futuros filtros)
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
        st.warning("Candles sin datos (s != 'ok'). Prueba otro ticker o aumenta días.")
else:
    st.info("Pon tu FINNHUB_API_KEY (o en Secrets), el símbolo (ej. AAPL) y pulsa ▶ Cargar.")
