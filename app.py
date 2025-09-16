import os, sys, time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import finnhub
import pandas as pd

# --- Config ---
load_dotenv()
API_KEY = os.getenv("FINNHUB_API_KEY")
if not API_KEY:
    print("ERROR: falta FINNHUB_API_KEY en .env o variables de entorno.")
    sys.exit(1)

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "AAPL"  # NASDAQ por defecto

client = finnhub.Client(api_key=API_KEY)

def unix_ts(dt): return int(time.mktime(dt.timetuple()))

# --- 1) Perfil de compañía: campos útiles para filtrar ---
profile = client.company_profile2(symbol=SYMBOL) or {}
print("\n=== Company Profile 2 ===")
for k in ["name","ticker","exchange","ipo","finnhubIndustry","country",
          "marketCapitalization","shareOutstanding","employeeTotal","weburl"]:
    if k in profile: print(f"{k}: {profile.get(k)}")

# --- 2) Velas diarias últimos ~4 meses para calcular métricas ---
to_dt = datetime.utcnow()
from_dt = to_dt - timedelta(days=140)  # margen para ~63 sesiones hábiles
candles = client.stock_candles(SYMBOL, "D", unix_ts(from_dt), unix_ts(to_dt))

if candles.get("s") != "ok" or not candles.get("c"):
    print("\nERROR al obtener velas. Respuesta:", candles)
    sys.exit(1)

df = pd.DataFrame({
    "t": candles["t"],
    "o": candles["o"],
    "h": candles["h"],
    "l": candles["l"],
    "c": candles["c"],
    "v": candles["v"],
})
df["date"] = pd.to_datetime(df["t"], unit="s")
df = df.sort_values("date").tail(65)  # ~3 meses hábiles

# --- 3) Métricas de test para tu screener ---
# Volumen $ 3m ≈ suma(cierre * volumen) de ~3 meses
df["dollar_vol"] = df["c"] * df["v"]
dollar_vol_3m = float(df["dollar_vol"].sum())

# Volatilidad diaria (desv. típica de rendimientos diarios)
df["ret"] = df["c"].pct_change()
daily_vol = float(df["ret"].std(skipna=True))

out = {
    "symbol": SYMBOL,
    "rows_used": int(df.shape[0]),
    "dollar_volume_3m": dollar_vol_3m,
    "daily_volatility": daily_vol,
}
print("\n=== Test metrics ===")
for k, v in out.items(): print(f"{k}: {v}")

# --- 4) Vista rápida de columnas disponibles (para futuros filtros) ---
print("\n=== Columns available from candles ===")
print(list(df.columns))

print("\nOK: conexión y métricas calculadas.")
