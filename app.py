"""Stock screener for Nasdaq Composite and S&P 500 using Finnhub data.

This module implements a command line utility that retrieves the
constituents of the Nasdaq Composite and S&P 500 indices and filters them
based on:

* Market capitalisation
* Average daily traded dollar volume over the last three months
* Number of employees
* Daily volatility (percentage of days with >= 2.5% moves)
* Trend heuristics (gentle upward drift or recovery after a >10% drawdown)

To run it you need a Finnhub API key stored in the environment variable
``FINNHUB_API_KEY``. You can place it in a `.env` file next to this
script or export it before running the tool.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import warnings
import time

import numpy as np
import pandas as pd
import finnhub
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

INDEX_SYMBOLS: Dict[str, str] = {
    "nasdaq_composite": "^IXIC",
    "sp500": "^GSPC",
}


@dataclass(frozen=True)
class ScreeningConfig:
    """Configuration values for the screener."""

    market_cap_min: float = 1_500_000  # USD
    avg_dollar_volume_min: float = 800_000  # USD
    employees_min: int = 150
    volatility_threshold: float = 0.025
    volatility_share_min: float = 0.40
    lookback_months: int = 3


def load_api_key(env_var: str = "FINNHUB_API_KEY") -> str:
    """Load Finnhub API key from the environment or a nearby .env file.

    Parameters
    ----------
    env_var:
        Environment variable name containing the API key.

    Returns
    -------
    str
        The API key.

    Raises
    ------
    RuntimeError
        If the key cannot be found.
    """

    # First try to load a .env file sitting next to this script. Falling
    # back to the default discovery if that is not present keeps things
    # flexible while avoiding crashes under some environments (e.g. in
    # tests where automatic discovery may fail).
    module_path = Path(__file__).resolve().parent
    env_path = module_path / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(
            "Finnhub API key not found. Please set the FINNHUB_API_KEY "
            "environment variable or provide a .env file."
        )
    return api_key


def get_client() -> finnhub.Client:
    """Instantiate a Finnhub client using the configured API key."""

    api_key = load_api_key()
    return finnhub.Client(api_key=api_key)


def months_ago(months: int, reference: Optional[datetime] = None) -> datetime:
    """Return a naive UTC datetime ``months`` months before ``reference``.

    ``months`` are approximated as 30 days in order to keep the code
    lightweight. The logic only needs a rough window for querying daily
    candles, so this is accurate enough for the screening heuristics.
    """

    if reference is None:
        reference = datetime.utcnow()
    return reference - timedelta(days=30 * months)


def candles_to_frame(candles: Dict[str, Sequence[float]]) -> pd.DataFrame:
    """Convert a Finnhub candles response into a tidy DataFrame."""

    if not candles or candles.get("s") != "ok":
        return pd.DataFrame()

    frame = pd.DataFrame({
        "time": pd.to_datetime(candles["t"], unit="s"),
        "open": candles["o"],
        "high": candles["h"],
        "low": candles["l"],
        "close": candles["c"],
        "volume": candles["v"],
    })
    frame.sort_values("time", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


class IndexScreener:
    """Encapsulates the logic for screening index constituents."""

    def __init__(self, client: finnhub.Client, throttle: float = 0.0):
        self._client = client
        self._throttle = throttle

    def fetch_constituents(self, indices: Iterable[str]) -> List[str]:
        """Fetch and combine constituents from several indices."""

        tickers: List[str] = []
        for index_symbol in indices:
            try:
                response = self._client.index_constituents(index_symbol)
            except Exception as exc:  # pragma: no cover - network errors
                LOGGER.error("Failed to download constituents for %s: %s", index_symbol, exc)
                self._respect_throttle()
                continue
            members = response.get("constituents", [])
            LOGGER.info("Fetched %d tickers from %s", len(members), index_symbol)
            tickers.extend(members)
            self._respect_throttle()
        unique = sorted(set(tickers))
        LOGGER.info("Total unique tickers retrieved: %d", len(unique))
        return unique

    def screen_symbol(self, symbol: str, config: ScreeningConfig) -> Optional[Dict[str, float]]:
        """Return screening metrics for a single symbol if it passes the filters."""

        metrics = self._fetch_metrics(symbol)
        if metrics is None:
            return None

        market_cap = self._extract_market_cap(metrics)
        if market_cap is None or market_cap < config.market_cap_min:
            return None

        employees = self._extract_employees(metrics)
        if employees is None or employees < config.employees_min:
            return None

        candles = self._fetch_candles(symbol, config.lookback_months)
        if candles.empty:
            return None

        avg_dollar_volume = self._compute_average_dollar_volume(candles)
        if avg_dollar_volume < config.avg_dollar_volume_min:
            return None

        volatility_share, days_evaluated = self._compute_volatility_share(
            candles, threshold=config.volatility_threshold
        )
        if days_evaluated == 0 or volatility_share < config.volatility_share_min:
            return None

        trend_signal, total_return, max_drawdown = self._classify_trend(candles)
        if trend_signal == "none":
            return None

        result = {
            "symbol": symbol,
            "market_cap_usd": market_cap,
            "avg_dollar_volume": avg_dollar_volume,
            "employees": employees,
            "volatility_share": volatility_share,
            "volatility_days": days_evaluated,
            "trend_signal": trend_signal,
            "total_return_pct": total_return * 100.0,
            "max_drawdown_pct": max_drawdown * 100.0,
        }

        profile = self._fetch_profile(symbol)
        if profile:
            name = profile.get("name") or profile.get("ticker")
            if name:
                result["name"] = name

        return result

    # --- Data fetching helpers -------------------------------------------------
    def _fetch_profile(self, symbol: str) -> Optional[Dict[str, object]]:
        try:
            profile = self._client.company_profile2(symbol=symbol)
        except Exception as exc:  # pragma: no cover - network errors
            LOGGER.debug("Failed to fetch profile for %s: %s", symbol, exc)
            self._respect_throttle()
            return None
        if not profile:
            self._respect_throttle()
            return None
        self._respect_throttle()
        return profile

    def _fetch_metrics(self, symbol: str) -> Optional[Dict[str, object]]:
        try:
            data = self._client.company_basic_financials(symbol, "all")
        except Exception as exc:  # pragma: no cover - network errors
            LOGGER.debug("Failed to fetch metrics for %s: %s", symbol, exc)
            self._respect_throttle()
            return None
        metrics = data.get("metric") if isinstance(data, dict) else None
        if not metrics:
            self._respect_throttle()
            return None
        self._respect_throttle()
        return metrics

    def _fetch_candles(self, symbol: str, months: int) -> pd.DataFrame:
        end = datetime.utcnow()
        start = months_ago(months, reference=end)
        try:
            raw = self._client.stock_candles(
                symbol,
                "D",
                int(start.timestamp()),
                int(end.timestamp()),
            )
        except Exception as exc:  # pragma: no cover - network errors
            LOGGER.debug("Failed to fetch candles for %s: %s", symbol, exc)
            self._respect_throttle()
            return pd.DataFrame()
        self._respect_throttle()
        return candles_to_frame(raw)

    # --- Metric extraction helpers --------------------------------------------
    @staticmethod
    def _extract_market_cap(metrics: Dict[str, object]) -> Optional[float]:
        value = metrics.get("marketCapitalization")
        if value is None:
            return None
        # Finnhub expresses market cap in millions of USD for equities.
        market_cap = float(value) * 1_000_000
        return market_cap

    @staticmethod
    def _extract_employees(metrics: Dict[str, object]) -> Optional[int]:
        for key in ("numberOfEmployees", "employeeCount", "fullTimeEmployees"):
            if key in metrics and metrics[key] is not None:
                try:
                    return int(metrics[key])
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _compute_average_dollar_volume(candles: pd.DataFrame) -> float:
        if candles.empty:
            return 0.0
        dollar_volume = candles["close"] * candles["volume"]
        return float(dollar_volume.mean())

    @staticmethod
    def _compute_volatility_share(
        candles: pd.DataFrame,
        threshold: float,
    ) -> tuple[float, int]:
        closes = candles["close"].astype(float)
        if len(closes) < 2:
            return 0.0, 0
        returns = closes.pct_change().abs().dropna()
        if returns.empty:
            return 0.0, 0
        volatile_days = int((returns >= threshold).sum())
        total_days = int(returns.count())
        share = volatile_days / total_days if total_days else 0.0
        return share, total_days

    @staticmethod
    def _classify_trend(candles: pd.DataFrame) -> tuple[str, float, float]:
        closes = candles["close"].astype(float)
        if closes.empty:
            return "none", 0.0, 0.0

        start_price = float(closes.iloc[0])
        end_price = float(closes.iloc[-1])
        total_return = (end_price / start_price) - 1 if start_price else 0.0

        x = np.arange(len(closes), dtype=float)
        y = closes.to_numpy(dtype=float)
        slope = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            try:
                slope, _ = np.polyfit(x, y, 1)
            except Exception:  # pragma: no cover - numerical issues
                slope = 0.0
        slope_normalised = slope / start_price if start_price else 0.0

        # Gentle upward drift: positive slope, modest cumulative return.
        if slope_normalised > 0 and 0.02 <= total_return <= 0.25:
            return "gentle_uptrend", total_return, IndexScreener._compute_max_drawdown(closes)

        # Recovery after sharp drawdown.
        drawdown = IndexScreener._compute_max_drawdown(closes)
        if drawdown <= -0.10:
            trough_index = IndexScreener._drawdown_trough_index(closes)
            if trough_index is not None:
                trough_price = float(closes.iloc[trough_index])
                post_trough = closes.iloc[trough_index:]
                if not post_trough.empty:
                    recovered = post_trough.max() >= trough_price * 1.05 and end_price > trough_price
                    if recovered:
                        return "post_selloff_recovery", total_return, drawdown
        return "none", total_return, drawdown if 'drawdown' in locals() else 0.0

    @staticmethod
    def _compute_max_drawdown(series: pd.Series) -> float:
        rolling_max = series.cummax()
        drawdowns = series / rolling_max - 1.0
        return float(drawdowns.min()) if not drawdowns.empty else 0.0

    @staticmethod
    def _drawdown_trough_index(series: pd.Series) -> Optional[int]:
        if series.empty:
            return None
        rolling_max = series.cummax()
        drawdowns = series / rolling_max - 1.0
        if drawdowns.empty:
            return None
        trough_position = int(np.argmin(drawdowns.to_numpy(dtype=float)))
        return trough_position

    def _respect_throttle(self) -> None:
        if self._throttle > 0:
            time.sleep(self._throttle)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-market-cap",
        type=float,
        default=ScreeningConfig.market_cap_min,
        help="Minimum market capitalisation in USD (default: %(default)s)",
    )
    parser.add_argument(
        "--min-dollar-volume",
        type=float,
        default=ScreeningConfig.avg_dollar_volume_min,
        help="Minimum average daily dollar volume over the lookback window in USD",
    )
    parser.add_argument(
        "--min-employees",
        type=int,
        default=ScreeningConfig.employees_min,
        help="Minimum number of employees",
    )
    parser.add_argument(
        "--volatility-threshold",
        type=float,
        default=ScreeningConfig.volatility_threshold,
        help="Absolute daily return threshold to consider a day volatile (default: %(default)s)",
    )
    parser.add_argument(
        "--volatility-share",
        type=float,
        default=ScreeningConfig.volatility_share_min,
        help="Minimum fraction of volatile days required over the lookback window",
    )
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=ScreeningConfig.lookback_months,
        help="Number of months of daily data to analyse",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.0,
        help="Optional delay in seconds between API calls to respect rate limits",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of tickers screened (useful when testing to avoid rate limits)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the filtered tickers as CSV",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def execute_screening(
    config: ScreeningConfig,
    *,
    throttle: float = 0.0,
    limit: Optional[int] = None,
    client: Optional[finnhub.Client] = None,
) -> pd.DataFrame:
    """Run the screening workflow and return the resulting DataFrame."""

    if client is None:
        client = get_client()
    screener = IndexScreener(client, throttle=throttle)

    tickers = screener.fetch_constituents(INDEX_SYMBOLS.values())
    results: List[Dict[str, float]] = []

    for idx, symbol in enumerate(tickers, start=1):
        record = screener.screen_symbol(symbol, config)
        if record:
            results.append(record)
        if limit and idx >= limit:
            break

    frame = pd.DataFrame(results)
    if not frame.empty:
        # Provide more user-friendly units.
        frame["market_cap_musd"] = frame["market_cap_usd"] / 1_000_000
        frame.sort_values("volatility_share", ascending=False, inplace=True)
        columns = [
            "symbol",
            "name" if "name" in frame.columns else None,
            "market_cap_musd",
            "avg_dollar_volume",
            "employees",
            "volatility_share",
            "volatility_days",
            "trend_signal",
            "total_return_pct",
            "max_drawdown_pct",
        ]
        frame = frame[[col for col in columns if col in frame.columns]]
    return frame


def run_screening(args: argparse.Namespace) -> pd.DataFrame:
    configure_logging(verbose=args.verbose)

    config = ScreeningConfig(
        market_cap_min=args.min_market_cap,
        avg_dollar_volume_min=args.min_dollar_volume,
        employees_min=args.min_employees,
        volatility_threshold=args.volatility_threshold,
        volatility_share_min=args.volatility_share,
        lookback_months=args.lookback_months,
    )

    frame = execute_screening(
        config,
        throttle=args.throttle,
        limit=args.limit,
    )
    return frame


def _is_running_in_streamlit() -> bool:
    """Return ``True`` when executed from ``streamlit run``."""

    if os.environ.get("STREAMLIT_RUNTIME"):
        return True
    if os.environ.get("STREAMLIT_SERVER_RUNNING"):
        return True
    if "streamlit" in sys.modules:
        return True
    return any(
        arg.startswith("--server.") or arg.startswith("--browser.")
        for arg in sys.argv[1:]
    )


def run_streamlit_app() -> None:
    """Render the screener with a Streamlit interface."""

    import streamlit as st

    st.set_page_config(page_title="Nasdaq & S&P 500 Screener", layout="wide")
    st.title("Screener Nasdaq Composite y S&P 500")
    st.write(
        "Ajusta los filtros para localizar valores del Nasdaq Composite y del "
        "S&P 500 que cumplan los requisitos de capitalización, liquidez y "
        "volatilidad solicitados."
    )

    default_config = ScreeningConfig()

    with st.sidebar:
        st.header("Filtros")
        st.caption(
            "Recuerda definir la variable de entorno `FINNHUB_API_KEY` con tu "
            "clave personal de Finnhub antes de ejecutar la aplicación."
        )

        market_cap_min_musd = st.number_input(
            "Capitalización mínima (millones USD)",
            min_value=0.0,
            value=default_config.market_cap_min / 1_000_000,
            step=10.0,
            help=(
                "Filtra compañías con una capitalización de mercado superior al "
                "valor indicado."
            ),
        )
        avg_dollar_volume_min_musd = st.number_input(
            "Volumen medio diario (millones USD)",
            min_value=0.0,
            value=default_config.avg_dollar_volume_min / 1_000_000,
            step=0.5,
            format="%.2f",
            help=(
                "Promedio del volumen diario negociado en dólares durante la "
                "ventana de análisis."
            ),
        )
        employees_min = st.number_input(
            "Número mínimo de empleados",
            min_value=0,
            value=default_config.employees_min,
            step=10,
        )
        volatility_threshold_pct = st.slider(
            "Movimiento mínimo para considerar un día volátil",
            min_value=0.5,
            max_value=10.0,
            value=default_config.volatility_threshold * 100,
            step=0.5,
            format="%.1f%%",
            help=(
                "Porcentaje de variación diaria absoluta utilizado como umbral "
                "de volatilidad."
            ),
        )
        volatility_share_pct = st.slider(
            "Porcentaje mínimo de días volátiles",
            min_value=0.0,
            max_value=100.0,
            value=default_config.volatility_share_min * 100,
            step=5.0,
            format="%.0f%%",
            help=(
                "Porcentaje de días con movimientos superiores al umbral "
                "durante el periodo analizado."
            ),
        )
        lookback_months = st.slider(
            "Meses a analizar",
            min_value=1,
            max_value=12,
            value=default_config.lookback_months,
            help="Cantidad de meses de datos diarios a solicitar.",
        )
        throttle = st.slider(
            "Pausa entre peticiones (segundos)",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help=(
                "Añade una espera entre peticiones para respetar los límites de "
                "la API."
            ),
        )
        limit = st.number_input(
            "Limitar símbolos analizados (0 = sin límite)",
            min_value=0,
            max_value=5000,
            value=0,
            step=10,
            help=(
                "Reduce el número de tickers evaluados, útil para pruebas o "
                "cuando te acerques al límite de peticiones."
            ),
        )
        run_clicked = st.button("Ejecutar screener", type="primary")

    limit_value = int(limit) if limit else None
    config_kwargs = {
        "market_cap_min": market_cap_min_musd * 1_000_000,
        "avg_dollar_volume_min": avg_dollar_volume_min_musd * 1_000_000,
        "employees_min": int(employees_min),
        "volatility_threshold": volatility_threshold_pct / 100.0,
        "volatility_share_min": volatility_share_pct / 100.0,
        "lookback_months": int(lookback_months),
    }

    @st.cache_data(show_spinner=False)
    def _cached_screening(
        config_items: tuple[tuple[str, float], ...],
        throttle_value: float,
        limit_value: Optional[int],
    ) -> pd.DataFrame:
        kwargs = {key: value for key, value in config_items}
        config_obj = ScreeningConfig(**kwargs)
        return execute_screening(
            config_obj,
            throttle=throttle_value,
            limit=limit_value,
        )

    had_error = False
    if run_clicked:
        try:
            with st.spinner("Descargando datos desde Finnhub..."):
                frame = _cached_screening(
                    tuple(sorted(config_kwargs.items())),
                    throttle,
                    limit_value,
                )
        except RuntimeError as exc:
            had_error = True
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover - UI-only feedback
            LOGGER.exception("Unexpected error while running Streamlit app: %s", exc)
            had_error = True
            st.error(f"Se produjo un error inesperado: {exc}")
        else:
            st.session_state["screening_results"] = frame
            st.session_state["screening_config"] = config_kwargs
            st.session_state["screening_limit"] = limit_value

    results = st.session_state.get("screening_results")
    if results is None:
        if had_error:
            st.info("No se pudieron obtener datos con los filtros actuales.")
        else:
            st.info("Configura los filtros y pulsa **Ejecutar screener** para comenzar.")
        return

    frame = results
    if frame.empty:
        st.warning("Ningún valor cumple los criterios establecidos.")
        return

    st.success(f"Se encontraron {len(frame)} valores que cumplen los filtros actuales.")
    if had_error:
        st.caption("Los resultados corresponden a la última ejecución correcta.")

    display_frame = frame.copy()
    rename_map = {
        "symbol": "Ticker",
        "name": "Nombre",
        "market_cap_musd": "Capitalización (M USD)",
        "avg_dollar_volume": "Volumen medio (USD)",
        "employees": "Empleados",
        "volatility_share": "Días volátiles",
        "volatility_days": "Días evaluados",
        "trend_signal": "Tendencia",
        "total_return_pct": "Rentabilidad total (%)",
        "max_drawdown_pct": "Drawdown máximo (%)",
    }
    display_frame.rename(
        columns={key: value for key, value in rename_map.items() if key in display_frame.columns},
        inplace=True,
    )
    formatters = {}
    if "Capitalización (M USD)" in display_frame.columns:
        formatters["Capitalización (M USD)"] = "{:,.2f}".format
    if "Volumen medio (USD)" in display_frame.columns:
        formatters["Volumen medio (USD)"] = "{:,.0f}".format
    if "Días volátiles" in display_frame.columns:
        formatters["Días volátiles"] = "{:.1%}".format
    if "Días evaluados" in display_frame.columns:
        formatters["Días evaluados"] = "{:.0f}".format
    if "Rentabilidad total (%)" in display_frame.columns:
        formatters["Rentabilidad total (%)"] = "{:.2f}".format
    if "Drawdown máximo (%)" in display_frame.columns:
        formatters["Drawdown máximo (%)"] = "{:.2f}".format

    st.dataframe(
        display_frame.style.format(formatters),
        use_container_width=True,
    )

    csv_data = frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar resultados en CSV",
        data=csv_data,
        file_name="screener_resultados.csv",
        mime="text/csv",
    )

    config_used = st.session_state.get("screening_config", config_kwargs)
    limit_used = st.session_state.get("screening_limit")
    st.caption(
        "Filtros aplicados: capitalización mínima "
        f"{config_used['market_cap_min']/1_000_000:,.2f} M USD, volumen medio "
        f"{config_used['avg_dollar_volume_min']/1_000_000:,.2f} M USD, al menos "
        f"{config_used['employees_min']} empleados, {config_used['volatility_share_min']*100:.0f}% "
        "de días volátiles con movimientos superiores a "
        f"{config_used['volatility_threshold']*100:.1f}% en {config_used['lookback_months']} meses."
    )
    if limit_used:
        st.caption(
            "La ejecución se detuvo tras analizar "
            f"{limit_used} símbolos por el límite configurado."
        )


def main() -> None:
    args = parse_arguments()
    frame = run_screening(args)
    if frame.empty:
        print("No tickers met the screening criteria.")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(frame.to_string(index=False, justify="center", formatters={
        "market_cap_musd": "{:.2f}".format,
        "avg_dollar_volume": "{:.0f}".format,
        "volatility_share": "{:.2%}".format,
        "total_return_pct": "{:.2f}".format,
        "max_drawdown_pct": "{:.2f}".format,
    }))

    if args.output:
        frame.to_csv(args.output, index=False)
        print(f"Saved {len(frame)} tickers to {args.output}")
 
 
if __name__ == "__main__":
    if _is_running_in_streamlit():
        run_streamlit_app()
    else:
        main()
