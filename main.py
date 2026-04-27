import json, logging, os, warnings
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from entsoe import EntsoePandasClient
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# Config 
TZ             = "Europe/Berlin"
COUNTRY        = "DE_LU"
CCGT_EFF       = 0.50          # thermal efficiency (LHV)
EMIT_FACTOR    = 0.202         # tCO2 / MWh_thermal (natural gas)
PRICE_CAP_HI   = 3_000.0      # EUPHEMIA upper limit €/MWh
PRICE_CAP_LO   = -500.0       # EUPHEMIA lower limit €/MWh
N_CV_SPLITS    = 5
Q_LOW, Q_HIGH  = 0.10, 0.90
OUT            = Path("output"); OUT.mkdir(exist_ok=True)

#  Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(OUT / "pipeline.log", mode="w")],
)
log = logging.getLogger("FVE")



# 1. DATA INGESTION

def _entsoe_client() -> EntsoePandasClient:
    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise EnvironmentError("ENTSOE_API_KEY not set.")
    return EntsoePandasClient(api_key=key)


def fetch_power_data(start_str: str, end_str: str) -> pd.DataFrame:
    """
    Fetches hourly DE_LU DA prices + load + wind/solar from ENTSO-E.
    15-min fundamentals are resampled to 1 h (mean MW).
    TTF and EUA are daily from Yahoo Finance, forward-filled to hourly.
    All timestamps are kept in Europe/Berlin (CET/CEST, DST-aware).
    Duplicate hours from DST fall-back (Oct: 02:00 appears twice) are
    deduplicated in run_qa().
    """
    client = _entsoe_client()
    start  = pd.Timestamp(start_str, tz=TZ)
    end    = pd.Timestamp(end_str,   tz=TZ)
    log.info(f"Fetching ENTSO-E {COUNTRY}  {start_str} → {end_str}")

    # DA prices (resampled to guarantee 1-hour frequency)
    da = pd.DataFrame({"DA_Price": client.query_day_ahead_prices(COUNTRY, start=start, end=end)}).resample("1h").mean()

    # Load + renewables (15-min → 1 h mean)
    load = client.query_load(COUNTRY, start=start, end=end)
    ren  = client.query_wind_and_solar_forecast(COUNTRY, start=start, end=end, psr_type=None)
    raw  = pd.concat([load, ren], axis=1)

    rename = {}
    for c in raw.columns:
        s = str(c)
        if   "Actual Load" in s or ("Load" in s and "Wind" not in s): rename[c] = "Load"
        elif "Solar"    in s: rename[c] = "Solar"
        elif "Offshore" in s: rename[c] = "Wind_Offshore"
        elif "Onshore"  in s: rename[c] = "Wind_Onshore"
    funda = raw.rename(columns=rename).resample("1h").mean()

    df = da.join(funda, how="outer").sort_index()

    # 2025 commodity proxy levels
    _FALLBACKS = {"TTF_Gas_Price": 35.0, "EUA_Price": 60.0}
    _TICKERS   = {
        "TTF_Gas_Price": ["TTF=F"],
        "EUA_Price":     ["EUASIF.IS", "XD9U.DE", "CO2.DE"],
    }
    for col, tickers in _TICKERS.items():
        series = None
        for ticker in tickers:
            try:
                raw_c = yf.download(ticker, start=start_str, end=end_str,
                                    progress=False, auto_adjust=True)[["Close"]]
                if raw_c.empty:
                    raise ValueError("empty dataframe")
                raw_c.columns = [col]
                raw_c.index   = (raw_c.index.tz_localize(TZ)
                                 if raw_c.index.tz is None
                                 else raw_c.index.tz_convert(TZ))
                series = raw_c[col].reindex(df.index, method="ffill").ffill().bfill()
                if series.isna().all():
                    raise ValueError("all-NaN after reindex")
                log.info(f"  {col}: loaded from {ticker}")
                break
            except Exception as exc:
                log.warning(f"  {ticker} failed ({exc}), trying next …")

        if series is None or series.isna().all():
            log.warning(f"  All tickers failed for {col}. Using constant {_FALLBACKS[col]}.")
            series = pd.Series(_FALLBACKS[col], index=df.index)

        df[col] = series

    return df



# 2. DATA QA

def run_qa(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Checks:
      1. Field coverage (% non-null per column)
      2. Duplicate timestamps  — DST fall-back produces 02:00 twice in October
      3. Missing values before/after forward-fill
      4. Price outliers vs EUPHEMIA caps  (−500 / +3000 €/MWh)
      5. DST hour-count sanity (8760 or 8784 h per full year)
    """
    report = {}

    # 1. Coverage
    cov = df.notna().mean()
    report["coverage_pct"] = cov.round(4).to_dict()
    low = cov[cov < 0.80]
    (log.warning if not low.empty else log.info)(
        f"[QA] Low coverage (<80%): {low.to_dict()}" if not low.empty
        else "[QA] Coverage OK (≥80% on all fields)"
    )

    # 2. Duplicate timestamps (DST fall-back — Oct clock change)
    n_dup = int(df.index.duplicated().sum())
    report["duplicate_timestamps"] = n_dup
    if n_dup:
        log.error(f"[QA] {n_dup} duplicate timestamps — keeping first (DST fall-back)")
        df = df[~df.index.duplicated(keep="first")]
    else:
        log.info("[QA] No duplicate timestamps")

    # 3. Missing values
    pre = df.isna().sum()
    report["missing_pre_fill"] = pre.to_dict()
    if pre.sum():
        log.warning(f"[QA] NaNs before fill: {pre[pre > 0].to_dict()}")
        df = df.ffill().bfill()
    log.info(f"[QA] Missing post-fill: {int(df.isna().sum().sum())}")

    # 4. Price outliers — cap to EUPHEMIA market limits
    hi = int((df["DA_Price"] > PRICE_CAP_HI).sum())
    lo = int((df["DA_Price"] < PRICE_CAP_LO).sum())
    report["outliers"] = {"above_cap": hi, "below_cap": lo}
    df.loc[df["DA_Price"] > PRICE_CAP_HI, "DA_Price"] = PRICE_CAP_HI
    df.loc[df["DA_Price"] < PRICE_CAP_LO, "DA_Price"] = PRICE_CAP_LO
    if hi or lo:
        log.warning(f"[QA] Capped: {hi} above +€{PRICE_CAP_HI}, {lo} below €{PRICE_CAP_LO}")
    else:
        log.info("[QA] No price outliers outside EUPHEMIA caps")

    # 5. DST hour-count sanity
    for yr in sorted(df.index.year.unique()):
        n   = int((df.index.year == yr).sum())
        exp = 8784 if yr % 4 == 0 else 8760
        status = "OK" if abs(n - exp) <= 4 else f"WARNING (expected ~{exp})"
        report.setdefault("dst_check", {})[str(yr)] = {"hours": n, "status": status}
        log.info(f"[QA] {yr}: {n} hours ({status})")

    log.info(f"[QA] Done. Shape: {df.shape}")
    return df, report


# 3. FEATURE ENGINEERING

_FEATURES = [
    "Load", "Residual_Load", "Wind_Total", "Solar", "RES_Penetration",
    "TTF_Gas_Price", "EUA_Price", "Gas_Cost", "Carbon_Cost",
    "CSS_Lag_24h", "CSS_Lag_168h",
    "Price_Lag_24h", "Price_Lag_168h", "Price_7d_Mean",
    "Hour", "DayOfWeek", "Month", "IsWeekend",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Targets the Clean Spark Spread (CSS):
        CSS = DA_Price − TTF/η − EUA×εf/η
    All lags are ≥ T−24 h (auction closes D−1 noon; no leakage possible).
    """
    df = df.copy()
    η, εf = CCGT_EFF, EMIT_FACTOR

    wind = (df.get("Wind_Offshore", pd.Series(0, index=df.index)).fillna(0)
          + df.get("Wind_Onshore",  pd.Series(0, index=df.index)).fillna(0))
    df["Wind_Total"]       = wind
    df["Renewable_Total"]  = wind + df["Solar"].fillna(0)
    df["Residual_Load"]    = df["Load"] - df["Renewable_Total"]
    df["RES_Penetration"]  = df["Renewable_Total"] / df["Load"].replace(0, np.nan)

    df["Gas_Cost"]           = df["TTF_Gas_Price"] / η
    df["Carbon_Cost"]        = (df["EUA_Price"] * εf) / η
    df["Clean_Spark_Spread"] = df["DA_Price"] - df["Gas_Cost"] - df["Carbon_Cost"]

    for lag in [24, 48, 168]:
        df[f"Price_Lag_{lag}h"] = df["DA_Price"].shift(lag)
        df[f"CSS_Lag_{lag}h"]   = df["Clean_Spark_Spread"].shift(lag)

    df["Price_7d_Mean"] = df["Price_Lag_24h"].rolling(168).mean()
    df["Hour"]      = df.index.hour
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"]     = df.index.month
    df["IsWeekend"] = (df.index.dayofweek >= 5).astype(int)

    before = len(df)
    df = df.dropna()
    log.info(f"Features: {before} → {len(df)} rows (lag warm-up removed)")
    return df



# 4. MODELS  (Baseline: SeasonalNaive | Main: HistGBM on CSS + Quantiles)

def predict_naive(df: pd.DataFrame) -> np.ndarray:
    """Baseline: same-hour, same-weekday, last week (T−168h). No fitting required."""
    return df["Price_Lag_168h"].values


def fit_gbm(df: pd.DataFrame):
    """Fits mean + q10 + q90 HistGBM on Clean Spark Spread."""
    X, y = df[_FEATURES], df["Clean_Spark_Spread"]
    kw = dict(max_iter=300, learning_rate=0.03, max_depth=5,
              min_samples_leaf=20, l2_regularization=0.1, random_state=42)
    m_mean = HistGradientBoostingRegressor(loss="squared_error", **kw).fit(X, y)
    m_q10  = HistGradientBoostingRegressor(loss="quantile", quantile=Q_LOW,
                                           max_iter=200, learning_rate=0.03,
                                           max_depth=4, random_state=42).fit(X, y)
    m_q90  = HistGradientBoostingRegressor(loss="quantile", quantile=Q_HIGH,
                                           max_iter=200, learning_rate=0.03,
                                           max_depth=4, random_state=42).fit(X, y)
    return m_mean, m_q10, m_q90


def predict_gbm(models, df: pd.DataFrame) -> dict:
    m_mean, m_q10, m_q90 = models
    X    = df[_FEATURES]
    fuel = df["Gas_Cost"].values + df["Carbon_Cost"].values
    return {
        "mean":        m_mean.predict(X) + fuel,
        "q10":         m_q10.predict(X)  + fuel,
        "q90":         m_q90.predict(X)  + fuel,
        "spread_mean": m_mean.predict(X),
    }


def _metrics(actual, pred):
    return {
        "mae":           float(mean_absolute_error(actual, pred)),
        "rmse":          float(np.sqrt(mean_squared_error(actual, pred))),
        "p95_abs_error": float(np.percentile(np.abs(actual - pred), 95)),  # tail metric
    }


def walk_forward_cv(df: pd.DataFrame) -> dict:
    log.info(f"Walk-forward CV ({N_CV_SPLITS} folds) …")
    naive_scores, gbm_scores = [], []

    for fold, (tr_idx, te_idx) in enumerate(TimeSeriesSplit(N_CV_SPLITS).split(df)):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        y = te["DA_Price"].values
        naive_scores.append(_metrics(y, predict_naive(te)))
        gbm_scores.append(_metrics(y, predict_gbm(fit_gbm(tr), te)["mean"]))
        log.info(f"  Fold {fold+1}: Naive MAE €{naive_scores[-1]['mae']:.2f} │ "
                 f"GBM MAE €{gbm_scores[-1]['mae']:.2f}")

    def agg(scores):
        return {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}

    summary = {"SeasonalNaive": agg(naive_scores), "GBM": agg(gbm_scores)}
    for m, s in summary.items():
        log.info(f"  CV {m:20s}: MAE €{s['mae']:.2f} | RMSE €{s['rmse']:.2f} "
                 f"| P95 €{s['p95_abs_error']:.2f}")
    return summary



# 5. PROMPT CURVE TRANSLATION
    """
    Function to convert the hourly forecast data into a prompt-month trading view.

    This function calculates the Risk Premium (RP) by subtracting the model's expected
    value from the traded forward price.
    If the RP is greater than 0, it means the market is rich, so we should output a SHORT signal.
    If the RP is less than 0, it means the market is cheap, so we should output a LONG signal.

    We also need to calculate the z-score of the RP. To do this, we divide the RP by the width 
    of the 80% confidence band (q90 minus q10).
    If the forward price is inside the 80% band, or if the absolute value of the RP z-score 
    is less than 0.5, we will return a NEUTRAL signal instead.

    Note: The signal will become invalid and shouldn't be used if:
    - The actual wind generation is more than 2 GW different from the ENTSO-E forecast (need to rerun).
    - The TTF gas or EUA carbon prices move by more than 5% during the day.
    - The forward price drops inside our model's 80% confidence band.
    - The absolute value of the RP z-score is less than 0.5 (not enough confidence to trade).
    """

def translate_to_curve(preds: dict, df_test: pd.DataFrame,
                        market_fwd: float, df_train: pd.DataFrame) -> dict:

    pm, pq10, pq90 = preds["mean"], preds["q10"], preds["q90"]
    model_mean, model_q10, model_q90 = float(np.mean(pm)), float(np.mean(pq10)), float(np.mean(pq90))

    is_peak      = (df_test.index.hour >= 8) & (df_test.index.hour < 20)
    peak_mean    = float(np.mean(pm[is_peak]))
    offpeak_mean = float(np.mean(pm[~is_peak]))

    rp   = market_fwd - model_mean
    band = max(model_q90 - model_q10, 1.0)
    rp_z = (rp / band) * 1.28

    in_band = model_q10 <= market_fwd <= model_q90
    if in_band or abs(rp_z) < 0.5:
        signal = "NEUTRAL — fwd inside model 80% band or |RP_z| < 0.5"
    elif rp > 0:
        signal = f"SHORT prompt @ €{market_fwd:.2f} | FV €{model_mean:.2f} | RP €{rp:+.2f}"
    else:
        signal = f"LONG prompt @ €{market_fwd:.2f} | FV €{model_mean:.2f} | RP €{rp:+.2f}"

    wow_wind_gw = float(
        (df_test["Wind_Total"].mean() - df_train["Wind_Total"].iloc[-168:].mean()) / 1000
    )

    view = {
        "model_mean": model_mean, "model_q10": model_q10, "model_q90": model_q90,
        "peak_mean": peak_mean,   "offpeak_mean": offpeak_mean,
        "peak_base_spread": peak_mean - offpeak_mean,
        "market_fwd": market_fwd, "risk_premium": rp, "rp_z": rp_z,
        "signal": signal,         "wow_wind_delta_gw": wow_wind_gw,
        "invalidation": [
            "Wind surprise > ±2 GW vs ENTSO-E forecast → re-run model",
            "TTF or EUA intraday move > 5% → recalibrate fuel cost",
            f"Fwd ({market_fwd:.1f}) inside 80% band [{model_q10:.1f}–{model_q90:.1f}] → no trade",
            "|RP_z| < 0.5 → insufficient signal confidence",
        ],
    }
    log.info(f"Curve view → {signal}")
    return view



# 6. AI WORKFLOW  (local Ollama — llama3.2)

_OLLAMA_URL   = "http://localhost:11434/api/generate"
_OLLAMA_MODEL = "llama3.2"


def _ollama(prompt: str, label: str) -> str:
    """Calls local Ollama. Logs prompt + response. Returns text or fallback string."""
    log.debug(f"[AI/{label}] PROMPT:\n{prompt}")
    try:
        r = requests.post(_OLLAMA_URL,
                          json={"model": _OLLAMA_MODEL, "prompt": prompt, "stream": False},
                          timeout=60)
        r.raise_for_status()
        text = r.json()["response"].strip()
        log.info(f"\n{'─'*60}\n[AI/{label}]\n{text}\n{'─'*60}")
        return text
    except Exception as exc:
        log.error(f"[AI/{label}] Ollama call failed: {exc}")
        return f"[AI {label} unavailable: {exc}]"


def ai_desk_commentary(view: dict, cv: dict) -> str:
    """
    LLM generates a 4-sentence morning desk brief from computed metrics ONLY.
    Prompt explicitly forbids invented numbers.
    """
    metrics = {
        "model_fv":          round(view["model_mean"], 2),
        "model_q10":         round(view["model_q10"],  2),
        "model_q90":         round(view["model_q90"],  2),
        "peak_fv":           round(view["peak_mean"],  2),
        "offpeak_fv":        round(view["offpeak_mean"], 2),
        "market_prompt":     round(view["market_fwd"], 2),
        "risk_premium":      round(view["risk_premium"], 2),
        "rp_z":              round(view["rp_z"], 2),
        "signal":            view["signal"],
        "wow_wind_delta_gw": round(view["wow_wind_delta_gw"], 2),
        "gbm_cv_mae":        round(cv["GBM"]["mae"], 2),
        "naive_cv_mae":      round(cv["SeasonalNaive"]["mae"], 2),
    }
    prompt = (
        "You are the lead quant on a European Power trading desk. "
        "Write EXACTLY 4 sentences. Rules:\n"
        "1. Use ONLY the values in the JSON metrics below — do NOT invent numbers.\n"
        "2. Sentence 1: State model fair-value range (q10–q90) vs the traded prompt price.\n"
        "3. Sentence 2: State the LONG/SHORT/NEUTRAL signal and justify with the RP z-score.\n"
        "4. Sentence 3: Discuss the WoW wind delta and its directional price implication.\n"
        "5. Sentence 4: Name the single most important invalidation condition.\n"
        "No headers, no bullet points, no markdown.\n\n"
        f"METRICS:\n{json.dumps(metrics, indent=2)}"
    )
    return _ollama(prompt, "DESK_COMMENTARY")


def ai_qa_rules(schema: dict) -> list:
    """
    LLM proposes 5 validation rules from the column schema.
    Rules are logged; a future step could auto-execute them.
    """
    prompt = (
        "You are a data engineer at an energy trading firm.\n"
        "Given this DataFrame schema (column: dtype), propose exactly 5 "
        "validation rules as a JSON array. Each element must have keys: "
        '"name" (str), "description" (str), "severity" ("ERROR" or "WARNING").\n'
        "Output ONLY the raw JSON array — no markdown, no preamble.\n\n"
        f"Schema:\n{json.dumps(schema, indent=2)}"
    )
    raw = _ollama(prompt, "QA_RULES")
    try:
        rules = json.loads(raw.replace("```json", "").replace("```", "").strip())
        for r in rules:
            log.info(f"  [AI QA {r.get('severity')}] {r.get('name')}: {r.get('description')}")
        return rules
    except Exception as exc:
        log.error(f"Failed to parse AI QA rules JSON: {exc}")
        return []



# 7. VISUALISATION

def plot_forecast(df_test: pd.DataFrame, preds: dict):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    idx = df_test.index

    ax = axes[0]
    ax.fill_between(idx, preds["q10"], preds["q90"], alpha=0.25,
                    color="#1f77b4", label="80% Band")
    ax.plot(idx, preds["mean"],              lw=1.5, color="#1f77b4", label="GBM Forecast")
    ax.plot(idx, df_test["DA_Price"].values, lw=0.9, color="black",   label="Actual")
    ax.set_ylabel("€ / MWh"); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    ax.set_title("DE_LU Day-Ahead Price — Forecast vs Actual", fontweight="bold")

    ax = axes[1]
    ax.plot(idx, df_test["Clean_Spark_Spread"].values, lw=0.9, color="#d62728", label="Actual CSS")
    ax.plot(idx, preds["spread_mean"],                 lw=1.5, color="#1f77b4", ls="--", label="Forecast CSS")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_ylabel("€ / MWh"); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    ax.set_title("Clean Spark Spread (DA − Gas − Carbon)", fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(OUT / "fig1_forecast_vs_actual.png", dpi=150)
    plt.close()


def plot_cv_comparison(cv: dict):
    models = list(cv.keys())
    maes   = [cv[m]["mae"]  for m in models]
    rmses  = [cv[m]["rmse"] for m in models]
    x = np.arange(len(models)); w = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x - w/2, maes,  w, label="MAE",  alpha=0.85,
                  color=["#d62728", "#1f77b4"], edgecolor="k")
    ax.bar(       x + w/2, rmses, w, label="RMSE", alpha=0.55,
                  color=["#d62728", "#1f77b4"], edgecolor="k")
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                f"€{v:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("€ / MWh"); ax.legend(); ax.grid(axis="y", alpha=0.2)
    ax.set_title("Walk-Forward CV — Model Comparison\n(5-fold expanding window)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig2_cv_comparison.png", dpi=150)
    plt.close()



# MAIN

def main():
    log.info("═" * 65)
    log.info("European Power Fair Value Engine  │  DE_LU")
    log.info("═" * 65)

    # 1. Ingest & QA — training window Jan–Nov 2025
    raw_train          = fetch_power_data("2025-01-01", "2025-11-30")
    clean_train, qa_report = run_qa(raw_train)
    (OUT / "qa_report.json").write_text(json.dumps(qa_report, indent=2))

    # AI-proposed additional QA rules (logged for auditability)
    ai_qa_rules({c: str(d) for c, d in clean_train.dtypes.items()})

    # 2. Features
    feat_train = build_features(clean_train)

    # 3. Walk-forward CV  (SeasonalNaive vs GBM)
    cv_summary = walk_forward_cv(feat_train)

    # 4. Train final GBM on full training data
    log.info("Training final GBM on full training set …")
    final_models = fit_gbm(feat_train)

    # 5. OOS test window: 1–14 Dec 2025
    #    Fetch from 24 Nov so T-168h lags are valid from Dec-01 onwards
    raw_test          = fetch_power_data("2025-11-24", "2025-12-14")
    clean_test, _     = run_qa(raw_test)
    feat_test         = build_features(clean_test)
    test_window       = feat_test.loc["2025-12-01":"2025-12-14"]

    preds  = predict_gbm(final_models, test_window)
    actual = test_window["DA_Price"].values
    oos    = _metrics(actual, preds["mean"])
    log.info(f"OOS  MAE €{oos['mae']:.2f} │ RMSE €{oos['rmse']:.2f} │ P95 €{oos['p95_abs_error']:.2f}")

    # 6. Curve translation (Simulated forward price lowered to 85.0 for 2025 levels)
    view = translate_to_curve(preds, test_window, market_fwd=85.0, df_train=feat_train)
    (OUT / "curve_view.json").write_text(
        json.dumps({k: v for k, v in view.items() if k != "invalidation"}, indent=2)
    )

    # 7. AI desk commentary (Ollama / llama3.2)
    commentary = ai_desk_commentary(view, cv_summary)
    (OUT / "desk_commentary.txt").write_text(commentary + "\n")

    # 8. Figures
    plot_forecast(test_window, preds)
    plot_cv_comparison(cv_summary)
    log.info("Figures saved.")

    # 9. submission.csv  (id, y_pred) — test window Dec 1–14 2025
    pd.DataFrame({
        "id":     test_window.index.strftime("%Y-%m-%d %H:00"),
        "y_pred": np.round(preds["mean"], 2),
    }).to_csv(OUT / "submission.csv", index=False)
    log.info(f"submission.csv saved ({len(test_window)} rows).")

    log.info("═" * 65)
    log.info(f"Signal  : {view['signal']}")
    log.info(f"OOS MAE : €{oos['mae']:.2f}/MWh")
    log.info("═" * 65)


if __name__ == "__main__":
    main()