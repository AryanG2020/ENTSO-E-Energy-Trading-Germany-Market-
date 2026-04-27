# European Power Fair Value Engine — DE_LU

Forecasts DE_LU Day-Ahead prices from ENTSO-E fundamentals,
translates the forecast into a prompt curve trading view,
and generates an AI desk commentary via a local Ollama LLM.


## -------------------------------------Installation & Setup---------------------------
1. Install Local AI Engine (Ollama)

## Install via Terminal:
For macOS (Homebrew): *** brew install ollama ***I
For Linux: *** curl -fsSL https://ollama.com/install.sh | sh ***I
For Windows: Download the installer from ollama.com.

2. Open a separate terminal and run: *** ollama serve ***I 


## We use a virtual environment to isolate dependencies and prevent library pathing errors.

3. Create and activate Python environment: 
- *** python3 -m venv venv ***
- *** source venv/bin/activate  # Windows: .\venv\Scripts\activate ***I

4. Install requirements: *** pip install -r requirements.txt ***

5. Configure API Key (Get from https://transparency.entsoe.eu) and run: 
*** export ENTSOE_API_KEY="your_api_key_here" ***

6. Execute Pipeline:*** python main.py ***

## -------------------------------------------- Data Sources----------------------------

| Data | Source | Endpoint |


| DE_LU DA prices (hourly) | ENTSO-E | `query_day_ahead_prices` |
| Actual load (15-min → 1h) | ENTSO-E | `query_load` |
| Wind & Solar forecast (15-min → 1h) | ENTSO-E | `query_wind_and_solar_forecast` |
| TTF gas price (daily) | Yahoo Finance | `TTF=F` |
| EUA carbon price (daily) | Yahoo Finance | `CO2.DE` |

ENTSO-E API docs: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html

## ---------------------------------------------Outputs (`output/`)---------------------

| File | Description |

| `submission.csv` | OOS predictions Dec 1–14 2025: `id`, `y_pred` |
| `qa_report.json` | Coverage, duplicates, outliers, DST check |
| `curve_view.json` | Model fair value, risk premium, trading signal |
| `desk_commentary.txt` | AI-generated morning desk brief |
| `fig1_forecast_vs_actual.png` | Forecast vs actual + CSS |
| `fig2_cv_comparison.png` | SeasonalNaive vs GBM CV metrics |
| `pipeline.log` | Full run log with all AI prompts and responses |



## ------------------------------------------------------------------------------------------ ##


## Model Choice: Option A (Hourly DA → Curve average)

Target: **Clean Spark Spread** (CSS = DA_Price − TTF/η − EUA×εf/η)

Forecasting CSS instead of raw price removes the co-integrated fuel/carbon signal
from the residual, making the ML problem structurally cleaner. DA_Price is
reconstructed as `ŷ_price = ŷ_CSS + Gas_Cost + Carbon_Cost`.

## Validation

5-fold expanding-window walk-forward CV.
Metrics: MAE, RMSE, P95 absolute error (tail metric).

## Timezone / DST handling

All timestamps are kept in `Europe/Berlin` throughout.
- ENTSO-E data is returned already localised by `entsoe-py`.
- Yahoo Finance daily closes are `.tz_localize("Europe/Berlin")` before merge.
- DST fall-back (Oct): duplicate 02:00 timestamps are detected and deduplicated in QA.
- DST sanity check: row count per year verified against 8760/8784 expected hours.

## AI Component (Programmatic)
- **Automated Commentary:** Uses `llama3.2` to generate a 4-sentence desk brief based on computed metrics.
- **Dynamic QA:** Proposes and executes validation rules based on the dataframe schema.
- **Auditability:** Full prompts and raw JSON responses are logged in `pipeline.log` to monitor hallucinations/failures.