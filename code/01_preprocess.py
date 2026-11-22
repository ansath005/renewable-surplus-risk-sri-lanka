# =====================================================================
#  PART 1 of 5  --  DATA LOADING & PREPROCESSING
#  Reads the Excel data, fills blank rows, builds totals & demand.
#  Run this whole block in ONE cell. Output: step1_clean.csv
# =====================================================================
import pandas as pd

# ---- SETTINGS --------------------------------------------------------
DATA_FILE   = "final_cleaned_data.xlsx"   # your Excel file
SHEET       = "Final Data Set"            # the sheet name
LOAD_FACTOR = 0.70                        # load factor for demand estimate (0.65-0.75)

RENEW_COLS  = ["hydro", "solar", "wind", "biomass"]   # renewable sources (GWh)
FOSSIL_COLS = ["coal", "oil"]                          # fossil sources   (GWh)
NUM_COLS    = ["total_net_generation"] + RENEW_COLS + FOSSIL_COLS + ["peak_demand"]
# ----------------------------------------------------------------------

def preprocess():
    # 1. LOAD the Excel sheet fresh every run (no stale-notebook state).
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET)
    df.columns = [str(c).strip().lower() for c in df.columns]  # tidy headers

    # 2. Check the needed columns are present.
    need = ["date"] + NUM_COLS
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}\nFound: {list(df.columns)}")

    # 3. Dates -> sort -> index by date.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
    print(f"Loaded {len(df)} days: {df.index.min().date()} to {df.index.max().date()}")

    # 4. Ensure numeric, then fill short blank gaps (<=3 days) by time interpolation.
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before = int(df[NUM_COLS].isna().sum().sum())
    df = df.interpolate(method="time", limit=3)
    print(f"Blank cells filled: {before - int(df[NUM_COLS].isna().sum().sum())} "
          f"({int(df[NUM_COLS].isna().sum().sum())} left)")

    # 5. Consistency check: do the six sources add up to the reported total?
    df["discrepancy_%"] = ((df[RENEW_COLS+FOSSIL_COLS].sum(axis=1) - df["total_net_generation"])
                           / df["total_net_generation"] * 100)
    print(f"Source-vs-total gap: mean {df['discrepancy_%'].mean():+.2f}%, "
          f"worst |{df['discrepancy_%'].abs().max():.2f}|%")

    # 6. Aggregate renewable & fossil; estimate demand (GWh) = P(MW)*24*LF/1000.
    df["G_ren"]    = df[RENEW_COLS].sum(axis=1)
    df["G_fossil"] = df[FOSSIL_COLS].sum(axis=1)
    df["G_total"]  = df["total_net_generation"]
    df["D_est"]    = df["peak_demand"] * 24 * LOAD_FACTOR / 1000
    print(f"Estimated demand: mean {df['D_est'].mean():.1f} GWh (LF={LOAD_FACTOR})")

    # 7. Save the clean file for the next parts.
    keep = ["G_total","G_ren","G_fossil","D_est"] + RENEW_COLS + FOSSIL_COLS + ["peak_demand"]
    df[keep].to_csv("step1_clean.csv")
    print("Saved -> step1_clean.csv")
    return df

df = preprocess()
