# =====================================================================
#  PART 4 of 5  --  SURPLUS-RISK DAY IDENTIFICATION   (Objective 2)
#  Reads step3_indicators.csv. Run this whole block in ONE cell.
#  Output: step4_risk.csv
# =====================================================================
import pandas as pd

PERCENTILE = 0.90     # a risk day is in the top 10% (above the 90th percentile)

df = pd.read_csv("step3_indicators.csv", parse_dates=["date"]).set_index("date")

# 1. The 90th-percentile threshold of each indicator.
thr = {c: df[c].quantile(PERCENTILE) for c in ["RS","RDR","SRI"]}
print("90th-percentile thresholds:", ", ".join(f"{k}={v:.3f}" for k,v in thr.items()))

# 2. Flag each indicator above its line.
for c in ["RS","RDR","SRI"]:
    df[f"flag_{c}"] = (df[c] > thr[c]).astype(int)

# 3. Two definitions of a risk day:
#    SRI-only  -> ~10% of days (composite index; primary definition)
#    ANY-of-3  -> above any of the three lines (a bit more than 10%)
df["risk_SRI"] = df["flag_SRI"]
df["risk_ANY"] = df[["flag_RS","flag_RDR","flag_SRI"]].max(axis=1)
for name, col in [("SRI-only","risk_SRI"), ("ANY-of-three","risk_ANY")]:
    n = int(df[col].sum())
    print(f"  {name:14s}: {n} risk days ({n/len(df)*100:.1f}%)")

df["risk_day"]   = df["risk_SRI"]                          # primary definition
df["exceedance"] = (df["SRI"] - thr["SRI"]).clip(lower=0)  # magnitude above line
df.to_csv("step4_risk.csv")
print("\nSaved -> step4_risk.csv")
