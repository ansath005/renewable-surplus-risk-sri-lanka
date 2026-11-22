# =====================================================================
#  PART 3 of 5  --  IMBALANCE INDICATORS (RS, RDR, SRI)   (Objective 2)
#  Reads step1_clean.csv. Run this whole block in ONE cell.
#  Output: step3_indicators.csv
# =====================================================================
import pandas as pd

W1, W2 = 0.5, 0.5     # SRI weights (equal, per methodology)

df = pd.read_csv("step1_clean.csv", parse_dates=["date"]).set_index("date")

# 1. Renewable Share: renewable / total generation.
df["RS"]  = df["G_ren"] / df["G_total"]
# 2. Renewable-to-Demand Ratio: renewable / estimated demand.
df["RDR"] = df["G_ren"] / df["D_est"]

# 3. Z-score each (puts them on one common scale so they can be combined).
def z(s): return (s - s.mean()) / s.std()
df["z_RS"], df["z_RDR"] = z(df["RS"]), z(df["RDR"])

# 4. Surplus-Risk Index: weighted average of the two z-scores.
df["SRI"] = W1*df["z_RS"] + W2*df["z_RDR"]

print("Indicator summary:")
print(df[["RS","RDR","SRI"]].describe().round(3).to_string())
df.to_csv("step3_indicators.csv")
print("\nSaved -> step3_indicators.csv")
