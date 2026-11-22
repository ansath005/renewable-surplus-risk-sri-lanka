# =====================================================================
#  PART 2 of 5  --  DESCRIPTIVE & SEASONAL ANALYSIS   (Objective 1)
#  Reads step1_clean.csv. Run this whole block in ONE cell.
#  Output: part2_patterns.png
# =====================================================================
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # save plots without a display window
import matplotlib.pyplot as plt

df = pd.read_csv("step1_clean.csv", parse_dates=["date"]).set_index("date")

# 1. Summary statistics of the main series.
print("Summary statistics (GWh):")
print(df[["G_total","G_ren","G_fossil","D_est"]].describe().round(2).to_string())

# 2. Sri Lanka monsoon seasons (they drive hydro & wind -> renewable output).
def monsoon(m):
    if m in (12, 1, 2):        return "NE Monsoon (Dec-Feb)"
    if m in (3, 4):            return "1st Inter-monsoon (Mar-Apr)"
    if m in (5, 6, 7, 8, 9):   return "SW Monsoon (May-Sep)"
    return                          "2nd Inter-monsoon (Oct-Nov)"
SEASON_ORDER = ["NE Monsoon (Dec-Feb)", "1st Inter-monsoon (Mar-Apr)",
                "SW Monsoon (May-Sep)", "2nd Inter-monsoon (Oct-Nov)"]
df["season"] = df.index.month.map(monsoon)
df["month"]  = df.index.month

# 3. Mean generation by season + renewable share of the mix.
tbl = df.groupby("season")[["G_ren","G_fossil","G_total"]].mean().reindex(SEASON_ORDER).round(2)
tbl["renew_%"] = (tbl["G_ren"]/tbl["G_total"]*100).round(1)
print("\nMean generation by season (GWh) and renewable share:")
print(tbl.to_string())

# 4. Plots: trend over time + seasonal shape.
monthly = df.groupby("month")[["G_ren","G_fossil"]].mean()
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(df.index, df["G_ren"], lw=0.8, label="Renewable")
ax[0].plot(df.index, df["G_fossil"], lw=0.8, alpha=0.8, label="Fossil")
ax[0].set_title("Daily generation over time"); ax[0].set_ylabel("GWh"); ax[0].legend()
ax[1].plot(monthly.index, monthly["G_ren"], "o-", label="Renewable")
ax[1].plot(monthly.index, monthly["G_fossil"], "s-", label="Fossil")
ax[1].set_title("Seasonal shape (mean by month)"); ax[1].set_xlabel("Month"); ax[1].legend()
plt.tight_layout(); plt.savefig("part2_patterns.png", dpi=110)
print("\nSaved -> part2_patterns.png")
