# =====================================================================
#  PART 5 of 5  --  CHARACTERISE RISK DAYS + MANN-KENDALL TREND (Objective 2)
#  BACKUP VERSION: needs NO extra package (no pymannkendall).
#  Uses only pandas, numpy, scipy, matplotlib (all come with Anaconda).
#  Reads step4_risk.csv. Run this whole block. Output: part5_risk_summary.png
# =====================================================================
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- self-contained Mann-Kendall trend test --------------------------
def mann_kendall(x):
    """Returns (trend, p_value, sen_slope). No external package needed."""
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]; n = len(x)
    s = sum(np.sign(x[j]-x[i]) for i in range(n-1) for j in range(i+1, n))
    _, counts = np.unique(x, return_counts=True)                  # tie correction
    tie = sum(c*(c-1)*(2*c+5) for c in counts)
    var = (n*(n-1)*(2*n+5) - tie) / 18
    z = (s-1)/np.sqrt(var) if s > 0 else (s+1)/np.sqrt(var) if s < 0 else 0.0
    p = 2*(1 - norm.cdf(abs(z)))
    trend = "no trend" if p >= 0.05 else ("increasing" if s > 0 else "decreasing")
    slope = float(np.median([(x[j]-x[i])/(j-i)
                             for i in range(n-1) for j in range(i+1, n)]))
    return trend, p, slope
# ---------------------------------------------------------------------

df = pd.read_csv("step4_risk.csv", parse_dates=["date"]).set_index("date")

def monsoon(m):
    if m in (12, 1, 2):        return "NE Monsoon (Dec-Feb)"
    if m in (3, 4):            return "1st Inter-monsoon (Mar-Apr)"
    if m in (5, 6, 7, 8, 9):   return "SW Monsoon (May-Sep)"
    return                          "2nd Inter-monsoon (Oct-Nov)"
SEASON_ORDER = ["NE Monsoon (Dec-Feb)", "1st Inter-monsoon (Mar-Apr)",
                "SW Monsoon (May-Sep)", "2nd Inter-monsoon (Oct-Nov)"]
df["season"] = df.index.month.map(monsoon)
df["year"]   = df.index.year
risk = df[df["risk_day"] == 1]
sri_thr = df["SRI"].quantile(0.90)

print(f"Frequency : {len(risk)} of {len(df)} days ({len(risk)/len(df)*100:.1f}%)")
gap = risk["SRI"] - sri_thr
print(f"Magnitude : mean {gap.mean():.3f}, max {gap.max():.3f} above the SRI line")

counts = risk.groupby("season").size().reindex(SEASON_ORDER, fill_value=0)
rate   = (counts / df.groupby("season").size().reindex(SEASON_ORDER) * 100).round(1)
print("\nSeasonality (risk-day count | % of that season's days):")
for s in SEASON_ORDER:
    print(f"  {s:28s} {counts[s]:3d}  | {rate[s]:5.1f}%")
print("\nRisk days per year:")
print(risk.groupby("year").size().to_string())

t1 = mann_kendall(df["risk_day"].resample("ME").sum().values)
print(f"\nMann-Kendall (monthly risk-day count): {t1[0]} (p={t1[1]:.4f}, slope={t1[2]:.3f})")
t2 = mann_kendall(df["SRI"].resample("ME").mean().values)
print(f"Mann-Kendall (monthly mean SRI level): {t2[0]} (p={t2[1]:.4f})")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(df.index, df["SRI"], lw=0.6, color="teal")
ax[0].axhline(sri_thr, color="red", ls="--", label="90th pct")
ax[0].scatter(risk.index, risk["SRI"], color="red", s=8, label="Risk day")
ax[0].set_title("Surplus-Risk Index with flagged days"); ax[0].legend()
rate.plot(kind="bar", ax=ax[1], color="darkorange")
ax[1].set_title("Risk-day rate by season (%)"); ax[1].set_ylabel("%")
plt.setp(ax[1].get_xticklabels(), rotation=20, ha="right", fontsize=7)
plt.tight_layout(); plt.savefig("part5_risk_summary.png", dpi=110)
print("\nSaved -> part5_risk_summary.png")
