# =====================================================================
#  PART 5 of 5  --  CHARACTERISE RISK DAYS + MANN-KENDALL TREND  (Objective 2)
#  Reads step4_risk.csv. Needs: pip install pymannkendall
#  Run this whole block in ONE cell. Output: part5_risk_summary.png
# =====================================================================
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymannkendall as mk

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

# 1. Frequency.
print(f"Frequency : {len(risk)} of {len(df)} days ({len(risk)/len(df)*100:.1f}%)")

# 2. Magnitude (how far above the SRI line, on risk days).
gap = risk["SRI"] - sri_thr
print(f"Magnitude : mean {gap.mean():.3f}, max {gap.max():.3f} above the SRI line")

# 3. Seasonality.
counts = risk.groupby("season").size().reindex(SEASON_ORDER, fill_value=0)
rate   = (counts / df.groupby("season").size().reindex(SEASON_ORDER) * 100).round(1)
print("\nSeasonality (risk-day count | % of that season's days):")
for s in SEASON_ORDER:
    print(f"  {s:28s} {counts[s]:3d}  | {rate[s]:5.1f}%")
print("\nRisk days per year:")
print(risk.groupby("year").size().to_string())

# 4. Mann-Kendall trend test (non-parametric: no normality assumption).
r1 = mk.original_test(df["risk_day"].resample("ME").sum().values)
print(f"\nMann-Kendall (monthly risk-day count): {r1.trend} "
      f"(p={r1.p:.4f}, Sen slope={r1.slope:.3f} days/month)")
r2 = mk.original_test(df["SRI"].resample("ME").mean().values)
print(f"Mann-Kendall (monthly mean SRI level): {r2.trend} (p={r2.p:.4f})")

# 5. Plots.
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
