# Final code: Renewable Energy Surplus Risk Analysis (Sri Lanka)

Put every script in ONE folder together with `final_cleaned_data.xlsx`, then run them in the order below.

Install once:
    pip install pandas numpy matplotlib openpyxl scipy pymannkendall scikit-learn statsmodels xgboost

## Objectives 1 and 2: patterns, indicators, risk days (folder objective1_2)

| Order | Script | What it does | Output |
|---|---|---|---|
| 1 | 01_preprocess.py | Reads the Excel, fills 16 blank cells, checks sources vs total, estimates demand (peak MW x 24 x 0.70 / 1000) | step1_clean.csv |
| 2 | 02_descriptive.py | Daily, monthly and seasonal generation patterns (Objective 1) | part2_patterns.png |
| 3 | 03_indicators.py | RS, RDR and SRI = 0.5 z(RS) + 0.5 z(RDR) | step3_indicators.csv |
| 4 | 04_risk_identification.py | Flags potential surplus-risk days (SRI above 90th percentile): 144 days, 10.0% | step4_risk.csv |
| 5 | 05_characterise_trend.py | Frequency, severity, Mann-Kendall trend test (no trend, p = 0.94) | part5_risk_summary.png |
| 5 (backup) | 05_characterise_trend_NO_INSTALL.py | Same as step 5 without the pymannkendall package | part5_risk_summary.png |

Objectives 1-2 are descriptive, so the SRI uses the whole record (threshold 1.374).

## Objective 3: next-day prediction (folder objective3)

| Order | Script | What it does | Output |
|---|---|---|---|
| 1 | objective3_leakfree.py | Main analysis. SRI and threshold (1.460) from the training period only; SARIMA, logistic regression, random forest, XGBoost and persistence on 428 walk-forward days (30 risk days) | results_leakfree.csv, importance_leakfree.csv, pred_*.csv |
| 2 | objective3_checks.py | Block-bootstrap 95% intervals and the re-estimation check asked for by the supervisor (needs objective3_leakfree.py in the same folder) | printed results |
| 3 | build_page5.py | Optional: the six CSV files for Power BI Page 5 (needs objective3_leakfree.py in the same folder) | page5/*.csv |

## Do not use these older scripts

They used the SRI threshold from the full dataset, which leaks future information into the prediction label:
00_prepare_data.py, 06_features_split.py, 07_sarima.py to 11_compare.py, _wf.py, objective3_full.py.
The analysis_parts folder is an earlier draft of Objectives 1-2 and is replaced by the scripts here.
