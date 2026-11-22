# Leakage-free rebuild: SRI standardisation AND 90th-pct threshold calibrated
# on the TRAINING PERIOD ONLY (first 70% of days), then applied to all days.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix)

RENEW=['hydro','solar','wind','biomass']; FOSSIL=['coal','oil']; LF=0.70
df=pd.read_excel("final_cleaned_data.xlsx",sheet_name=0)
df.columns=[c.strip().lower() for c in df.columns]
df['date']=pd.to_datetime(df['date']); df=df.sort_values('date').set_index('date')
cols=['total_net_generation']+RENEW+FOSSIL+['peak_demand']
for c in cols: df[c]=pd.to_numeric(df[c],errors='coerce')
df[cols]=df[cols].interpolate(method='time',limit=3)
df['G_ren']=df[RENEW].sum(axis=1); df['G_total']=df['total_net_generation']
df['D_est']=df['peak_demand']*24*LF/1000
df['RS']=df['G_ren']/df['G_total']; df['RDR']=df['G_ren']/df['D_est']

# ---- CALIBRATE ON TRAINING PERIOD ONLY (first 70% of days) ----
N=len(df); cut=int(N*0.70)
rs_mu,rs_sd=df['RS'].iloc[:cut].mean(),df['RS'].iloc[:cut].std()
rd_mu,rd_sd=df['RDR'].iloc[:cut].mean(),df['RDR'].iloc[:cut].std()
df['SRI']=0.5*((df['RS']-rs_mu)/rs_sd)+0.5*((df['RDR']-rd_mu)/rd_sd)   # train-calibrated z
thr=df['SRI'].iloc[:cut].quantile(0.90)                                # train-calibrated threshold
df['risk_day']=(df['SRI']>thr).astype(int)
print(f"Total days {N} | train days {cut} ({df.index[0].date()}..{df.index[cut-1].date()})")
print(f"Train-calibrated SRI 90th pct threshold = {thr:.3f}")
print(f"Risk days: all={int(df['risk_day'].sum())} ({df['risk_day'].mean()*100:.1f}%) | "
      f"train={int(df['risk_day'].iloc[:cut].sum())} | test-region(after cut)={int(df['risk_day'].iloc[cut:].sum())}")

# ---- FEATURES + next-day target ----
d=df.reset_index()
d['target']=d['risk_day'].shift(-1)
for lag in [1,2,3,7]:
    d[f'SRI_lag{lag}']=d['SRI'].shift(lag); d[f'RS_lag{lag}']=d['RS'].shift(lag); d[f'RDR_lag{lag}']=d['RDR'].shift(lag)
for lag in [1,2,7]: d[f'Gren_lag{lag}']=d['G_ren'].shift(lag)
d['SRI_roll7']=d['SRI'].shift(1).rolling(7).mean(); d['SRI_roll14']=d['SRI'].shift(1).rolling(14).mean()
d['RS_roll7']=d['RS'].shift(1).rolling(7).mean()
d['risk_today']=d['risk_day']
d['month']=d['date'].dt.month; d['day_of_year']=d['date'].dt.dayofyear
d['season']=d['month'].map(lambda m:1 if m in(12,1,2) else 2 if m in(3,4) else 3 if m in(5,6,7,8,9) else 4)
d['month_sin']=np.sin(2*np.pi*d['month']/12); d['month_cos']=np.cos(2*np.pi*d['month']/12)
FEATURES=([f'SRI_lag{l}' for l in[1,2,3,7]]+[f'RS_lag{l}' for l in[1,2,3,7]]+[f'RDR_lag{l}' for l in[1,2,3,7]]+
          [f'Gren_lag{l}' for l in[1,2,7]]+['SRI_roll7','SRI_roll14','RS_roll7','risk_today','month','day_of_year','season','month_sin','month_cos'])
m=d.dropna(subset=FEATURES+['target']).reset_index(drop=True); m['target']=m['target'].astype(int)
X,y=m[FEATURES],m['target']
print(f"Modelled rows {len(m)} | positives {int(y.sum())} ({y.mean()*100:.1f}%)")

def walk_forward(make,scale=False,refit=30,start_frac=0.70):
    start=int(len(m)*start_frac); proba=pd.Series(index=m.index,dtype=float); clf=None; sc=None
    for i in range(start,len(m)):
        if clf is None or (i-start)%refit==0:
            Xtr,ytr=X.iloc[:i],y.iloc[:i]
            if scale: sc=StandardScaler().fit(Xtr); Xtr_=sc.transform(Xtr)
            else: Xtr_=Xtr.values
            clf=make(); clf.fit(Xtr_,ytr)
        xi=X.iloc[[i]].values; xi_=sc.transform(xi) if scale else xi
        proba.iloc[i]=clf.predict_proba(xi_)[0,1]
    out=m.loc[proba.dropna().index,['date','target']].copy(); out['proba']=proba.dropna().values
    return out

# SARIMA benchmark: forecast SRI, threshold from training portion of the SRI series only
sri=m['SRI'] if 'SRI' in m else None
sri=d.set_index('date')['SRI'].reindex(pd.to_datetime(m['date'])).reset_index(drop=True)
start=int(len(m)*0.70); sar_thr=sri.iloc[:start].quantile(0.90)
fc=pd.Series(index=m.index,dtype=float); fit=None; anc=0
for i in range(start,len(m)):
    if fit is None or (i-start)%30==0:
        fit=SARIMAX(sri.iloc[:i],order=(1,0,1),seasonal_order=(1,1,1,7),
                    enforce_stationarity=False,enforce_invertibility=False).fit(disp=False); anc=i
    fc.iloc[i]=fit.forecast(steps=i-anc+1).iloc[-1]
f=fc.dropna(); sar=m.loc[f.index,['date','target']].copy()
sar['proba']=((f-f.min())/(f.max()-f.min())).values; sar['pred']=(f>sar_thr).astype(int).values

lr=walk_forward(lambda:LogisticRegression(class_weight='balanced',max_iter=1000),scale=True)
rf=walk_forward(lambda:RandomForestClassifier(n_estimators=300,class_weight='balanced_subsample',random_state=42,n_jobs=-1))
xg=walk_forward(lambda:XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,scale_pos_weight=9,eval_metric='logloss',random_state=42,n_jobs=-1))

preds={'SARIMA':sar.set_index('date'),'Logistic Regression':lr.set_index('date'),
       'Random Forest':rf.set_index('date'),'XGBoost':xg.set_index('date')}
common=sorted(set.intersection(*[set(p.index) for p in preds.values()]))
yt=preds['XGBoost'].loc[common,'target'].astype(int); prev=yt.mean()
print(f"\nEvaluation window: {len(common)} days | positives {int(yt.sum())} | prevalence {prev*100:.1f}%\n")
hdr=f"{'Model':<22}{'Prec':>6}{'Rec':>6}{'F1':>6}{'PR-AUC':>8}{'ROC-AUC':>9}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>5}"
print(hdr); print('-'*len(hdr))
def row(name,pred,score):
    tn,fp,fn,tp=confusion_matrix(yt,pred).ravel()
    print(f"{name:<22}{precision_score(yt,pred,zero_division=0):>6.3f}{recall_score(yt,pred,zero_division=0):>6.3f}"
          f"{f1_score(yt,pred,zero_division=0):>6.3f}{average_precision_score(yt,score):>8.3f}{roc_auc_score(yt,score):>9.3f}"
          f"{tp:>4}{fp:>4}{fn:>4}{tn:>5}")
for name,p in preds.items():
    d2=p.loc[common]; score=d2['proba']
    pred=d2['pred'].astype(int) if 'pred' in d2 else (score>=0.5).astype(int)
    row(name,pred,score)
# persistence baseline
md=m.set_index('date').loc[common]; bp=md['risk_today'].astype(int)
tn,fp,fn,tp=confusion_matrix(yt,bp).ravel()
print(f"{'Baseline(persist)':<22}{precision_score(yt,bp,zero_division=0):>6.3f}{recall_score(yt,bp,zero_division=0):>6.3f}"
      f"{f1_score(yt,bp,zero_division=0):>6.3f}{prev:>8.3f}{0.5:>9.3f}{tp:>4}{fp:>4}{fn:>4}{tn:>5}")

# ---- extra checks for the supervisor's questions ----
print("\nDates:",df.index[0].date(),"->",df.index[-1].date(),"| eval window:",common[0].date(),"->",common[-1].date())
full=pd.date_range(df.index[0],df.index[-1]); print("missing calendar dates:",len(full)-len(df))
print("corr(RS,RDR) =",round(df['RS'].corr(df['RDR']),3))
print("Persistence scored as 0/1: PR-AUC=%.3f ROC-AUC=%.3f"%(average_precision_score(yt,bp),roc_auc_score(yt,bp)))
xs=preds['XGBoost'].loc[common,'proba'].values; ls=preds['Logistic Regression'].loc[common,'proba'].values; yv=yt.values
rng=np.random.default_rng(42); B=2000; L=14; n=len(yv); res={'xgb':[], 'lr':[], 'per':[]}
for _ in range(B):
    st=rng.integers(0,n-L+1,size=int(np.ceil(n/L))); idx=np.concatenate([np.arange(s,s+L) for s in st])[:n]
    if yv[idx].sum()<3: continue
    res['xgb'].append(average_precision_score(yv[idx],xs[idx])); res['lr'].append(average_precision_score(yv[idx],ls[idx]))
    res['per'].append(average_precision_score(yv[idx],bp.values[idx]))
for k,v in res.items(): print(k,"PR-AUC 95%% block-bootstrap CI: %.2f-%.2f"%tuple(np.percentile(v,[2.5,97.5])))
for nm,p in preds.items(): p.loc[common].to_csv("pred_"+nm.split()[0].lower()+".csv")

# ---- save the results table + feature importance (model fitted at the last refit point) ----
rows=[]
for nm,p in preds.items():
    d2=p.loc[common]; sc_=d2['proba']; pr_=d2['pred'].astype(int) if 'pred' in d2 else (sc_>=0.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(yt,pr_).ravel()
    rows.append([nm,precision_score(yt,pr_,zero_division=0),recall_score(yt,pr_,zero_division=0),f1_score(yt,pr_,zero_division=0),
                 average_precision_score(yt,sc_),roc_auc_score(yt,sc_),tp,fp,fn,tn])
tn,fp,fn,tp=confusion_matrix(yt,bp).ravel()
rows.append(['Persistence',precision_score(yt,bp),recall_score(yt,bp),f1_score(yt,bp),average_precision_score(yt,bp),roc_auc_score(yt,bp),tp,fp,fn,tn])
pd.DataFrame(rows,columns=['Model','Precision','Recall','F1','PR-AUC','ROC-AUC','TP','FP','FN','TN']).round(3).to_csv('results_leakfree.csv',index=False)
st=int(len(m)*0.70); last=st+30*((len(m)-1-st)//30)
rfm=RandomForestClassifier(n_estimators=300,class_weight='balanced_subsample',random_state=42,n_jobs=-1).fit(X.iloc[:last].values,y.iloc[:last])
xgm=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,scale_pos_weight=9,eval_metric='logloss',random_state=42,n_jobs=-1).fit(X.iloc[:last].values,y.iloc[:last])
imp=pd.DataFrame({'RandomForest':rfm.feature_importances_,'XGBoost_gain':xgm.feature_importances_},index=FEATURES)
imp.round(3).to_csv('importance_leakfree.csv')
print("\nTop 5 RF :",imp['RandomForest'].sort_values(ascending=False).head(5).round(2).to_dict())
print("Top 5 XGB:",imp['XGBoost_gain'].sort_values(ascending=False).head(5).round(2).to_dict())
i85=int(len(m)*0.85); print("Fixed 70/15/15 split positives: val",int(y.iloc[st:i85].sum()),"test",int(y.iloc[i85:].sum()))
print("Saved -> results_leakfree.csv, importance_leakfree.csv, pred_*.csv")
