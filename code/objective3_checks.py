# Supervisor round 2: (A) uncertainty intervals, (B) sensitivity: SRI parameters + threshold
# re-estimated at EVERY refit from data available at that forecast origin.
import warnings; warnings.filterwarnings("ignore")
exec(open("objective3_leakfree.py").read().split("# ---- extra checks for the supervisor's questions ----")[0])   # reuse main pipeline objects

# ---------- (A) moving-block bootstrap (14-day blocks, 2000 resamples) ----------
yv=yt.values; n=len(yv); L=14; B=2000; rng=np.random.default_rng(42)
S={k:preds[k].loc[common,'proba'].values for k in preds}
P={k:(preds[k].loc[common,'pred'].values if 'pred' in preds[k] else (S[k]>=0.5).astype(int)) for k in preds}
S['Persistence']=bp.values.astype(float); P['Persistence']=bp.values
out={k:{'pr':[],'f1':[],'rec':[]} for k in S}; diff=[]
for _ in range(B):
    st=rng.integers(0,n-L+1,size=int(np.ceil(n/L))); idx=np.concatenate([np.arange(s,s+L) for s in st])[:n]
    if yv[idx].sum()<3: continue
    for k in S:
        out[k]['pr'].append(average_precision_score(yv[idx],S[k][idx]))
        out[k]['f1'].append(f1_score(yv[idx],P[k][idx],zero_division=0)); out[k]['rec'].append(recall_score(yv[idx],P[k][idx],zero_division=0))
    diff.append(out['XGBoost']['pr'][-1]-out['Persistence']['pr'][-1])
print("\n(A) 95% block-bootstrap intervals (pooled 428 days)")
for k in S:
    q=lambda v:"%.2f-%.2f"%tuple(np.percentile(v,[2.5,97.5]))
    print(f"  {k:<20} PR-AUC {q(out[k]['pr'])} | F1 {q(out[k]['f1'])} | Recall {q(out[k]['rec'])}")
print("  XGBoost minus Persistence PR-AUC: %.2f (95%% %.2f to %.2f)"%(np.mean(diff),*np.percentile(diff,[2.5,97.5])))

# ---------- (B) recalibrate SRI mean/sd + threshold at every refit ----------
base=df.reset_index()[['date','G_ren','RS','RDR']].copy(); OFF=len(base)-1-len(m)   # rows dropped at the start (lags)
def build(k):                      # k = number of days known at the forecast origin
    b=base.copy()
    b['SRI']=0.5*(b.RS-b.RS[:k].mean())/b.RS[:k].std()+0.5*(b.RDR-b.RDR[:k].mean())/b.RDR[:k].std()
    th=b.SRI[:k].quantile(0.90); b['risk_day']=(b.SRI>th).astype(int); b['target']=b['risk_day'].shift(-1)
    for lag in [1,2,3,7]:
        b[f'SRI_lag{lag}']=b.SRI.shift(lag); b[f'RS_lag{lag}']=b.RS.shift(lag); b[f'RDR_lag{lag}']=b.RDR.shift(lag)
    for lag in [1,2,7]: b[f'Gren_lag{lag}']=b.G_ren.shift(lag)
    b['SRI_roll7']=b.SRI.shift(1).rolling(7).mean(); b['SRI_roll14']=b.SRI.shift(1).rolling(14).mean(); b['RS_roll7']=b.RS.shift(1).rolling(7).mean()
    b['risk_today']=b.risk_day; b['month']=b.date.dt.month; b['day_of_year']=b.date.dt.dayofyear
    b['season']=b.month.map(lambda q:1 if q in(12,1,2) else 2 if q in(3,4) else 3 if q in(5,6,7,8,9) else 4)
    b['month_sin']=np.sin(2*np.pi*b.month/12); b['month_cos']=np.cos(2*np.pi*b.month/12)
    return b.iloc[OFF:-1].reset_index(drop=True), th
mk={'Logistic Regression':(lambda:LogisticRegression(class_weight='balanced',max_iter=1000),True),
    'Random Forest':(lambda:RandomForestClassifier(n_estimators=300,class_weight='balanced_subsample',random_state=42,n_jobs=-1),False),
    'XGBoost':(lambda:XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,scale_pos_weight=9,eval_metric='logloss',random_state=42,n_jobs=-1),False)}
start=int(len(m)*0.70); rec=[]
for i0 in range(start,len(m),30):
    mb,th=build(i0+OFF+1); i1=min(i0+30,len(mb))
    Xtr,ytr=mb[FEATURES].iloc[:i0],mb['target'].iloc[:i0].astype(int); Xte=mb[FEATURES].iloc[i0:i1]
    blk=pd.DataFrame({'y':mb['target'].iloc[i0:i1].astype(int).values,'Persistence':mb['risk_today'].iloc[i0:i1].values})
    for k,(f,scale) in mk.items():
        if scale: sc=StandardScaler().fit(Xtr); c=f().fit(sc.transform(Xtr),ytr); blk[k]=c.predict_proba(sc.transform(Xte))[:,1]
        else: c=f().fit(Xtr.values,ytr); blk[k]=c.predict_proba(Xte.values)[:,1]
    sf=SARIMAX(mb['SRI'].iloc[:i0],order=(1,0,1),seasonal_order=(1,1,1,7),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
    fcv=sf.forecast(steps=i1-i0).values; blk['SARIMA']=fcv-th; rec.append(blk)
R=pd.concat(rec,ignore_index=True); yy=R.y
print(f"\n(B) Recalibrated at every refit: {len(R)} days | positives {int(yy.sum())} ({yy.mean()*100:.1f}%)")
for k in ['SARIMA','Logistic Regression','Random Forest','XGBoost','Persistence']:
    s=R[k]; p=(s>0).astype(int) if k=='SARIMA' else (s>=0.5).astype(int)
    print(f"  {k:<20} Prec {precision_score(yy,p,zero_division=0):.3f} Rec {recall_score(yy,p,zero_division=0):.3f} F1 {f1_score(yy,p,zero_division=0):.3f} PR-AUC {average_precision_score(yy,s):.3f} ROC-AUC {roc_auc_score(yy,s):.3f}")
