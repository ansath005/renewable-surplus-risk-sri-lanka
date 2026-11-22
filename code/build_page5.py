# Builds the Power BI Page 5 tables from the leakage-free Objective 3 pipeline.
import warnings; warnings.filterwarnings("ignore")
exec(open("objective3_leakfree.py").read().split("# ---- extra checks for the supervisor's questions ----")[0])
import os; os.makedirs("page5", exist_ok=True)

names={'SARIMA':'SARIMA','Logistic Regression':'Logistic Regression','Random Forest':'Random Forest','XGBoost':'XGBoost'}
order={'Persistence':1,'SARIMA':2,'Random Forest':3,'Logistic Regression':4,'XGBoost':5}
sri_d=d.set_index('date')['SRI']; dates=pd.DatetimeIndex(common)
# target on date t = risk on t+1 -> the day being predicted is t+1
pred_day=pd.Series(m['date'].values, index=m['date'].values)

rows=[]; cm=[]
def add(model, score, pred):
    for dt,s,p,a in zip(dates, score, pred, yt.values):
        o='TP' if (p==1 and a==1) else 'FP' if (p==1 and a==0) else 'FN' if (p==0 and a==1) else 'TN'
        rows.append([dt.date(), (dt+pd.Timedelta(days=1)).date(), model, int(a), float(s), int(p), o])
for nm,p in preds.items():
    d2=p.loc[common]; s=d2['proba'].values
    pr=d2['pred'].astype(int).values if 'pred' in d2 else (s>=0.5).astype(int)
    add(nm, s, pr)
add('Persistence', bp.values.astype(float), bp.values.astype(int))
L=pd.DataFrame(rows, columns=['forecast_date','target_date','model','actual_risk','score','predicted_risk','outcome'])
L['model_order']=L['model'].map(order)
L.to_csv('page5/model_predictions_long.csv', index=False)

# metrics + verified bootstrap intervals (from objective3_checks.py, 14-day blocks, 2000 resamples)
ci={'SARIMA':(0.03,0.64,0.00,0.62,0.00,0.69),'Logistic Regression':(0.29,0.87,0.30,0.79,0.64,1.00),
    'Random Forest':(0.23,0.76,0.00,0.68,0.00,0.62),'XGBoost':(0.28,0.85,0.32,0.74,0.40,0.81),
    'Persistence':(0.11,0.71,0.30,0.83,0.30,0.85)}
met=[]
for mdl,g in L.groupby('model'):
    y_=g.actual_risk; p_=g.predicted_risk; s_=g.score
    tn,fp,fn,tp=confusion_matrix(y_,p_).ravel()
    met.append([mdl,order[mdl],round(precision_score(y_,p_,zero_division=0),3),round(recall_score(y_,p_,zero_division=0),3),
                round(f1_score(y_,p_,zero_division=0),3),round(average_precision_score(y_,s_),3),round(roc_auc_score(y_,s_),3),
                tp,fp,fn,tn,*ci[mdl]])
    for a_lab,p_lab,c,cell in [('Risk','Risk',tp,'TP'),('Normal','Risk',fp,'FP'),('Risk','Normal',fn,'FN'),('Normal','Normal',tn,'TN')]:
        cm.append([mdl,order[mdl],a_lab,p_lab,cell,int(c)])
M=pd.DataFrame(met,columns=['model','model_order','precision','recall','f1','pr_auc','roc_auc','tp','fp','fn','tn',
                            'pr_auc_low','pr_auc_high','f1_low','f1_high','recall_low','recall_high']).sort_values('model_order')
M.to_csv('page5/model_metrics.csv', index=False)
pd.DataFrame(cm,columns=['model','model_order','actual','predicted','cell','count']).to_csv('page5/confusion_matrix_long.csv', index=False)

# SRI line for the evaluation period (train-calibrated SRI, actual risk days)
S=pd.DataFrame({'date':dates.date,'sri':sri_d.reindex(dates).round(4).values,'sri_threshold':round(thr,3),
                'risk_day':d.set_index('date')['risk_day'].reindex(dates).values})
S.to_csv('page5/sri_test_period.csv', index=False)

# feature importance (last refit point; RF impurity, XGBoost gain)
st=int(len(m)*0.70); last=st+30*((len(m)-1-st)//30)
rfm=RandomForestClassifier(n_estimators=300,class_weight='balanced_subsample',random_state=42,n_jobs=-1).fit(X.iloc[:last].values,y.iloc[:last])
xgm=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,scale_pos_weight=9,eval_metric='logloss',random_state=42,n_jobs=-1).fit(X.iloc[:last].values,y.iloc[:last])
grp=lambda f:'Calendar' if f in('month','day_of_year','season','month_sin','month_cos') else 'Rolling mean' if 'roll' in f else 'Today risk flag' if f=='risk_today' else 'Lag'
fi=[]
for mdl,imp in [('Random Forest',rfm.feature_importances_),('XGBoost',xgm.feature_importances_)]:
    s=pd.Series(imp,index=FEATURES).sort_values(ascending=False)
    for r,(f,v) in enumerate(s.items(),1): fi.append([mdl,f,grp(f),round(float(v),4),r])
pd.DataFrame(fi,columns=['model','feature','feature_group','importance','rank']).to_csv('page5/feature_importance.csv', index=False)

ref=[['Training period (label calibration)','1 Aug 2022 to 3 May 2025'],['Train-calibrated SRI 90th percentile',f'{thr:.3f}'],
     ['Evaluation period (forecast dates)',f'{common[0].date()} to {common[-1].date()}'],['Out-of-sample days',len(common)],
     ['Potential surplus-risk days in evaluation',int(yt.sum())],['Prevalence (no-skill PR-AUC)',f'{yt.mean():.3f}'],
     ['Validation','Expanding-window walk-forward, refit every 30 days'],['Classifier cut-off','0.5 probability (not tuned)'],
     ['XGBoost minus Persistence PR-AUC','0.19 (95% interval 0.04 to 0.33)'],
     ['Bootstrap','Moving-block, 14-day blocks, 2000 resamples']]
pd.DataFrame(ref,columns=['item','value']).to_csv('page5/model_reference.csv', index=False)
print(M[['model','precision','recall','f1','pr_auc','roc_auc','tp','fp','fn','tn']].to_string(index=False))
print("rows long:",len(L),"| eval:",common[0].date(),"->",common[-1].date(),"| positives",int(yt.sum()),"| thr",round(thr,3))
