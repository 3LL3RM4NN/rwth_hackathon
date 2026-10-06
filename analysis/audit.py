"""One-off raw-data audit (NOT part of the pipeline)."""
import glob, pandas as pd, numpy as np
pd.set_option('display.width',200); pd.set_option('display.max_rows',100)
fs=sorted(glob.glob('data/15min/*.csv'))
rows=[]; ap=[]; gaps=[]; ndays_ts=[]
for f in fs:
    d=pd.read_csv(f,sep=';',dtype={'AffectsTimePoint':str})
    ts=pd.to_datetime(d.Timestamp,utc=True)
    hh=d.Household_ID.iloc[0]
    assert (ts.dt.tz is not None)
    rows.append(dict(hh=hh,group=d.Group.iloc[0],n=len(d),nan_target=int(d.kWh_received_Total.isna().sum()),
        sorted=bool(ts.is_monotonic_increasing),offsets=sorted(set(d.Timestamp.str[-6:])),
        first=ts.min(),last=ts.max(),expected=int((ts.max()-ts.min())/pd.Timedelta('15min'))+1,
        ap=sorted(d.AffectsTimePoint.dropna().unique().tolist()),ap_nan=int(d.AffectsTimePoint.isna().sum())))
    full=pd.date_range(ts.min(),ts.max(),freq='15min')
    miss=full.difference(pd.DatetimeIndex(ts))
    gaps.append(pd.DataFrame({'hh':hh,'ts':miss}))
    # AffectsTimePoint transitions
    a=d.AffectsTimePoint.astype(str)
    chg=a.ne(a.shift()); ap.append(pd.DataFrame({'hh':hh,'ts':ts[chg].values,'val':a[chg].values,'group':d.Group.iloc[0]}))
    d['ts']=ts
    if hh in (100105,):
        w=d[(d.ts>='2023-10-28')&(d.ts<'2023-10-31')]
        print('100105 rows per day Oct28-30:',w.groupby(w.ts.dt.date).size().to_dict())
        print(w.ts.dt.strftime('%d %H:%M').iloc[[0,-1]].tolist())
r=pd.DataFrame(rows); G=pd.concat(gaps); A=pd.concat(ap)
print('households',len(r),'groups',r.group.value_counts().to_dict())
print('offsets',r.offsets.astype(str).value_counts().to_dict(),'| all sorted',r['sorted'].all())
print('NaN target total',r.nan_target.sum(),'| rows',r.n.sum(),'| expected',r.expected.sum(),'| missing rows',(r.expected-r.n).sum(), f"({100*(r.expected-r.n).sum()/r.expected.sum():.2f}%)")
print('ap values per hh:',r.ap.astype(str).value_counts().to_dict(),'ap_nan',r.ap_nan.sum())
# 1 DST
G['date']=G.ts.dt.date
print('\n-- rows per UTC day (all households): ', end='')
# DST: households missing anything on 2023-10-29 / 2023-03-26 / 2022 / 2021
for day in ['2023-03-26','2023-10-29','2022-03-27','2022-10-30','2021-03-28','2021-10-31','2020-10-25','2020-03-29']:
    g=G[G.date==pd.Timestamp(day).date()]
    print(day,'households with missing rows that day:',g.hh.nunique(),'| avg missing/hh',round(len(g)/max(g.hh.nunique(),1),1))
# 2 missingness
r['miss']=r.expected-r.n; r['miss_pct']=100*r.miss/r.expected
print('\nper-household miss%:',r.miss_pct.describe().round(2).to_dict())
print('households with 0 missing',(r.miss==0).sum(),'| <1%',(r.miss_pct<1).sum(),'| >10%',(r.miss_pct>10).sum(),'| >25%',(r.miss_pct>25).sum())
print(r.sort_values('miss_pct').tail(5)[['hh','group','first','last','miss','miss_pct']])
print('by group:',r.groupby('group').miss_pct.agg(['count','mean','median']).round(2).to_dict('index'))
# gap-run lengths
G=G.sort_values(['hh','ts']); new=(G.groupby('hh').ts.diff()!=pd.Timedelta('15min'))
G['run']=new.cumsum(); runs=G.groupby('run').agg(hh=('hh','first'),start=('ts','min'),n=('ts','size'))
print('\ngap runs:',len(runs),'| share of missing intervals in runs >=1 day:',round(runs[runs.n>=96].n.sum()/runs.n.sum(),3),'| runs <1h:',(runs.n<4).sum(),'| median run',runs.n.median())
print('run length bins:',pd.cut(runs.n,[0,4,96,96*7,96*30,1e9]).value_counts().sort_index().to_dict())
# by date
bd=G.groupby('date').size()
act=pd.concat([pd.DataFrame({'date':pd.date_range(a,b).date}) for a,b in zip(r['first'].dt.normalize(),r['last'].dt.normalize())]).groupby('date').size()
frac=(bd.reindex(act.index).fillna(0)/(act*96))
print('\ntop dates by missing fraction among active households:');print(frac.sort_values().tail(8).round(3))
fm=frac.copy(); fm.index=pd.to_datetime(fm.index)
print('by year-month (missing %):');print((100*fm.groupby(fm.index.to_period('M')).mean()).round(1).to_string())
print('by month-of-year:',(100*fm.groupby(fm.index.month).mean()).round(1).to_dict())
print('start-of-record gaps? households whose first ts not midnight:',(r['first'].dt.time!=pd.Timestamp('00:00').time()).sum())
print('last date counts:',r['last'].dt.date.value_counts().head(3).to_dict())
# 3 AffectsTimePoint
print('\nAffectsTimePoint transitions per hh:',A.groupby('hh').size().value_counts().sort_index().to_dict())
print(A.groupby(['group','val']).size())
A=A.sort_values(['hh','ts']); seq=A.groupby('hh').val.agg(lambda x:' > '.join(x)); print(seq.value_counts().head(8))
