"""One-off raw-data audit part 3 (NOT part of the pipeline)."""
import glob, pandas as pd, numpy as np
rows=[];dst=[]
DST=pd.to_datetime(['2019-10-27','2022-10-30','2023-10-29'])
for f in sorted(glob.glob('data/15min/*.csv')):
    d=pd.read_csv(f,sep=';',dtype={'AffectsTimePoint':str})
    ts=pd.to_datetime(d.Timestamp,utc=True).dt.tz_localize(None)
    full=pd.date_range(ts.min(),ts.max(),freq='15min'); m=full.difference(pd.DatetimeIndex(ts))
    nd=int(m.normalize().isin(DST).sum())
    vis=ts[d.AffectsTimePoint=='during visit']
    rows.append(dict(hh=d.Household_ID.iloc[0],exp=len(full),miss=len(m),miss_dst=nd,visit=vis.min().normalize() if len(vis) else pd.NaT,
        in_test=bool(((ts>='2023-03-16')).any()),first=ts.min()))
r=pd.DataFrame(rows).sort_values('miss',ascending=False)
tot=r.miss.sum(); print('missing rows',tot,'| DST-day rows',r.miss_dst.sum(),f'({100*r.miss_dst.sum()/tot:.1f}%)')
for k in (10,27,42): print(f'top {k} hh share of missing rows: {100*r.miss.head(k).sum()/tot:.1f}%')
rest=r.iloc[42:]; print('excluding top-42: miss % =',round(100*(rest.miss-rest.miss_dst).sum()/rest.exp.sum(),2),'incl DST',round(100*rest.miss.sum()/rest.exp.sum(),2))
print('hh with visit in test period (>=2023-03-16):',int((r.visit>='2023-03-16').sum()),'| visit in 2023-2024:',int((r.visit>='2023-01-01').sum()))
print('visit year dist:',r.visit.dt.year.value_counts().sort_index().to_dict())
