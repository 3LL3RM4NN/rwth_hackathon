"""One-off raw-data audit part 2 (NOT part of the pipeline)."""
import glob, pandas as pd, numpy as np
pd.set_option('display.width',200); pd.set_option('display.max_rows',60)
fs=sorted(glob.glob('data/15min/*.csv'))
D=[]
for f in fs:
    d=pd.read_csv(f,sep=';',dtype={'AffectsTimePoint':str}); d['ts']=pd.to_datetime(d.Timestamp,utc=True).dt.tz_localize(None); D.append(d)
d=pd.concat(D,ignore_index=True); del D
d['date']=d.ts.dt.normalize()
# --- DST days: rows per hh-day around transitions
for day in ['2019-10-27','2020-10-25','2021-10-31','2022-10-30','2023-10-29','2023-03-26']:
    t=pd.Timestamp(day); x=d[d.date.between(t-pd.Timedelta(days=1),t+pd.Timedelta(days=1))]
    pres=x.groupby('date').Household_ID.nunique()
    # active households = those with data both before and after
    rows=x.groupby(['date','Household_ID']).size().unstack(0).reindex(columns=pd.date_range(t-pd.Timedelta(days=1),periods=3)).fillna(0)
    act=rows[(rows.iloc[:,0]>0)&(rows.iloc[:,2]>0)]
    print(day,'active hh',len(act),'| rows/day dist (day-1, day, day+1):',[int(act.iloc[:,i].median()) for i in range(3)],'| hh with 0 rows on day:',int((act.iloc[:,1]==0).sum()),'| hh with !=96 rows on day>0:',int(((act.iloc[:,1]>0)&(act.iloc[:,1]!=96)).sum()),'| max rows',int(act.iloc[:,1].max()))
print('rows per hh-day overall:',d.groupby(['Household_ID','date']).size().value_counts().head(5).to_dict())
# --- NaN target rows
n=d.kWh_received_Total.isna()
print('\nNaN target rows',int(n.sum()),f'({100*n.mean():.2f}% of rows); households with any NaN:',d[n].Household_ID.nunique())
per=d.groupby('Household_ID').apply(lambda g:g.kWh_received_Total.isna().sum()); print('NaN rows per hh top:',per.sort_values().tail(6).to_dict(),'| hh>0 median',per[per>0].median())
dd=d[n].groupby(['Household_ID','date']).size(); print('NaN rows per hh-day: share of full 96-days',round((dd==96).mean(),3),'| partial',int((dd<96).sum()))
# is NaN at start/end of file (leading NaN)?
lead=0
for h,g in d.groupby('Household_ID'):
    v=g.kWh_received_Total.to_numpy(); first=np.argmax(~np.isnan(v)) if (~np.isnan(v)).any() else len(v)
    lead+=first
print('NaN rows that are leading (before first real value):',lead,'of',int(n.sum()))
print('heatpump NaN where total present:',int((d.kWh_received_HeatPump.isna()&~n).sum()),'of',int((~n).sum()))
print('NaN-share by group:',d.groupby('Group').kWh_received_Total.apply(lambda s:s.isna().mean()).round(3).to_dict())
print('NaN by year:',(100*n.groupby(d.ts.dt.year).mean()).round(2).to_dict())
# --- AffectsTimePoint
print('\nAffectsTimePoint dtype (raw):',d.AffectsTimePoint.dtype,'| uniques:',d.AffectsTimePoint.unique().tolist())
t=d[d.Group=='treatment'].copy()
sp=t.groupby(['Household_ID','AffectsTimePoint']).ts.agg(['min','max','size']).reset_index()
w=sp.pivot(index='Household_ID',columns='AffectsTimePoint',values=['min','max','size'])
w=w.dropna(subset=[('min','during visit')])
print('hh with 3 phases:',len(w))
print('during visit: duration days',((w[('max','during visit')]-w[('min','during visit')]).dt.days).describe().round(1).to_dict())
print('during visit: rows/hh',w[('size','during visit')].describe().round(0).to_dict())
print('before max < during min ?',(w[('max','before visit')]<=w[('min','during visit')]).all(),'| during max < after min ?',(w[('max','during visit')]<=w[('min','after visit')]).all())
print('visit (during-start) dates range:',w[('min','during visit')].min().date(),w[('min','during visit')].max().date())
# monotone check: label changes only 2x in time order
bad=0
for h,g in t.groupby('Household_ID'):
    a=g.sort_values('ts').AffectsTimePoint; bad+=(a!=a.shift()).sum()-1>2
print('hh with >2 changes:',bad)
# test-period relevance
test0=pd.Timestamp('2023-03-16'); tt=d[d.ts>=test0]
print('test-period label share (treatment):',tt[tt.Group=='treatment'].AffectsTimePoint.value_counts(normalize=True).round(3).to_dict())
print('test-period label share (all rows):',tt.AffectsTimePoint.value_counts(normalize=True).round(3).to_dict())
# does the label coincide with consumption shift? mean daily kWh by phase (hh-normalised)
x=t.dropna(subset=['kWh_received_Total']).groupby(['Household_ID','AffectsTimePoint']).kWh_received_Total.mean().unstack()
x=x.dropna(subset=['before visit','after visit'])
print('hh mean kWh/15min before vs after (n=%d): %.4f vs %.4f ; median ratio after/before %.3f'%(len(x),x['before visit'].mean(),x['after visit'].mean(),(x['after visit']/x['before visit']).median()))
# DST spring/other missing: visit-related? households missing share by group, 'unknown' vs
print('visit-days with NaN/missing: n/a')
