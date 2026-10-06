"""One-off audit of static / auxiliary columns (NOT part of the pipeline)."""
import glob, pandas as pd, numpy as np
pd.set_option('display.width',200)
H=pd.read_csv('data/smart_meter_meta_data/households.csv',sep=';'); M=pd.read_csv('data/smart_meter_meta_data/meta_data.csv',sep=';')
print(H.shape,M.shape); print(H.isna().sum().to_dict())
print('PV flag:',H.Installation_HasPVSystem.value_counts(dropna=False).to_dict(),'| Group:',H.Group.value_counts().to_dict(),'| Weather_ID:',H.Weather_ID.value_counts().to_dict())
print('households without survey row:',(~H.Household_ID.isin(M.Household_ID)).sum(),'| survey rows not in households:',(~M.Household_ID.isin(H.Household_ID)).sum(),'| dup survey',M.Household_ID.duplicated().sum())
print(M.drop(columns='Household_ID').isna().mean().round(2).to_dict())
for c in M.columns[1:]:
    if c not in('Survey_Building_LivingArea','Survey_Building_Residents'): print(c,M[c].value_counts(dropna=False).to_dict())
print(M[['Survey_Building_LivingArea','Survey_Building_Residents']].describe().round(1).to_dict())
print('Group x PV:\n',pd.crosstab(H.Group,H.Installation_HasPVSystem.fillna('NA')))
print('Group x Weather:\n',pd.crosstab(H.Group,H.Weather_ID))
print('Group x hasSurvey:\n',pd.crosstab(H.Group,H.Household_ID.isin(M.Household_ID)))
print('Protocols_Available x Group:\n',pd.crosstab(H.Group,H.Protocols_Available))
# HeatPump / Other availability
rows=[]
for f in sorted(glob.glob('data/15min/*.csv')):
    d=pd.read_csv(f,sep=';',usecols=['Household_ID','Timestamp','kWh_received_Total','kWh_received_HeatPump','kWh_received_Other'])
    t=pd.to_datetime(d.Timestamp,utc=True).dt.tz_localize(None)
    ok=d.kWh_received_Total.notna()
    hp=d.kWh_received_HeatPump.notna(); ot=d.kWh_received_Other.notna()
    rows.append(dict(hh=d.Household_ID.iloc[0],n=int(ok.sum()),hp=int((hp&ok).sum()),ot=int((ot&ok).sum()),
        hp_dev=int((hp&ok&(t<'2023-03-16')).sum()),hp_test=int((hp&ok&(t>='2023-03-16')).sum()),
        hp_first=t[hp].min() if hp.any() else pd.NaT, hp_last=t[hp].max() if hp.any() else pd.NaT,
        resid=float((d.kWh_received_Total-d.kWh_received_HeatPump-d.kWh_received_Other)[hp&ot].abs().max()) if (hp&ot).any() else np.nan))
r=pd.DataFrame(rows)
print('\nHeatPump non-NaN share of valid-total rows:',round(r.hp.sum()/r.n.sum(),4),'| Other:',round(r.ot.sum()/r.n.sum(),4))
print('households with any HeatPump:',(r.hp>0).sum(),'| >50% coverage:',((r.hp/r.n)>0.5).sum(),'| in dev window:',(r.hp_dev>0).sum(),'| in test:',(r.hp_test>0).sum())
print('hp rows dev',r.hp_dev.sum(),'test',r.hp_test.sum()); print(r[r.hp>0][['hp_first','hp_last']].describe())
print('max |Total-HP-Other| where both present:',r.resid.max())
