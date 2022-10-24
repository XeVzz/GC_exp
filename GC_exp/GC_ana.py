"""
@author: Min Lun Wu
"""

import numpy as np
import pandas as pd
import os
import STGC as gc
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import kstest, normaltest
# from scipy import stats

from datetime import datetime
from math import floor, ceil

# %% functions
def normalize(data):
    op=(data-np.nanmean(data))/np.nanstd(data)
    
    return op

def df_plot(data):
    # clean nan
    data=data[~np.isnan(data)]
    res=(np.max(data)-np.min(data))/10
    # draw Hist
    counts, bins = np.histogram(data,bins=np.arange(floor(np.min(data)),ceil(np.max(data))+res,res))
    fig,ax=plt.subplots(1,1,figsize=(8,6))
    _,_,_=plt.hist(bins[:-1], bins, weights=counts/sum(counts),label='PDF hist')
    # draw CDF
    a=np.sort(data[~np.isnan(data)])    
    p = 1. * np.arange(len(a)) / float(len(a) - 1)    
    plt.plot(a, p,linewidth=2,label='CDF curve')
    # draw normal distribution CDF 
    loc,scale=norm.fit(data)
    n = norm(loc=loc, scale=scale)
    x = np.linspace(np.min(data), np.max(data), len(data))

    # x = np.arange(np.min(data), np.max(data), res)
    plt.plot(x, n.cdf(x),'r-.',label='norm. CDF')
    # k-s test
    # stat,p_val=ks_2samp(x, n.cdf(x))
    stat,p_val=kstest(data, 'norm',args=(data.mean(),data.std()))
    critical=1.36/np.sqrt(len(data))
    
    plt.text(floor(np.min(data)), 0.68, 'D/critical = '+str(round(stat/critical,6)), color='r', size=12,weight='bold')
    plt.text(floor(np.min(data)), 0.63, 'p_val = '+str(round(p_val,6)), color='r', size=12,weight='bold')
    # draw mean line
    plt.plot(np.full([5],np.mean(data)),np.arange(5),'k--',linewidth=2)
    plt.text(np.mean(data)+0.05, 0.95, 'mean = '+str(round(np.mean(data),3)), size=12,weight='bold')
    plt.grid()
    # show std
    std=np.std(data)
    plt.text(floor(np.min(data))+0.05, 0.58, 'std = '+str(round(std,3)), size=12,weight='bold')

    #fig set    
    ax.set_xlabel('bias',fontsize=15)
    ax.set_ylabel('probability',fontsize=15) 
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.yticks(np.arange(0,len(data),len(data)/10),np.round(np.arange(0,1,0.1),2))
    # plt.axis([int(np.nanmin(a))-1,int(np.nanmax(a))+1,0,1])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=16,loc=2) 

    return ax

def scat_plot(x,y):
    x=x[~np.isnan(x)]; x_std=np.std(x); x_mean=np.mean(x)
    y=y[~np.isnan(y)]; y_std=np.std(y); y_mean=np.mean(y)
    
    fig,ax=plt.subplots(1,1,figsize=(15,13))
    ax.plot(np.arange(100)-50,np.full([100],0),'k:',linewidth=3,alpha=0.3)
    ax.plot(np.full([100],0),np.arange(100)-50,'k:',linewidth=3,alpha=0.3)

    # plt.text(5, 7.5, 'dT_ref = '+str(np.nanstd(gc_log['dT'])), ha='left', wrap=True, size=14,weight='bold')
    # plt.text(5, 8.5, 'dRH_ref = '+str(np.nanstd(gc_log['dRH'])), ha='left', wrap=True, size=14,weight='bold')
    
    cir1=plt.Circle((x_mean/x_std, y_mean/y_std), 1, facecolor='none',edgecolor='g',linewidth=3,label='std range')
    cir2=plt.Circle((x_mean/x_std, y_mean/y_std), 2, facecolor='none',edgecolor='b',linewidth=3,label='std*2 range')
    cir3=plt.Circle((x_mean/x_std, y_mean/y_std), 3, facecolor='none',edgecolor='orange',linewidth=3,label='std*3 range')

    ax.add_patch(cir1)
    ax.add_patch(cir2)
    ax.add_patch(cir3)

    plt.plot(x_mean/x_std,y_mean/y_std,'*',c='r',markersize=12,label='mean')      
    plt.scatter(x/x_std, y/y_std, facecolors='none', edgecolors='k',s=50)

    ax.set_aspect('equal', 'box')
    ax.set_ylim(-10, 10)
    ax.set_xlim(-10, 10)
    
    plt.grid()
    ax.legend(fontsize=20) 
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # ax.set_xlabel('dT',fontsize=20)
    # ax.set_ylabel('dRH',fontsize=20)
    # plt.title('dT-dRH scatter',fontsize=20)

    return ax, [[x_std, y_std], [x_mean, y_mean]]

#%%  file path & list setting
ST_path='./ST_gc_no files/'

ST_files = [_ for _ in os.listdir(ST_path) if _.endswith(".csv")]
# VS_files = [_ for _ in os.listdir(VS_path) if _.endswith(".txt")]

st_vs_cc=0; gc_cc=0; gc_len=500

gc_log=dict()
new_gc_log=dict()
diff=dict()

no_launch=0
for key in ['dP', 'dT', 'dRH', 'num']:
    gc_log[key]=np.full(len(ST_files),np.nan)
    new_gc_log[key]=np.full(len(ST_files),np.nan)
    diff[key]=np.full(len(ST_files),np.nan)

mon_idx=np.full(len(ST_files),np.nan)
time_idx=np.full(len(ST_files),np.nan)

ori_clmns=['Time','Pressure(hPa)','Temperature(degree C)','Humidity(%)']

base_data=dict()

base_data[0]=pd.DataFrame(pd.read_csv(ST_path+ST_files[0]))
base_data[0].columns=base_data[0].columns.str.strip()
base_data[0]=base_data[0][ori_clmns].sort_values(by='Time').reset_index(drop=True)
base_data[0][ori_clmns[1]]=base_data[0][ori_clmns[1]]-1.6849
base_data[0][ori_clmns[2]]=base_data[0][ori_clmns[2]]-2.0124
base_data[0][ori_clmns[3]]=base_data[0][ori_clmns[3]]-0.1327

base_data[1]=pd.DataFrame(pd.read_csv(ST_path+ST_files[1]))
base_data[1].columns=base_data[1].columns.str.strip()
base_data[1]=base_data[1][ori_clmns].sort_values(by='Time').reset_index(drop=True)
base_data[1][ori_clmns[1]]=base_data[1][ori_clmns[1]]-2.2678
base_data[1][ori_clmns[2]]=base_data[1][ori_clmns[2]]-1.6153
base_data[1][ori_clmns[3]]=base_data[1][ori_clmns[3]]-3.6843

# %% file loop
# for i in np.arange(2,len(ST_files)):
for i in [5]:

# channel identify
    fname=ST_files[i].split('_')
    channel=int(fname[1][-1])
    
    if channel <5 :
        base=base_data[0]
        base_dt=base['Time'].apply(lambda x: datetime.strptime(x,'%Y/%m/%d %H:%M:%S.%f'))

    elif channel>=5 :
        base=base_data[1]
        base_dt=base['Time'].apply(lambda x: datetime.strptime(x,'%Y/%m/%d %H:%M:%S.%f'))

#  ST_file
    target_data=[]
    target_data=pd.DataFrame(pd.read_csv(ST_path+ST_files[i]))
    target_data.columns=target_data.columns.str.strip()
    target_data=target_data.sort_values(by='Time').reset_index(drop=True)
    target_dt=target_data['Time'].apply(lambda x: datetime.strptime(x,'%Y/%m/%d %H:%M:%S.%f'))

##find corresponding data
    crpd_num=np.full([len(target_data)],np.nan)
    crpd_data=pd.DataFrame()
    for num in range(len(target_data)):
        tmp=base_dt[(base_dt==target_dt[num])].index.tolist()
        if len(tmp) >=1:    
            crpd_num[num]=tmp[0]
            crpd_data=crpd_data.append(base.iloc[tmp[0],:], ignore_index=True)
        else:
            continue
         
    if len(crpd_data) <=120:
        base=base_data[1]
        base_dt=base['Time'].apply(lambda x: datetime.strptime(x,'%Y/%m/%d %H:%M:%S.%f'))
        crpd_num=np.full([len(target_data)],np.nan)
        crpd_data=pd.DataFrame()
        for num in range(len(target_data)):
            tmp=base_dt[(base_dt==target_dt[num])].index.tolist()
            if len(tmp) >=1:    
                crpd_num[num]=tmp[0]
                crpd_data=crpd_data.append(base.iloc[tmp[0],:], ignore_index=True)
            else:
                continue
            
    if len(crpd_data) <=120:
        print(ST_files[i]+' has no corresponding data')
        continue

    crpd_data=crpd_data.loc[:,ori_clmns].reset_index(drop=True)
    crpd_data.columns=['datetime','P','T','RH']

##
    P=target_data['Pressure(hPa)']
    T=target_data['Temperature(degree C)']
    RH=target_data['Humidity(%)']
    
    valid= (P>=0) & (P<1050) & (T!=-46.85) & (T!=40.99) & (RH>=0) & ~np.isnan(crpd_num)
    target=target_data.loc[valid,ori_clmns].reset_index(drop=True)
    target.columns=['datetime','P','T','RH']

# gc
    if len(target)<630:
        first=1-1; last=len(target)-1
    else:
        first=len(target)-10-600-1; last=len(target)-10-1
        
    gc_data=dict()
    
    gc_data['P_st']=target['P'][first:last].values
    gc_data['T_st']=target['T'][first:last].values
    gc_data['RH_st']=target['RH'][first:last].values

##
    gc_log['num'][i]=len(gc_data['P_st'])

    gc_data['P_obs']=crpd_data['P'][first:last].values
    gc_data['T_obs']=crpd_data['T'][first:last].values
    gc_data['RH_obs']=crpd_data['RH'][first:last].values

##
    gc_log['dP'][i],gc_log['dT'][i],gc_log['dRH'][i]=gc.bias(gc_data['P_st'],gc_data['T_st'],gc_data['RH_st'],
                                                             gc_data['P_obs'],gc_data['T_obs'],gc_data['RH_obs'])

    # if gc_log['dP'][i]>0.5:
        # gc_log['dP'][i]=0
    # if abs(gc_log['dT'][i])<=0.3:
        # gc_log['dP'][i]=0
    # if abs(gc_log['dRH'][i])<=2:
        # gc_log['dRH'][i]=0

# %% CDF & PDF
ax=df_plot(gc_log['dT'])
ax.set_xlabel('bias ($^\circ$C)',fontsize=15)
plt.title('dT distribution',fontsize=20)
plt.savefig('dT_dist', dpi=500) 

ax=df_plot(gc_log['dP'])
ax.set_xlabel('bias (hPa)',fontsize=15)
plt.title('dP distribution',fontsize=20)
plt.savefig('dP_dist', dpi=500) 

ax=df_plot(gc_log['dRH'])
ax.set_xlabel('bias (%)',fontsize=15)
plt.title('dRH distribution',fontsize=20)
plt.savefig('dRH_dist', dpi=500) 

# %% dT-dRH scatter
ax,par=scat_plot(gc_log['dT'],gc_log['dRH'])
plt.text(-9.7, -7.3, 'dT_std = '+str(round(par[0][0],3))+' $^\circ$C',ha='left', size=16,weight='bold')
plt.text(-9.7, -8, 'dRH_std = '+str(round(par[0][1],3))+' %', ha='left', size=16,weight='bold')
plt.text(-9.7, -9, 'dT_mean = '+str(round(par[1][0],3))+' $^\circ$C', ha='left', size=16,weight='bold',color='r')
plt.text(-9.7, -9.7, 'dRH_mean = '+str(round(par[1][1],3))+' %', ha='left', size=16,weight='bold',color='r')
ax.set_xlabel('dT / T_std',fontsize=24)
ax.set_ylabel('dRH / RH_std',fontsize=24)
plt.title('dT-dRH scatter',fontsize=30)
plt.savefig('dT-dRH_scatter', dpi=300)     

# %% dP-dT scatter
ax,par=scat_plot(gc_log['dP'],gc_log['dT'])
plt.text(-9.7, -7.3, 'dP_std = '+str(round(par[0][0],3))+' hPa', ha='left', size=16,weight='bold')
plt.text(-9.7, -8, 'dT_std = '+str(round(par[0][1],3))+' $^\circ$C', ha='left', size=16,weight='bold')
plt.text(-9.7, -9, 'dP_mean = '+str(round(par[1][0],3))+' hPa', ha='left', size=16,weight='bold',color='r')
plt.text(-9.7, -9.7, 'dT_mean = '+str(round(par[1][1],3))+' $^\circ$C', ha='left', size=16,weight='bold',color='r')
ax.set_xlabel('dP / P_std',fontsize=24)
ax.set_ylabel('dT / T_std',fontsize=24)
plt.title('dP-dT scatter',fontsize=30)
plt.savefig('dP-dT_scatter', dpi=300) 

# %% K-S test
data=gc_log['dT']
data=data[~np.isnan(data)]
# counts, bins = np.histogram(data,bins=np.arange(floor(np.min(data)),ceil(np.max(data))+res,res))
a=np.sort(data[~np.isnan(data)])    
p = 1. * np.arange(len(a)) / float(len(a) - 1)  

loc,scale=norm.fit(data)
n = norm(loc=loc, scale=scale)
x = np.linspace(np.min(data), np.max(data), len(data))
norm_p=n.cdf(x)
DD=n.cdf(x)-p
D=max(abs(DD))
critical=1.36/np.sqrt(len(data))

# stats,p_val=kstest(p, 'norm')

a=kstest(data, 'norm',args=(data.mean(),data.std()))
critical=1.36/np.sqrt(len(data))

aa=normaltest(data,nan_policy='omit')

# %%
# data=gc_log['dRH']
# data=data[~np.isnan(data)]
# ks_dRH=kstest(data, 'norm')
# loc,scale=norm.fit(data)
# n = norm(loc=loc, scale=scale)
# x = np.arange(np.min(data), np.max(data), 0.2)

# # stat,p_val=ks_2samp(x, n.cdf(x))

# # %%
# data=gc_log['dP']
# data=data[~np.isnan(data)]
# ks_dP=kstest(data, 'norm')
# loc,scale=norm.fit(data)
# n = norm(loc=loc, scale=scale)
# x = np.arange(np.min(data), np.max(data), 0.2)
# # stat,p_val=ks_2samp(x, n.cdf(x))



