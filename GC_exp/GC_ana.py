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
def ct(cid):
    color_table=dict()
    color_table['fr']=(0.74, 0.11, 0.11) #filezilla red
    color_table['mo']=(0.93, 0.56, 0.23) #matlab orange
    color_table['eg']=(0.12, 0.44, 0.27) #excel green
    color_table['wb']=(0.16, 0.34, 0.60) #word blue 
    color_table['gr']=(0.86, 0.64, 0.49) #ground

    cidx=list(color_table.keys())
    if cid in color_table:
        key=cid
    else:
        key=cidx[cid]

    return color_table[key]

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

def scat_plot(x,y,colormsk=[]):
    x=x[~np.isnan(x)]; x_std=np.std(x); x_mean=np.mean(x);
    y=y[~np.isnan(y)]; y_std=np.std(y); y_mean=np.mean(y)
    
    fig,ax=plt.subplots(1,1,figsize=(15,13))
    ax.plot(np.arange(100)-50,np.full([100],0),'k:',linewidth=3,alpha=0.3)
    ax.plot(np.full([100],0),np.arange(100)-50,'k:',linewidth=3,alpha=0.3)
    
    if len(np.unique(colormsk))==0:
        plt.plot(x_mean/x_std,y_mean/y_std,'*',c=ct(0),markersize=12,label='mean')  
        cir1=plt.Circle((x_mean/x_std, y_mean/y_std), 1, facecolor='none',edgecolor=ct(1),linewidth=3,label='std range')
        cir2=plt.Circle((x_mean/x_std, y_mean/y_std), 2, facecolor='none',edgecolor=ct(2),linewidth=3,label='std*2 range')
        cir3=plt.Circle((x_mean/x_std, y_mean/y_std), 3, facecolor='none',edgecolor=ct(3),linewidth=3,label='std*3 range')
        ax.add_patch(cir1); ax.add_patch(cir2); ax.add_patch(cir3)
        plt.scatter(x/x_std, y/y_std, facecolors='none', edgecolors='k',alpha=0.7,s=50)

    else :
        plt.plot(x_mean/x_std,y_mean/y_std,'*',c='k',markersize=12,label='mean')
        cir1=plt.Circle((x_mean/x_std, y_mean/y_std), 1, facecolor='none',edgecolor='k',alpha=0.8,linewidth=1.5,label='std range')
        cir2=plt.Circle((x_mean/x_std, y_mean/y_std), 2, facecolor='none',edgecolor='k',linestyle='dashed',linewidth=1.5,label='std*2 range')
        cir3=plt.Circle((x_mean/x_std, y_mean/y_std), 3, facecolor='none',edgecolor='k',linestyle='dotted',linewidth=2.3,label='std*3 range')
        ax.add_patch(cir1); ax.add_patch(cir2); ax.add_patch(cir3)
        for color in np.unique(colormsk):
            plt.scatter(x[colormsk==color]/x_std, y[colormsk==color]/y_std, facecolors='none', edgecolors=color,alpha=0.5,s=40)
                 
    ax.set_aspect('equal', 'box')
    ax.set_ylim(floor(y_mean/y_std)-10, ceil(y_mean/y_std)+10)
    ax.set_xlim(floor(x_mean/x_std)-10, ceil(x_mean/x_std)+10)
    
    plt.grid()
    ax.legend(fontsize=20) 
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    return ax, [[x_std, y_std], [x_mean, y_mean]]

#%%  file path & list setting
ST_path='./ST_gc_no files/'
ST_files = [_ for _ in os.listdir(ST_path) if _.endswith(".csv")]

gc_log=dict()
for key in ['dP', 'dT', 'dRH', 'num']:
    gc_log[key]=np.full(len(ST_files),np.nan)

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

base=pd.concat([base_data[0],base_data[1]]).sort_values(by='Time').reset_index(drop=True)
# base_dt=base['Time'].apply(lambda x: datetime.strptime(x,'%Y/%m/%d %H:%M:%S.%f'))

# %% file loop
for i in np.arange(2,len(ST_files)):
# for i in [4]:
    
#  ST_file
    target_data=[]
    target_data=pd.DataFrame(pd.read_csv(ST_path+ST_files[i]))
    target_data.columns=target_data.columns.str.strip()
    target_data=target_data.sort_values(by='Time').reset_index(drop=True)
    # target_dt=target_data['Time'].apply(lambda x: datetime.strptime(x,'%Y/%m/%d %H:%M:%S.%f'))

# find corresponding data
    crpd_tag=base['Time'].isin(target_data['Time'])
    crpd_data=base[crpd_tag]

    if len(crpd_data) <=120:
        print(ST_files[i]+' has no corresponding data')
        continue

    crpd_data=crpd_data.loc[:,ori_clmns].reset_index(drop=True)
    crpd_data.columns=['datetime','P','T','RH']

# target data generate
    P=target_data['Pressure(hPa)']
    T=target_data['Temperature(degree C)']
    RH=target_data['Humidity(%)']
    
    valid= (P>=0) & (P<1050) & (T!=-46.85) & (T!=40.99) & (RH>=0) & target_data['Time'].isin(base['Time'])
    target=target_data.loc[valid,ori_clmns].reset_index(drop=True)
    target.columns=['datetime','P','T','RH']

# gc data prepare
    if len(target)<630:
        first=1-1; last=len(target)-1
    else:
        first=len(target)-10-600-1; last=len(target)-10-1

    gc_data=dict()
    gc_data['P_st']=target['P'][first:last].values
    gc_data['T_st']=target['T'][first:last].values
    gc_data['RH_st']=target['RH'][first:last].values

    gc_data['P_obs']=crpd_data['P'][first:last].values
    gc_data['T_obs']=crpd_data['T'][first:last].values
    gc_data['RH_obs']=crpd_data['RH'][first:last].values
    
    gc_log['num'][i]=len(gc_data['P_st'])
    gc_log['dP'][i],gc_log['dT'][i],gc_log['dRH'][i]=gc.bias(gc_data['P_st'],gc_data['T_st'],gc_data['RH_st'],
                                                             gc_data['P_obs'],gc_data['T_obs'],gc_data['RH_obs'])

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

# %% scatter color as batch
# colormsk=np.concatenate((np.full(116,'b'), np.full(154,'r')))
# colormsk=colormsk[~np.isnan(gc_log['dT'])]

# %% dT-dRH scatter
ax,par=scat_plot(gc_log['dT'],gc_log['dRH'])
xlim=par[1][0]/par[0][0]-10
ylim=par[1][1]/par[0][1]-10
plt.text(xlim+0.3, ylim+2.7, 'dT_std = '+str(round(par[0][0],3))+' $^\circ$C',ha='left', size=16,weight='bold')
plt.text(xlim+0.3, ylim+2, 'dRH_std = '+str(round(par[0][1],3))+' %', ha='left', size=16,weight='bold')
plt.text(xlim+0.3, ylim+1, 'dT_mean = '+str(round(par[1][0],3))+' $^\circ$C', ha='left', size=16,weight='bold',color='r')
plt.text(xlim+0.3, ylim+0.3, 'dRH_mean = '+str(round(par[1][1],3))+' %', ha='left', size=16,weight='bold',color='r')
ax.set_xlabel('dT / T_std',fontsize=24)
ax.set_ylabel('dRH / RH_std',fontsize=24)
plt.title('dT-dRH scatter',fontsize=30)
plt.savefig('dT-dRH_scatter', dpi=300)     

# %% dP-dT scatter
ax,par=scat_plot(gc_log['dP'],gc_log['dT'])
xlim=par[1][0]/par[0][0]-10
ylim=par[1][1]/par[0][1]-10
plt.text(xlim+0.3, ylim+2.7, 'dP_std = '+str(round(par[0][0],3))+' hPa', ha='left', size=16,weight='bold')
plt.text(xlim+0.3, ylim+2, 'dT_std = '+str(round(par[0][1],3))+' $^\circ$C', ha='left', size=16,weight='bold')
plt.text(xlim+0.3, ylim+1, 'dP_mean = '+str(round(par[1][0],3))+' hPa', ha='left', size=16,weight='bold',color='r')
plt.text(xlim+0.3, ylim+0.3, 'dT_mean = '+str(round(par[1][1],3))+' $^\circ$C', ha='left', size=16,weight='bold',color='r')
ax.set_xlabel('dP / P_std',fontsize=24)
ax.set_ylabel('dT / T_std',fontsize=24)
plt.title('dP-dT scatter',fontsize=30)
plt.savefig('dP-dT_scatter', dpi=300) 

