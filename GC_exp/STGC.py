"""
@author: mlwu

Storm Tracker Ground Check
"""

import numpy as np

#%%
def lin_reg(x,y):
    
    xm=np.mean(x)
    ym=np.mean(y)
    
    yx=sum((x-xm)*(y-ym))
    xx=sum((x-xm)**2)
    
    b=yx/xx
    c=ym-b*xm
    d=np.nanmean(y-x)
    
    return b, c, d

#%%
def bias(P_st,T_st,RH_st,P_obs,T_obs,RH_obs):
    
    _,_,dP=lin_reg(P_st,P_obs)
    _,_,dT=lin_reg(T_st,T_obs)
    _,_,dRH=lin_reg(RH_st,RH_obs)

    # b_RH,c_RH,_=lin_reg(T_obs-T_st,RH_obs-RH_st)
    # dRH=b_RH*dT+c_RH
    
    return dP, dT, dRH

