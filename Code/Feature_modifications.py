#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import itertools


def create_seasons(df):
    #Lets create a column with the seasons of the year
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d')
    df['months'] =  df['time'].dt.month #extract months
#The months of the seasons depend if the location is in the Northern or Southern Hemisphere

    df["season"] = str(np.nan)

    for i in df.index:
        #Northern Hemisphere*************************
        if df.loc[i,"lat"]>=0:
            if (df.loc[i,"months"]==1) or (df.loc[i,"months"]==2) or (df.loc[i,"months"]==12):
                df.loc[i,"season"]="winter"
            if (df.loc[i,"months"]==3) or (df.loc[i,"months"]==4) or (df.loc[i,"months"]==5):
                df.loc[i,"season"]="spring"
            if (df.loc[i,"months"]==6) or (df.loc[i,"months"]==7) or (df.loc[i,"months"]==8):
                df.loc[i,"season"]="summer"
            if (df.loc[i,"months"]==9) or (df.loc[i,"months"]==10) or (df.loc[i,"months"]==11):
                df.loc[i,"season"]="fall"
                #Southern Hemisphere************************
        else:
            if (df.loc[i,"months"]==1) or (df.loc[i,"months"]==2) or (df.loc[i,"months"]==12):
                df.loc[i,"season"]="summer"
            if (df.loc[i,"months"]==3) or (df.loc[i,"months"]==4) or (df.loc[i,"months"]==5):
                df.loc[i,"season"]="fall"
            if (df.loc[i,"months"]==6) or (df.loc[i,"months"]==7) or (df.loc[i,"months"]==8):
                df.loc[i,"season"]="winter"
            if (df.loc[i,"months"]==9) or (df.loc[i,"months"]==10) or (df.loc[i,"months"]==11):
                df.loc[i,"season"]="spring"
                
    return df["season"]

def poly_degree2(df , list_num_col):
    
    for i in list_num_col:
        df[f"{i}_2"] = np.power(df[i],2)
        
    column_combinations = list(itertools.combinations(list_num_col, 2))
    for col1, col2 in column_combinations:
        df[f'{col1}_{col2}'] = df[col1] * df[col2]
    
    return df
    
    
    
    


