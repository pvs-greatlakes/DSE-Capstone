# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:35:40 2024

@author: PVS
"""

import sys, os
sys.path.insert(0, r'D:/Training/Python') 
from   models_comp  import  draw_cm, draw_roc, evaluate_model_SM_performance, calculate_VIF, check_linearity, draw_countplot, draw_boxplot, IsFloat, build_LR, build_models
import pandas            as  pd
import numpy             as  np
import seaborn           as  sns
import matplotlib.pyplot as  plt
import statsmodels.api   as  sm
print("\nCurrent working directory %s" %os.getcwd())

os.chdir(r'G:\DSE-FT-C-May24-G5\Data')

file  = 'G:\DSE-FT-C-May24-G5\Data\Customer-Churn-Records.csv'
df    =  pd.read_csv(file, encoding = 'Latin-1')
print(df.columns)
print("\n Rows {0} Columns {1}".format(df.shape[0], df.shape[1]))
print(df.info())
n               =   df.shape[0]
Xcols           =   ['CreditScore', 'Geography','Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',\
                    'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Complain', 'Satisfaction Score', 'Card Type', 'Point Earned']
ycols           =   ['Exited']
X               =   df.loc[:, Xcols]
y               =   df.loc[:, ycols]               
target          =   'Exited'  
classlist       =   ['Churn', 'No Churn']

### Build models
build_models(df, Xcols, ycols, classlist)
### END