import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#reading data
dataset=pd.read_csv('covid.csv')
dataset.set_index('Country', inplace=True)
df_train=dataset.iloc[:-1,:-4]
df_test=dataset.iloc[-1,:-4]
df_test=df_test.to_frame()
df_test=df_test.transpose()

x=df_train.iloc[:,[0,1,2,3,4,5,7,8,9,11]]
y=df_train.iloc[:,10]

#back-elimination for feature selection
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((19,1)).astype(int), values=x,axis=1)
x_opt=x[:,[0,4,5,6,7,9,10,13]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt,axis=1).fit()
regressor_OLS.summary()

x=x.iloc[:,[3,4,5,6,8,9,12]]

corrmat=x.corr()
f,ax=plt.subplots(figsize=(9,16))
sns.heatmap(corrmat, vmax=1, square=True)
""" 3rd week most crucial. Total cases depend on it
Total cases highly during lockdown"""

x['Recovery rate']=x['Recovered']/x['Total cases']
x['Death rate']=x['Deaths']/x['Total cases']
x['cases during lockdown']=x['Total cases']-x['Cases before Lockdown']

df_test['Recovery rate']=df_test['Recovered']/df_test['Total cases']
df_test['Death rate']=df_test['Deaths']/df_test['Total cases']
df_test['cases during lockdown']=df_test['Total cases']-df_test['Cases before Lockdown']
df_test=df_test.iloc[:,[3,4,5,7,9,11,14]]

#Assumed the max of the data 
df_test.at['dip after lockdown days']=33

x_train=x.iloc[:-2,:]
x_test=x.iloc[[-2,-1],:]
y_train=y.iloc[:-2]
y_test=y.iloc[[-2,-1]]

#model-fitting
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred=regressor.predict(x_test)

import sklearn
sklearn.metrics.r2_score(y_test,y_pred)

""" lockdown of 51 days acc to model with assumed value of dip 33 without scale-accuracy 93%"""