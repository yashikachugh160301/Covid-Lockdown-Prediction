import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('covid.csv')
dataset.set_index('Country', inplace=True)
dataset=dataset.drop(dataset.index[0])
df=dataset.iloc[:,[0,2,3,4,5,14,15,16,17,19]]

x=df.loc[:, df.columns!='Lockdown days']
y=df.iloc[:,5]

x_train=x.iloc[:-2,:]
x_test=x.iloc[-1,:]

y_train=y.iloc[:-2]
y_test=y.iloc[-1]

x_train.isnull().sum()

#data interpolation

x_train=x_train.fillna(np.nan)
x_train=x_train.interpolate(method='linear').round()

#visualisation
import seaborn as sns
corrmat=x_train.corr()
f,ax=plt.subplots(figsize=(9,16))
sns.heatmap(corrmat, vmax=1, square=True)

#recovery rate
x_train['Recovery rate']=x_train['Recovered']/x_train['Total cases']
x_test['Recovery rate']=x_test['Recovered']/x_test['Total cases']

sns.lineplot(x=df['Lockdown days'],y=x_train['Recovery rate'])
sns.lineplot(x=x_train.iloc[:,8],y=x_train['Recovery rate'])

x_train['dip after lockdown days'].describe()
 
import sympy
reduced_form,inds=sympy.Matrix(x_train.values).rref()
"""no two factors are linearly dependent"""

x_train['cases during lockdown']= x_train['Total cases']-x_train['Cases before Lockdown']
x_test['cases during lockdown']= x_test['Total cases']-x_test['Cases before Lockdown']
sns.lineplot(x=x_train.iloc[:,8],y=x_train.iloc[:,10])

sns.lineplot(x=df['Lockdown days'],y=x_train.iloc[:,10])


# predict dip time otherwise assume 32 days- max acc. to data
x_dummy=x_train.iloc[:,[0,1,2,3,4,6,7,8,9,10]]
y_dummy=x_train['dip after lockdown days']
y_dummy=y_dummy.to_frame()

x_dummy_test=x.iloc[-1,[0,1,2,3,4,6,7,8]]
x_dummy_test['Recovery rate']=x_dummy_test['Recovered']/x_dummy_test['Total cases']
x_dummy_test['cases during lockdown']= x_dummy_test['Total cases']-x_dummy_test['Cases before Lockdown']
x_dummy_test=x_dummy_test.fillna(18422)

""" Quadratic interpolation gives 18422-week 4"""
x_dummy_test=x_dummy_test.to_frame()
x_dummy_test=x_dummy_test.transpose()
'fitting the model'
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_dummy, y_dummy)

# Predicting the Test set results
y_pred = regressor.predict(x_dummy_test).round()

""" filling in x-test"""
x_test=x_test.to_frame()
x_test=x_test.transpose()
x_test.at['India','dip after lockdown days']=26
x_test=x_test.iloc[0,:]
x_test=x_test.fillna(np.nan)
x_test=x_test.fillna(18422)

''' fitting model to find number of lockdown days'''
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x_train, y_train)

# Predicting a new result
y_pred = regressor.predict(x_test).round()
""" Around 50 days of lcokdown is required for INDIA right now we have for 40 days"""

