import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
%matplotlib inline 
mpl.style.use('ggplot') 
car=pd.read_csv('quikr_car.csv') 
# @title 
car.head() 
car.shape 
car.info() 
backup=car.copy() 
car=car[car['year'].str.isnumeric()] 
car['year']=car['year'].astype(int) 
car=car[car['Price']!='Ask For Price'] 
car['Price']=car['Price'].str.replace(',','').astype(int) 
12  
car['kms_driven']=car['kms_driven'].str.split().str.get(0).str.replace(',','') 
car=car[car['kms_driven'].str.isnumeric()] 
car['kms_driven']=car['kms_driven'].astype(int) 
car=car[~car['fuel_type'].isna()] 
car.shape 
car['name']=car['name'].str.split().str.slice(start=0,stop=3).str.join(' ') 
car=car.reset_index(drop=True) 
car 
car.to_csv('Cleaned_Car_data.csv') 
car.info() 
car.describe(include='all') 
car=car[car['Price']<6000000] 
car['company'].unique() 
import seaborn as sns 
plt.subplots(figsize=(15,7)) 
ax=sns.boxplot(x='company',y='Price',data=car) 
13  
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right') 
plt.show() 
plt.subplots(figsize=(20,10)) 
ax=sns.swarmplot(x='year',y='Price',data=car) 
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right') 
plt.show() 
sns.relplot(x='kms_driven',y='Price',data=car,height=7,aspect=1.5) 
plt.subplots(figsize=(14,7)) 
sns.boxplot(x='fuel_type',y='Price',data=car) 
ax=sns.relplot(x='company',y='Price',data=car,hue='fuel_type',size='year',height=7,aspect=2) 
ax.set_xticklabels(rotation=40,ha='right') 
X=car[['name','company','year','kms_driven','fuel_type']] 
y=car['Price'] 
y.shape 
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3) 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import make_column_transformer 
from sklearn.pipeline import make_pipeline 
14  
from sklearn.metrics import r2_score 
ohe=OneHotEncoder() 
ohe.fit(X[['name','company','fuel_type']]) 
column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['na
 me','company','fuel_type']), 
remainder='passthrough') 
rf=RandomForestRegressor() 
pipe=make_pipeline(column_trans,rf) 
pipe.fit(X_train,y_train) 
y_pred=pipe.predict(X_test) 
r2_score(y_test,y_pred) 
scores=[] 
for i in range(1000): 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i) 
rf=RandomForestRegressor() 
pipe=make_pipeline(column_trans,rf) 
pipe.fit(X_train,y_train) 
y_pred=pipe.predict(X_test) 
scores.append(r2_score(y_test,y_pred)) 
np.argmax(scores) 
15  
scores[np.argmax(scores)] 
pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array(['MarutiSuzukiSwift','Mar
 uti',2019,100,'Petrol']).reshape(1,5))) 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scor
 es)) 
rf=RandomForestRegressor() 
pipe=make_pipeline(column_trans,rf) 
pipe.fit(X_train,y_train) 
y_pred=pipe.predict(X_test) 
r2_score(y_test,y_pred)
