# -*- coding: utf-8 -*-

#%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#carregando o dataset no datafram df
df = pd.read_csv('diabetic_data.csv')
#substituindo os campos com ? para nan
df.replace('?', np.nan, inplace=True)

#tirando campos de ID que não são necessários
df = df.drop(['encounter_id','patient_nbr','admission_type_id','discharge_disposition_id','admission_source_id'],axis=1)
#tirando as linhas com nan
df.dropna(inplace=True)
#tirando as linhas com genero "Unknow/Invalid
df=df[df['gender']!='Unknown/Invalid']

#colocando uma series com true para os campos com NO e colocando 0 para NO e 1 para Yes ou >30 <30
condition = df['readmitted']!='NO'
df['readmitted'] = np.where(condition,1,0)

diag_cols = ['diag_1','diag_2','diag_3']
for col in diag_cols:
    df[col] = df[col].str.replace('E','-')
    df[col] = df[col].str.replace('V','-')
    condition = df[col].str.contains('250')
    df.loc[condition,col] = '250'

df[diag_cols] = df[diag_cols].astype(float)

# diagnosis grouping
for col in diag_cols:
    df['temp']=np.nan
    
    condition = df[col]==250
    df.loc[condition,'temp']='Diabetes'
    
    condition = (df[col]>=390) & (df[col]<=458) | (df[col]==785)
    df.loc[condition,'temp']='Circulatory'
    
    condition = (df[col]>=460) & (df[col]<=519) | (df[col]==786)
    df.loc[condition,'temp']='Respiratory'
    
    condition = (df[col]>=520) & (df[col]<=579) | (df[col]==787)
    df.loc[condition,'temp']='Digestive'
    
    condition = (df[col]>=580) & (df[col]<=629) | (df[col]==788)
    df.loc[condition,'temp']='Genitourinary'
    
    condition = (df[col]>=800) & (df[col]<=999)
    df.loc[condition,'temp']='Injury'
    
    condition = (df[col]>=710) & (df[col]<=739)
    df.loc[condition,'temp']='Muscoloskeletal'
    
    condition = (df[col]>=140) & (df[col]<=239)
    df.loc[condition,'temp']='Neoplasms'
    
    condition = df[col]==0
    df.loc[condition,col]='?'
    df['temp']=df['temp'].fillna('Others')
    condition = df['temp']=='0'
    df.loc[condition,'temp']=np.nan
    df[col]=df['temp']
    df.drop('temp',axis=1,inplace=True)

df.dropna(inplace=True)

cat_cols = list(df.select_dtypes('object').columns)
for col in cat_cols:
    df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col])], axis=1)
test = df.columns

from sklearn import preprocessing
x = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

df.columns = test

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(df.drop('readmitted', axis=1), df['readmitted'], test_size=0.3, random_state=2)

X_train.shape, X_test.shape

result = []

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf = rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

result.append(['RF',accuracy_score(y_test, pred_rf)])

mlp = MLPClassifier([100]*5, early_stopping=True, learning_rate='adaptive',random_state=0)
mlp = mlp.fit(X_train, y_train)
pred_mlp = mlp.predict(X_test)

result.append(['MLP', accuracy_score(y_test, pred_mlp)])

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
dt_pred = regressor.predict(X_test)

result.append(['DT', accuracy_score(y_test, dt_pred)])


regressor = LogisticRegression(solver='liblinear',random_state=0)
regressor.fit(X_train, y_train)

pred_lr = regressor.predict(X_test)

pred_svr = regressor.predict(X_test)

result.append(['LR', accuracy_score(y_test, pred_lr)])

result = pd.DataFrame(result, columns=['Name', 'Accuracy'])
result = result.reset_index()
result

