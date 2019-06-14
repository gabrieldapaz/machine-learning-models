# -*- coding: utf-8 -*-

#%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_aux = pd.DataFrame()

#carregando o dataset no datafram df
df = pd.read_csv('diabetic_data.csv')
print(df.shape)
print(df.info())
print(df.describe())

df.columns

df_aux.insert(loc=0,column ="Time in hospital",value= df['time_in_hospital'].describe())
df_aux.insert(loc =1,column ="# of lab procedures",value= df['num_lab_procedures'].describe())
df_aux.insert(loc =2,column ="# of procedures",value= df['num_procedures'].describe())
df_aux.insert(loc=3,column ="# of medications",value= df['num_medications'].describe())
df_aux.insert(loc=4,column ="# of outpatient visits",value= df['number_outpatient'].describe())
df_aux.insert(loc=5,column ="# of emergency visits",value= df['number_emergency'].describe())  
df_aux.insert(loc=6,column ="# of adimissions",value= df['number_inpatient'].describe())  
df_aux.insert(loc=7,column ="# of diagnoses",value= df['number_diagnoses'].describe())

df_aux = df_aux.transpose()
df_aux.drop(columns= ['count','25%', '50%', '75%'], axis=1, inplace= True)

df['gender'].value_counts(normalize=True) *100
df['race'].value_counts(normalize=True) *100
df['age'].value_counts(normalize=True) *100
df['readmitted'].value_counts(normalize=True) *100

for col in df.columns:
    if df[col].dtype == object:
         print(col,df[col][df[col] == '?'].count())
# gender was coded differently so we use a custom count for this one
print('gender', df['gender'][df['gender'] == 'Unknown/Invalid'].count())

#tirando campos de ID que não são necessários
df = df.drop(['weight','payer_code','medical_specialty'], axis = 1)
df = df.drop(['encounter_id','patient_nbr','admission_type_id','discharge_disposition_id','admission_source_id'],axis=1)
df = df.drop(['citoglipton', 'examide'], axis = 1)

#tirando as linhas com genero "Unknow/Invalid
df=df[df['gender']!='Unknown/Invalid']

df.replace('?', np.nan, inplace=True)
df.isnull().sum()

"""
from fancyimpute import KNN    
# X is the complete data matrix
# X_incomplete has the same values as X except a subset have been replace with NaN

# Use 3 nearest rows which have a feature to fill in each row's missing features
X_filled_knn = KNN(k=3).complete(X_incomplete)
"""

df.dropna(axis=0, inplace=True)
df.isnull().sum()


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

y = df['readmitted']
X = df.drop(columns='readmitted')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import statsmodels.formula.api as sm
X = np.append(arr =np.ones((98052,1)).astype(int), values = X, axis =1)
aux = np.arange(0,133)
aux.tolist()
X_opt = X[:,aux]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

def backwardElimination(x, sl):
    num_atri = len(x[0])
    for i in range(0, num_atri):
        regressor_OLS = sm.OLS(y, x).fit()
        max_atri = max(regressor_OLS.pvalues)
        if max_atri > sl:
            for j in range(0, num_atri - i):
                if (regressor_OLS.pvalues[j] == max_atri):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_Modeled = backwardElimination(X_opt, SL)

import statsmodels.formula.api as sm
X = np.append(arr =np.ones((98052,1)).astype(int), values = X, axis =1)
aux = np.arange(0,133)
aux.tolist()
X_opt = X[:,aux]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

def backwardElimination(x, sl):
    num_atri = len(x[0])
    for i in range(0, num_atri):
        regressor_OLS = sm.OLS(y, x).fit()
        max_atri = max(regressor_OLS.pvalues)
        if max_atri > sl:
            for j in range(0, num_atri - i):
                if (regressor_OLS.pvalues[j] == max_atri):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
df_corr = backwardElimination(X_opt, SL)


#X_train, X_test, y_train, y_test = train_test_split(X_Modeled, y, test_size=0.2, random_state=0)
#X_train.shape, X_test.shape

kf = KFold(n_splits=3, random_state=0, shuffle=False)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


result = []

rf = RandomForestClassifier(n_estimators=200, random_state=0)
rf = rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

result.append(['RF',accuracy_score(y_test, pred_rf)])

from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred_rf)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
train_results = []
test_results = []
for estimator in n_estimators:
   rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   pred_rf = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred_rf)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")
line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()

min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
   rf = RandomForestClassifier(min_samples_split=min_samples_split)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   pred_rf = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred_rf)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train AUC")
line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
print("Accuracy is {0:.2f}".format(accuracy_score(y_test, pred_rf)))
print("Precision is {0:.2f}".format(precision_score(y_test, pred_rf)))
print("Recall is {0:.2f}".format(recall_score(y_test, pred_rf)))
print("AUC is {0:.2f}".format(roc_auc_score(y_test, pred_rf)))

confusion_matrix(y_test, pred_rf)


mlp = MLPClassifier([100]*5, early_stopping=True, learning_rate='adaptive',random_state=0)
mlp = mlp.fit(X_train, y_train)
pred_mlp = mlp.predict(X_test)

result.append(['MLP', accuracy_score(y_test, pred_mlp)])

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
print("Accuracy is {0:.2f}".format(accuracy_score(y_test, pred_mlp)))
print("Precision is {0:.2f}".format(precision_score(y_test, pred_mlp)))
print("Recall is {0:.2f}".format(recall_score(y_test, pred_mlp)))
print("AUC is {0:.2f}".format(roc_auc_score(y_test, pred_mlp)))

confusion_matrix(y_test, pred_mlp)

train_input = X_Modeled
train_output = y
# Data balancing applied using SMOTE
from imblearn.over_sampling import SMOTE
from collections import Counter
print('Original dataset shape {}'.format(Counter(train_output)))
smt = SMOTE(random_state=20)
train_input_new, train_output_new = smt.fit_sample(train_input, train_output)
print('New dataset shape {}'.format(Counter(train_output_new)))
train_input_new = pd.DataFrame(train_input_new)
X_train, X_test, y_train, y_test = train_test_split(train_input_new, train_output_new, test_size=0.20, random_state=0)

regressor = DecisionTreeClassifier(max_depth=28, criterion = "entropy", min_samples_split=10)
regressor.fit(X_train, y_train)
dt_pred = regressor.predict(X_test)

result.append(['DT', accuracy_score(y_test, dt_pred)])

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
print("Accuracy is {0:.2f}".format(accuracy_score(y_test, dt_pred)))
print("Precision is {0:.2f}".format(precision_score(y_test, dt_pred)))
print("Recall is {0:.2f}".format(recall_score(y_test, dt_pred)))
print("AUC is {0:.2f}".format(roc_auc_score(y_test, dt_pred)))

confusion_matrix(y_test, dt_pred)

regressor = LogisticRegression(solver='liblinear',random_state=0)
regressor.fit(X_train, y_train)
pred_lr = regressor.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
print("Accuracy is {0:.2f}".format(accuracy_score(y_test, pred_lr)))
print("Precision is {0:.2f}".format(precision_score(y_test, pred_lr)))
print("Recall is {0:.2f}".format(recall_score(y_test, pred_lr)))
print("AUC is {0:.2f}".format(roc_auc_score(y_test, pred_lr)))

confusion_matrix(y_test, pred_lr)

result.append(['LR', accuracy_score(y_test, pred_lr)])

result = pd.DataFrame(result, columns=['Name', 'Accuracy'])
result = result.reset_index()
result

