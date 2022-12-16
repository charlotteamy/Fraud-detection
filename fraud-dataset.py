#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import matplotlib.pyplot as py
import seaborn as sns
import gc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn.metrics import accuracy_score
import graphviz 
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import plot_importance
from xgboost import to_graphviz
from sklearn import svm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


df1 = pd.read_csv('../input/paysim1/PS_20174392719_1491204439457_log.csv')


# # Understand the data

# In[3]:


df1.head()


# In[4]:


df1.info()


# In[5]:


df1.shape


# In[6]:


df1.isna().sum()


# In[ ]:





# # Target variable analysis

# In[7]:


df1['isFraud'].value_counts()


# Target variable is highly imbalanced will need to rebalance it.

# # Univariate analysis

# In[8]:


num_cols = [colname for colname in df1.columns if df1[colname].dtype in ['int64','float64']]
cat_cols = [colname for colname in df1.columns if (df1[colname].dtype in ['object'])]


# In[9]:


num_cols


# In[10]:


cat_cols


# In[11]:


#Numerical multi plot
fig = py.figure(figsize = (18,16))
for index,col in enumerate(num_cols[:9]):
    py.subplot(5,2,index+1)
    sns.kdeplot(df1.loc[:,col].dropna())
fig.tight_layout(pad = 1.0)


# In[12]:


df1['nameOrig'].value_counts().loc[lambda x : x>1]


# In[13]:


df1['nameDest'].value_counts().loc[lambda x : x>1]


# # Bivariate analysis

# In[14]:


sns.violinplot(x = df1['isFraud'], y = df1['amount'])


# In[15]:


#Numberical violin plots
fig3 = py.figure(figsize = (18,20))
for index,i in enumerate(num_cols[:6]):
    py.subplot(5,2,index+1)
    sns.violinplot(x = df1['isFraud'],y = df1[i])
fig3.tight_layout(pad = 1.0)


# In[16]:


df2 = df1[df1['amount']<150000]


# In[17]:


sns.violinplot(x = df2['isFraud'],y = df1['amount'])


# In[18]:


sns.violinplot(x = df1['isFraud'],y = df1[df1['oldbalanceOrg']<100000].oldbalanceOrg)


# In[19]:


sns.violinplot(x = df1['isFraud'],y = df1[df1['newbalanceOrig']<10000].newbalanceOrig)


# In[20]:


px.histogram(df1[df1['isFraud'] == 1].newbalanceOrig)


# In[21]:


gc.collect()


# In[22]:


pd.crosstab(index=df1['isFraud'], columns=df1['type'], values=df1['type'], aggfunc=pd.Series.count)


# In[23]:


df1[df1['isFraud'] == 1]['nameDest'].value_counts().loc[lambda x : x>1]


# In[24]:


df1[df1['isFraud'] == 1]['nameOrig'].value_counts().loc[lambda x : x>1]


# In[25]:


#How many transactions have destination balances being zero?

df1['zerobaldes'] = (df1['oldbalanceDest']) ==0 & (df1['newbalanceDest'] ==0)
df1['error'] = (df1['oldbalanceDest'] ==0) & (df1['newbalanceDest'] ==0) & (df1['amount'] != 0)
print(df1['zerobaldes'].value_counts())
print(df1['error'].value_counts())


# In[26]:


fraud = df1[df1['isFraud'] == 1]
notfraud = df1[df1['isFraud'] == 0]

print('The proportion of fraudulent transactions where the destination balances are zero are:', fraud['zerobaldes'].sum() / len(fraud['zerobaldes']) *100, '%')
print('The proportion of non fraudulent transactions where the destination balances are zero are:', notfraud['zerobaldes'].sum() / len(notfraud['zerobaldes']) *100, '%')


# In[27]:


df1[df1['amount'] ==0]


# # Pre-processing
# Steps required:
# Split into test and train data
# Rebalance the dataset
# Scale numberical variables
# Encode categorical variables

# In[28]:


#Add new variables
df1['zerobaldes'] = (df1['oldbalanceDest']) ==0 & (df1['newbalanceDest'] ==0)
df1['error'] = (df1['oldbalanceDest'] ==0) & (df1['newbalanceDest'] ==0) & (df1['amount'] != 0)


# In[29]:


X = df1.drop(columns = ['isFraud','isFlaggedFraud'])
y = df1['isFraud']


# In[30]:


print(X.shape)
print(y.shape)


# In[31]:


del df1


# In[32]:


gc.collect()


# In[33]:


x_train, x_test, y_train, y_test = train_test_split(X, y)


# In[34]:


x_train.head()


# In[35]:


x_test.head()


# In[36]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[37]:


x_train.columns


# In[38]:


#Remove columns that will be tricky to process for now
x_train.drop(columns = ['nameOrig','nameDest'], inplace = True)
x_test.drop(columns = ['nameOrig','nameDest'], inplace = True)


# In[39]:


x_train.head()


# In[40]:


#One hot encode the remaining categorical variable
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(x_train[['type']]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(x_test[['type']]))


# In[41]:


OHcolnames = OH_encoder.get_feature_names_out()
OH_cols_train.columns = OHcolnames
OH_cols_test.columns = OHcolnames


# In[42]:


# One-hot encoding removed index; put it back
OH_cols_train.index = x_train.index
OH_cols_test.index = x_test.index


# In[43]:


# Remove categorical columns (will replace with one-hot encoding)
x_train.drop(columns = 'type', inplace = True)
x_test.drop(columns = 'type', inplace = True)


# In[44]:


x_train.head()


# In[45]:


# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([x_train, OH_cols_train], axis=1)
OH_X_test = pd.concat([x_test, OH_cols_test], axis=1)


# In[46]:


OH_X_train.info()


# In[47]:


# Scale the data for continuous variables on training data
OH_X_train_s_cols = OH_X_train.loc[:,'step':'newbalanceDest'].columns
OH_X_train_s_index = OH_X_train.loc[:,'step':'newbalanceDest'].index

OH_X_test_s_cols = OH_X_test.loc[:,'step':'newbalanceDest'].columns
OH_X_test_s_index = OH_X_test.loc[:,'step':'newbalanceDest'].index

# Set up the standard scaler
scaler = StandardScaler()

OH_X_train_s = pd.DataFrame(scaler.fit_transform(OH_X_train.loc[:,'step':'newbalanceDest']))
OH_X_test_s = pd.DataFrame(scaler.transform(OH_X_test.loc[:,'step':'newbalanceDest']))

# Add the columns and index back in
OH_X_train_s.columns = OH_X_train_s_cols
OH_X_train_s.index = OH_X_train_s_index

OH_X_test_s.columns = OH_X_test_s_cols
OH_X_test_s.index = OH_X_test_s_index

#Merge the numerical and categorical data back together
OH_X_train_s_en = pd.concat([OH_X_train_s, OH_X_train.loc[:,'zerobaldes':'type_TRANSFER']], axis=1)

OH_X_test_s_en = pd.concat([OH_X_test_s, OH_X_test.loc[:,'zerobaldes':'type_TRANSFER']], axis=1)


# In[48]:


y_train.head()


# In[49]:


print('The count of records in the train dataset is')
print(y_train.count())
print('The proportion of fraud in the train dataset is')
print(y_train.sum())


# In[50]:


print('The count of records in the test dataset is')
print(y_test.count())
print('The proportion of fraud in the test dataset is')
print(y_test.sum())


# In[51]:


#Upsample the fraudlent cases
#Combine train datasets back together

dftrain = OH_X_train_s_en
dftrain['isFraud'] = y_train
dftrain


# In[52]:


# Separate the case of fraud and no fraud
df_0 = dftrain[dftrain.isFraud == 0]
df_1 = dftrain[dftrain.isFraud == 1]

## Upsample the minority class
df_minorityupsampled = resample(df_1, n_samples=len(df_1)*4) 

#Downsample the majority class

df_majoritydownsampled = resample(df_0, n_samples=len(df_minorityupsampled)*5) 

# Combine majority class with upsampled minority class
new_df = pd.concat([df_majoritydownsampled, df_minorityupsampled])
new_df = shuffle(new_df)


# In[53]:


print('The count of records in the resampled dataset is')
print(new_df['step'].count())
print('The amount of fraud in the resampled dataset is')
print(new_df['isFraud'].sum())
print('The proportion of fraud in the resampled dataset is')
print((new_df['isFraud'].sum()/new_df['step'].count())*100)


# In[54]:


#Split the target variable back out again
y_train = new_df['isFraud']
X_train = new_df.drop(columns = 'isFraud')


# # Build the model - decision tree 1

# In[55]:


clf = tree.DecisionTreeClassifier()


# In[56]:


#param_grid = {
   # "max_depth": [3,5,10,15,20,None],
    #"min_samples_split": [2,5,7,10],
   # "min_samples_leaf": [1,2,5]
#}
#grid_cv = GridSearchCV(clf, param_grid, scoring="roc_auc", n_jobs=-1, cv=3).fit(OH_X_train, y_train)
#print("Param for GS", grid_cv.best_params_)
#print("CV score for GS", grid_cv.best_score_)
#print("Train AUC ROC Score for GS: ", roc_auc_score(y_train, grid_cv.predict(X_train)))
#print("Test AUC ROC Score for GS: ", roc_auc_score(y_test, grid_cv.predict(X_test)))


# In[57]:


clf.fit(X_train,y_train)
predictions_dt = clf.predict(OH_X_test_s_en)
predictions_dtp = clf.predict_proba(OH_X_test_s_en)


# In[58]:


fpr2, tpr2, thresholds_keras = roc_curve(y_test, predictions_dtp[:,1])


# In[59]:


DTAUCscore = roc_auc_score(y_test, predictions_dtp[:,1])
DTAUCscore


# In[60]:


px.line(x = fpr2, y = tpr2, title = 'ROC curve')


# # Build model - XGBoost Classifier

# In[61]:


get_ipython().run_cell_magic('time', '', 'xgb_model = xgb.XGBClassifier(max_depth = 2)')


# In[62]:


cross_val_score(xgb_model,X_train,y_train,scoring = 'roc_auc' )


# In[63]:


xgb_model.fit(X_train, y_train)


# In[64]:


predictions = xgb_model.predict(OH_X_test_s_en)
predictions_p = xgb_model.predict_proba(OH_X_test_s_en)


# In[65]:


fpr, tpr, thresholds_keras = roc_curve(y_test, predictions_p[:,1])


# In[66]:


px.line(x = fpr, y = tpr, title = 'ROC curve')


# In[67]:


XGBoostAUC = roc_auc_score(y_test.values, predictions_p[:,1])
XGBoostAUC


# In[68]:


print('hello')


# # Plot the most important metrics

# In[69]:


xgb_model.get_booster().feature_names


# In[70]:


plot_importance(xgb_model)


# In[71]:


to_graphviz(xgb_model)


# In[ ]:





# In[ ]:





# In[ ]:




