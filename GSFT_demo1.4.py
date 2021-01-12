# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 03:33:09 2021

@author: asus
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale

import pandas as pd
import numpy as np

#%matplotlib inline
df= pd.read_csv('C:/Users/asus/Downloads/train.csv')
dftmp= pd.read_csv('C:/Users/asus/Downloads/train.csv')
dftmp2= pd.read_csv('C:/Users/asus/Downloads/train.csv')
df.hist(bins=50 , figsize=(20,20))
plt.show()
corr_mat = df.corr()

fig2=plt.figure()
sns.set(rc={'figure.figsize':(20,15)})
k = 41
cols = corr_mat.nlargest(k, 'loan_default')['loan_default'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.title('Correlation Matrix')
plt.show()
sv = corr_mat['loan_default'].sort_values(ascending = False)
#The repartition of the target
fig7=plt.figure()
ax7=plt.axes()
the_target = dftmp2['loan_default']
the_target.replace(to_replace=[1,0], value= ['YES','NO'], inplace = True)
plt.title('Target repartition')
ax7 = ax7.set(xlabel='Default proportion')
the_target.value_counts().plot.pie()
plt.show()
df.dtypes
def nan_count_df(df_to_print):
    
    nan_count = df_to_print.isnull().sum()

    nan_percentage = (nan_count / len(df))*100

    nan_df=pd.concat([nan_percentage], axis=1)
    nan_df=nan_df.rename(columns={0:'Percentage'})
    nan_df=nan_df[nan_df.Percentage != 0]
    nan_df = nan_df.sort_values(by='Percentage',ascending=False)
    return nan_df

nan_count_df(df)
df = df.fillna(df.mode().iloc[0])
df=df.rename(columns={'Date.of.Birth': 'Date_of_Birth','Employment.Type': 'Employment_Type', 'PERFORM_CNS.SCORE.DESCRIPTION': 'PERFORM_CNS_SCORE_DESCRIPTION'})

df.columns
now = pd.Timestamp('now')
df['Date_of_Birth'] = pd.to_datetime(df['Date_of_Birth'], format='%d-%m-%y')
df['Date_of_Birth'] = df['Date_of_Birth'].where(df['Date_of_Birth'] < now, df['Date_of_Birth'] -  np.timedelta64(100, 'Y'))
df['Age'] = (now - df['Date_of_Birth']).astype('<m8[Y]')
#sns.distplot(df['Age']
df['Age'] = np.log(df['Age'])

#dummy =df.dtypes 
def two_cat_encoding(df_to_transf):
    le = LabelEncoder()

    for cols in df_to_transf:
        if df_to_transf[cols].dtype == 'object':
            if len(list(df_to_transf[cols].unique())) == 2:
                le.fit(df_to_transf[cols])
                df_to_transf[cols] = le.transform(df_to_transf[cols])
    return df_to_transf
df=two_cat_encoding(df)
df["PERFORM_CNS_SCORE_DESCRIPTION"].replace(to_replace=['Not Scored: More than 50 active Accounts found', 'Not Scored: No Activity seen on the customer (Inactive)','Not Scored: No Updates available in last 36 months','Not Enough Info available on the customer','Not Scored: Only a Guarantor','Not Scored: Sufficient History Not Available','Not Scored: Not Enough Info available on the customer'], value= 'Not Scored', inplace = True)

columns_to_drop = ['UniqueID','Date_of_Birth','MobileNo_Avl_Flag','DisbursalDate','AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH','SEC.OVERDUE.ACCTS','Aadhar_flag','PAN_flag','VoterID_flag','Driving_flag','Passport_flag']
df=df.drop(columns=columns_to_drop)
df = pd.get_dummies(df)
df.columns

df.dtypes.value_counts()
#Data Spliting 
X =df.drop('loan_default',axis=1)
y = df['loan_default']  

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#Logistic Regration
from sklearn.externals import joblib 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0, class_weight=None,fit_intercept=True,max_iter=200)
lr.fit(X_train, y_train)
print('Score  = ',lr.score(X_test, y_test)*100, '%')


joblib.dump(lr, 'GSFT_Loan.pkl')
