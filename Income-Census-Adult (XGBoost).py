#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv(r'F:\jps  trainers & Full stack datascience\Full Stack Data Science\ADULT DATASETS WITH NULL VALUES\adult.data',header=None)
test=pd.read_csv(r'F:\jps  trainers & Full stack datascience\Full Stack Data Science\ADULT DATASETS WITH NULL VALUES\adult.test',skiprows=1,header=None)

col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num','marital_status', 'occupation','relationship', 
              'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'wage_class']

train.columns=col_labels
test.columns=col_labels

#find out numerical and categorical features for train set
numerical_features=[feature for feature in train.columns if train[feature].dtype!='O']
categorical_features=[feature for feature in  train.columns if  train[feature].dtype=='O' and feature!='wage_class']

#Find out distinct values for each numerical feature
train[numerical_features].nunique()

#Find out distinct values for each categorical feature
train[categorical_features].nunique()

#Check for imbalanced target (In our case 76% are in class <=50K and 24% >50K)
train['wage_class'].value_counts('f') 

#another way to check for imbalanced target
train['wage_class'].value_counts()/(np.float(len(train)))

#Now let's do some graphs for train set in order to find key relationships !!!
ax=sns.countplot(train['wage_class'],hue=train['wage_class'],edgecolor='k',palette='Set2')
ax.set_xlabel('wage class')
ax.set_title('wage_class count')
plt.show()

#bx=sns.countplot(data=train,x=train['wage_class'],hue=train['wage_class'])
ax=plt.figure(figsize=(24,12))
ax=sns.countplot(train['workclass'],hue=train['wage_class'],edgecolor='k',palette='Set2')
ax.set_xlabel('workclass')
ax.set_title('workclass count')
plt.show()


#ax=plt.figure(figsize=(24,12))
ax=sns.countplot(train['education'],hue=train['wage_class'],edgecolor='k',palette='Set2')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('education')
ax.set_title('education / wage class')
plt.show()

ax=plt.figure(figsize=(22,14))
ax=sns.countplot(train['marital_status'],hue=train['wage_class'],edgecolor='k',palette='Set2')
ax.set_xlabel('marital status')
ax.set_title('marital status / wage class')
plt.show()

ax=plt.figure(figsize=(24,12))
ax=sns.countplot(train['occupation'],hue=train['wage_class'],edgecolor='k',palette='Set2')
ax.set_xlabel('occupation')
ax.set_title('occupation / wage class')
plt.show()


ax=plt.figure(figsize=(24,12))
ax=sns.countplot(train['relationship'],hue=train['wage_class'],edgecolor='k',palette='Set2')
ax.set_xlabel('relationship')
ax.set_title('relationship status / wage class')
plt.show()

ax=plt.figure(figsize=(24,12))
ax=sns.countplot(train['race'],hue=train['wage_class'],edgecolor='k',palette='Set2')
ax.set_xlabel('race')
ax.set_title('race  / wage class')
plt.show()

ax=plt.figure(figsize=(24,12))
ax=sns.countplot(train['sex'],hue=train['wage_class'],edgecolor='k',palette='Set2')
ax.set_xlabel('sex')
ax.set_title('sex  / wage class')
plt.show()

ax=plt.figure(figsize=(24,12))
ax=sns.countplot(train['native_country'],hue=train['wage_class'],edgecolor='k',palette='Set2')
ax.set_xlabel('native_country')
ax.set_title('native_country  / wage class')
plt.show()

ax=sns.countplot(train['occupation'], hue=train['wage_class'],edgecolor='k',palette='Set2')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Occupation / Wage Class')
ax.set_xlabel('Occupation')
plt.show()


ax=sns.distplot(train['age'],hist_kws=dict(edgecolor="k", linewidth=2),kde_kws={"color": "#ce0d55", "lw": 2})
ax.set_xlabel('Age')
ax.set_title('Age Distribution')
plt.show()

ax=sns.distplot(train['education_num'],hist_kws=dict(edgecolor="k", linewidth=2),kde_kws={"color": "#ce0d55", "lw": 2})
ax.set_xlabel('education_num')
ax.set_title('education_num Distribution')
plt.show()


ax=sns.distplot(train['capital_gain'],bins=10,kde=False,hist_kws=dict(edgecolor="k", linewidth=2))
ax.set_xlabel('capital_gain')
ax.set_title('capital_gain Distribution')
plt.show()


ax=sns.distplot(train['capital_loss'],bins=10,kde=False,hist_kws=dict(edgecolor="k", linewidth=2))
ax.set_xlabel('capital_loss')
ax.set_title('capital_loss Distribution')
plt.show()

ax=sns.distplot(train['hours_per_week'],bins=10,kde=False,hist_kws=dict(edgecolor="k", linewidth=2))
ax.set_xlabel('hours_per_week')
ax.set_title('hours_per_week Distribution')
plt.show()

ax=sns.distplot(train['fnlwgt'],bins=10,kde=False,hist_kws=dict(edgecolor="k", linewidth=2))
ax.set_xlabel('fnlwgt')
ax.set_title('fnlwgt Distribution')
plt.show()

#median age for each wage_class

ax=sns.barplot(train.groupby('wage_class')['age'].median().index,train.groupby('wage_class')['age'].median().values,edgecolor='k',palette='Set2')
ax.set_ylabel('Age')
ax.set_xlabel('wage class')
ax.set_title('Median age / wage class')
plt.show()

corr_train=train.copy()
for feature in categorical_features:
    corr_train.drop(feature,axis=1,inplace=True)

ax=sns.heatmap(corr_train.corr(),cmap='RdYlGn',annot=True)
ax.set_title('Correlation map')
plt.show()

#we do apply same process for test set also
tnumerical_features=[feature for feature in test.columns if test[feature].dtype!='O']
tcategorical_features=[feature for feature in test.columns if test[feature].dtype=='O' and feature!='wage_class']

#Check for imbalanced target (In our case approx 76% are in class <=50K and  approx 24% >50K)
test['wage_class'].value_counts('f')
#test['wage_class'].value_counts()/(np.float(len(test)))

#Now let's do some graphs for train set in order to find key relationships !!!
ax=sns.countplot(test['wage_class'],hue=test['wage_class'],edgecolor='k',palette='Set2')
ax.set_xlabel('wage class')
ax.set_title('wage_class count')
plt.show()

#bx=sns.countplot(data=train,x=train['wage_class'],hue=train['wage_class'])
ax=plt.figure(figsize=(24,12))
ax=sns.countplot(test['workclass'],hue=test['wage_class'],edgecolor='k',palette='Set2')
ax.set_xlabel('workclass')
ax.set_title('workclass count')
plt.show()


#ax=plt.figure(figsize=(24,12))
ax=sns.countplot(test['education'],hue=test['wage_class'],edgecolor='k',palette='Set2')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_xlabel('education')
ax.set_title('education / wage class')
plt.show()

ax=plt.figure(figsize=(22,14))
ax=sns.countplot(test['marital_status'],hue=test['wage_class'],edgecolor='k',palette='Set2')
ax.set_xlabel('marital status')
ax.set_title('marital status / wage class')
plt.show()

ax=plt.figure(figsize=(24,12))
ax=sns.countplot(test['occupation'],hue=test['wage_class'],edgecolor='k',palette='Set2')
ax.set_xlabel('occupation')
ax.set_title('occupation / wage class')
plt.show()


ax=plt.figure(figsize=(24,12))
ax=sns.countplot(test['relationship'],hue=test['wage_class'],edgecolor='k',palette='Set2')
ax.set_xlabel('relationship')
ax.set_title('relationship status / wage class')
plt.show()

ax=plt.figure(figsize=(24,12))
ax=sns.countplot(test['race'],hue=test['wage_class'],edgecolor='k',palette='Set2')
ax.set_xlabel('race')
ax.set_title('race  / wage class')
plt.show()

ax=plt.figure(figsize=(24,12))
ax=sns.countplot(test['sex'],hue=test['wage_class'],edgecolor='k',palette='Set2')
ax.set_xlabel('sex')
ax.set_title('sex  / wage class')
plt.show()

ax=plt.figure(figsize=(24,12))
ax=sns.countplot(test['native_country'],hue=test['wage_class'],edgecolor='k',palette='Set2')
ax.set_xlabel('native_country')
ax.set_title('native_country  / wage class')
plt.show()

ax=sns.countplot(test['occupation'], hue=test['wage_class'],edgecolor='k',palette='Set2')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Occupation / Wage Class')
ax.set_xlabel('Occupation')
plt.show()


ax=sns.distplot(test['age'],hist_kws=dict(edgecolor="k", linewidth=2),kde_kws={"color": "#ce0d55", "lw": 2})
ax.set_xlabel('Age')
ax.set_title('Age Distribution')
plt.show()

ax=sns.distplot(test['education_num'],hist_kws=dict(edgecolor="k", linewidth=2),kde_kws={"color": "#ce0d55", "lw": 2})
ax.set_xlabel('education_num')
ax.set_title('education_num Distribution')
plt.show()


ax=sns.distplot(test['capital_gain'],bins=10,kde=False,hist_kws=dict(edgecolor="k", linewidth=2))
ax.set_xlabel('capital_gain')
ax.set_title('capital_gain Distribution')
plt.show()


ax=sns.distplot(test['capital_loss'],bins=10,kde=False,hist_kws=dict(edgecolor="k", linewidth=2))
ax.set_xlabel('capital_loss')
ax.set_title('capital_loss Distribution')
plt.show()

ax=sns.distplot(test['hours_per_week'],bins=10,kde=False,hist_kws=dict(edgecolor="k", linewidth=2))
ax.set_xlabel('hours_per_week')
ax.set_title('hours_per_week Distribution')
plt.show()

ax=sns.distplot(test['fnlwgt'],bins=10,kde=False,hist_kws=dict(edgecolor="k", linewidth=2))
ax.set_xlabel('fnlwgt')
ax.set_title('fnlwgt Distribution')
plt.show()

#median age for each wage_class

ax=sns.barplot(test.groupby('wage_class')['age'].median().index,test.groupby('wage_class')['age'].median().values,edgecolor='k',palette='Set2')
ax.set_ylabel('Age')
ax.set_xlabel('wage class')
ax.set_title('Median age / wage class')
plt.show()

corr_test=test.copy()
for feature in categorical_features:
    corr_test.drop(feature,axis=1,inplace=True)

ax=sns.heatmap(corr_test.corr(),cmap='RdYlGn',annot=True)
ax.set_title('Correlation map')
plt.show()


#Feature Engineering
#convert <=50K and >50K to 0, 1 respectively
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
train['wage_class']=encoder.fit_transform(train['wage_class'])


#transform country feature to be 1 if country is the United States. Otherwise is equal to 0
train['native_country']=np.where(train['native_country']==' United-States',1,0)

#transform marital status and concatenate some classes to reduce distinct classes
train['marital_status']=train['marital_status'].replace({' Married-civ-spouse': 'Married', ' Never-married': 'Single',  
                                                        ' Separated':'Divorced', ' Married-spouse-absent' : 'Divorced', 
                                                         ' Divorced':'Divorced', 
                                                         ' Married-AF-spouse' :'Divorced', ' Widowed':'Widowed' })

#transform workclass feature to be 1 if the workclass is Private and 0 if doesn't
train['workclass']=np.where(train['workclass']==' Private',1,0)
#transform workclass feature to be 1 if the Sex is Male and 0 if doesn't
train['sex']=np.where(train['sex']==' Male',1,0)
#transform workclass feature to be 1 if the Race is White and 0 if doesn't
train['race']=np.where(train['race']==' White',1,0)
#create ordered label for education 
education_mapping={' Preschool':0,' 1st-4th':1,' 5th-6th':2,' 7th-8th':3,' 9th':4,' 10th':5,
                   ' 11th':6,' 12th':7,' HS-grad':8,' Some-college':0,' Assoc-acdm':10,
                   ' Assoc-voc':11, ' Bachelors':12, ' Prof-school':13, ' Masters':14,' Doctorate':15
                   }
train['education']=train['education'].map(education_mapping)


relationship_ordered=train.groupby(['relationship'])['wage_class'].count().sort_values().index
relationship_ordered={k:i for i,k in enumerate(relationship_ordered,0)}
train['relationship']=train['relationship'].map(relationship_ordered)  


occupation_ordered=train.groupby(['occupation'])['wage_class'].count().sort_values().index
occupation_ordered={k:i for i,k in enumerate(occupation_ordered,0)}
train['occupation']=train['occupation'].map(occupation_ordered)


marital_ordered=train.groupby(['marital_status'])['wage_class'].count().sort_values().index
marital_ordered={k:i for i,k in enumerate(marital_ordered,0)}
train['marital_status']=train['marital_status'].map(marital_ordered)

train.drop('fnlwgt',axis=1,inplace=True) # it is not a useful feature for predicting the wage class


#scaling the train set with StandardScaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_features_train=scaler.fit_transform(train.drop('wage_class',axis=1))
scaled_features_train=pd.DataFrame(scaled_features_train, columns=train.drop('wage_class',axis=1).columns)















#Test Set
#Now for the test set we have to apply all the transformations that we did before for the train set

test['wage_class']=np.where(test['wage_class']== ' >50K.',1,0)

test['wage_class'].value_counts()


#transform country feature to be 1 if country is the United States. Otherwise is equal to 0
test['native_country']=np.where(test['native_country']==' United-States',1,0)

#transform workclass feature to be 1 if the workclass is Private and 0 if doesn't
test['workclass']=np.where(test['workclass']==' Private',1,0)

#transform workclass feature to be 1 if the Sex is Male and 0 if doesn't
test['sex']=np.where(test['sex']==' Male',1,0)

test['race']=np.where(test['race']==' White',1,0)

test['education']=test['education'].map(education_mapping)

test['relationship']=test['relationship'].map(relationship_ordered) 

test['occupation']=test['occupation'].map(occupation_ordered)

#transform marital status and concatenate some classes to reduce distinct classes
test['marital_status']=test['marital_status'].replace({' Married-civ-spouse': 'Married', ' Never-married': 'Single',  
                                                        ' Separated':'Divorced', ' Married-spouse-absent' : 'Divorced', 
                                                         ' Divorced':'Divorced', 
                                                         ' Married-AF-spouse' :'Divorced', ' Widowed':'Widowed' })

test['marital_status']=test['marital_status'].map(marital_ordered)

test.drop('fnlwgt',axis=1,inplace=True)

scaled_features_test=scaler.transform(test.drop('wage_class',axis=1))
scaled_features_test=pd.DataFrame(scaled_features_test, columns=test.drop('wage_class',axis=1).columns)

final_test=pd.concat([scaled_features_test,test['wage_class']],axis=1)







