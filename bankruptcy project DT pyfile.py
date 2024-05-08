#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


data=pd.read_csv("Bankruptcy-prevention.csv",sep=';')
data.sample(10)


# In[3]:


data.info()


# In[4]:


#
data.isnull().sum()


# In[5]:


# Summary statistics
data.describe()


# In[6]:


data.count()


# In[7]:


data.columns


# In[8]:


#Count of duplicated rows
data[data.duplicated()].shape


# In[9]:


# Count of each class (bankruptcy and non-bankruptcy)
data[' class'].value_counts()


# In[10]:


plt.figure(figsize=(5,5))
sns.countplot(x=' class',data=data,palette='Set1')


# In[11]:


plt.figure(figsize=(15,12))
for i, predictor in enumerate(data.drop(columns=[' class'])):
    ax=plt.subplot(3,2,i+1)
    sns.countplot(data=data,x=predictor,hue=' class',palette='Set1')
    


# In[12]:


# Create a boxplot for each feature
plt.figure(figsize=(12, 8))
for i, feature in enumerate(data.columns[:-1]):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x=' class', y=feature, data=data,palette='rocket')
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)

plt.tight_layout()
plt.show()


# In[13]:


from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
data['class'] = encode.fit_transform(data[' class'])
data


# In[14]:


data.sample(10)


# In[15]:


data.drop(' class',axis=1,inplace=True)


# In[16]:


# Feature Relationships
correlation_matrix = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")


# In[17]:



sns.countplot(x = ' financial_flexibility', data = data, palette = 'plasma_r')


# In[18]:


sns.countplot(x = 'industrial_risk', data = data, palette = 'plasma_r')


# 

# In[19]:


sns.countplot(x = ' management_risk', data = data, palette = 'plasma_r')


# ## this shows that in most of the companies industrial risk and managment risk are high

# In[20]:



custom_palette = sns.color_palette("hot")
pd.crosstab(data['class'], data.industrial_risk).plot(kind='bar',color=custom_palette)


# In[21]:


data['class'].value_counts()


# In[22]:


a =data['class'].value_counts()[0]     
b =data['class'].value_counts()[1]   


fig1, ax1 = plt.subplots(figsize=(8, 6))
label = ['bankruptcy', 'non-bankruptcy']
count = [a, b]
colors = ['red', 'yellowgreen']
explode = (0, 0.1)  # explode 2nd slice
plt.pie(count, labels=label, autopct='%0.2f%%', explode=explode, colors=colors,shadow=True, startangle=90)
plt.show()


# In[23]:


data.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False,figsize=(12,10))
plt.show()


# In[24]:


#density plot
data.plot(kind='density', subplots=True, layout=(4,4), sharex=False, figsize=(12,10))
plt.show()


# In[ ]:





# In[25]:


sns.pairplot(data)


# ## Outlier Detection using Isolation Forest

# In[26]:


#The Isolation Forest algorithm is useful for identifying anomalies or outliers in a dataset by isolating them in the tree structure. 
from sklearn.ensemble import IsolationForest
clf = IsolationForest(random_state=10,contamination=.01)
clf.fit(data)
y_pred_outliers = clf.predict(data)
y_pred_outliers


# In[27]:


data['scores']=clf.decision_function(data)
data['anomaly']=clf.predict(data.iloc[:,0:7])
data


# In[28]:


# print the anomaly
data[data['anomaly']==-1]# we have outliers in our data so we drop that particular rows which has outlier that is -1 value


# In[29]:


data=data.drop(data.index[[27,72,192]],axis=0)
data


# In[30]:


## Spliting dataset into X and y
data=data.drop(['scores','anomaly'],axis=1)
data


# In[31]:


X = data.drop(['class'],axis=1)
y = data['class']


# In[32]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[33]:


# apply SelectKBest class to extract top 6 best features

bestfeatures = SelectKBest(score_func=chi2, k=6)
fit = bestfeatures.fit(X, y)


# In[34]:


featureScores_univ = pd.DataFrame({'variables':X.columns, 'Score':fit.scores_})
featureScores_univ.sort_values(by=['Score'], ascending=False)


# In[35]:


## High chi2 value suggest, feature is useful in predicting the class variable  


# In[36]:


from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import DecisionTreeClassifier
import matplotlib.pyplot as plt


# In[37]:


# use inbuilt class feature_importances of tree based classifiers
model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)

# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(6).plot(kind='barh', color='b')
plt.show()


# In[38]:


featureScores_dt = pd.DataFrame({'variables':X.columns, 'Score':model.feature_importances_})
featureScores_dt.sort_values(by=['Score'], ascending=False)


# In[39]:


data


# In[40]:


X


# In[41]:


y


# In[43]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=0)
model.fit(X, y)


# In[44]:


import pickle
# Saving the model to a file using pickle
with open("model1.pkl", "wb") as model_file:
    pickle.dump(model, model_file)


# In[ ]:




