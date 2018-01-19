
# coding: utf-8

# In[41]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
import sklearn.ensemble as ske
import tensorflow as tf
import tensorflow.contrib.learn as skflow


# In[12]:

titanic_df = pd.read_excel('titanic3.xls', 'titanic3', index_col=None, na_values=['NA'])


# In[13]:

titanic_df.head()


# In[14]:

titanic_df['survived'].mean()


# In[15]:

titanic_df.groupby('pclass').mean()


# In[16]:

class_sex_grouping = titanic_df.groupby(['pclass','sex']).mean()
class_sex_grouping


# In[17]:

class_sex_grouping['survived'].plot.bar()


# In[18]:

group_by_age = pd.cut(titanic_df["age"], np.arange(0, 90, 10))
age_grouping = titanic_df.groupby(group_by_age).mean()
age_grouping['survived'].plot.bar()


# In[19]:

titanic_df.count()


# In[21]:

titanic_df = titanic_df.drop(['body','cabin','boat'], axis=1)
titanic_df["home.dest"] = titanic_df["home.dest"].fillna("NA")


# In[22]:

titanic_df = titanic_df.dropna()


# In[23]:

titanic_df.count()


# In[24]:

def preprocess_titanic_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.sex = le.fit_transform(processed_df.sex)
    processed_df.embarked = le.fit_transform(processed_df.embarked)
    processed_df = processed_df.drop(['name','ticket','home.dest'],axis=1)
    return processed_df


# In[25]:

processed_df = preprocess_titanic_df(titanic_df)


# In[27]:

processed_df.count()
processed_df.head()


# In[28]:

X = processed_df.drop(['survived'], axis=1).values
y = processed_df['survived'].values


# In[29]:

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)


# In[30]:

clf_dt = tree.DecisionTreeClassifier(max_depth=10)


# In[31]:

clf_dt.fit (X_train, y_train)
clf_dt.score (X_test, y_test)


# In[32]:

shuffle_validator = cross_validation.ShuffleSplit(len(X), n_iter=20, test_size=0.2, random_state=0)
def test_classifier(clf):
    scores = cross_validation.cross_val_score(clf, X, y, cv=shuffle_validator)
    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))


# In[33]:

test_classifier(clf_dt)


# In[34]:

clf_rf = ske.RandomForestClassifier(n_estimators=50)
test_classifier(clf_rf)


# In[35]:

clf_gb = ske.GradientBoostingClassifier(n_estimators=50)
test_classifier(clf_gb)


# In[ ]:




# In[ ]:



