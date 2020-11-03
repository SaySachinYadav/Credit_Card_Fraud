#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[37]:


data = pd.read_csv(r"E:/work photo/project data science/credit card fraud/creditcard.csv")


# In[38]:


data.head()


# In[39]:


data.tail()


# In[40]:


fraud = data.loc[data['Class'] == 1]
normal = data.loc[data['Class']== 0]


# In[41]:


data.count()


# In[42]:


data.sum()


# In[43]:


len(fraud)


# In[44]:


len(normal)


# In[45]:


sns.relplot(x= 'Amount',y='Time',hue='Class',data =data)


# In[47]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[48]:


x = data.iloc[:,:-1]
y = data['Class']


# In[49]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.35)


# In[51]:


clf = linear_model.LogisticRegression(C=1e5)


# In[52]:


clf.fit(x_train,y_train)


# In[54]:


y_pred = np.array(clf.predict(x_test))
y= np.array(y_test)


# In[55]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[59]:


print(confusion_matrix(y,y_pred))


# In[60]:


print(accuracy_score(y,y_pred))


# In[61]:


print(classification_report(y,y_pred))


# In[ ]:




