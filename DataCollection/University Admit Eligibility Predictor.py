#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('Admission_predict.csv')


# In[3]:


data.head()


# In[4]:


data.drop(["Serial No."],axis=1,inplace=True)


# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


data.info()


# In[8]:


data.isnull().any()


# In[9]:


data.isnull().sum()


# In[20]:


df = pd.DataFrame(data)


# In[21]:


df.columns = df.columns.str.replace(' ', '_')


# In[22]:


print("\n\n", df)


# In[23]:


sns.relplot(x='CGPA',y='Chance_of_Admit_',data=data)


# In[24]:


sns.relplot(x='CGPA',y='Chance_of_Admit_',hue='TOEFL_Score',data=data)


# In[25]:


sns.relplot(x='CGPA',y='Chance_of_Admit_',data=data,kind="line")


# In[29]:


sns.countplot(x="University_Rating",data=data)


# In[31]:


b=sns.FacetGrid(data,col="University_Rating")
b.map(plt.hist,"CGPA")


# In[33]:


b=sns.PairGrid(data)
b.map(plt.scatter)


# In[34]:


sns.set(style="darkgrid")
b=sns.FacetGrid(data,col="University_Rating")
b.map(plt.hist,"CGPA")


# In[36]:


sns.set(style="white",color_codes=True)
sns.boxplot(x='University_Rating',y='Chance_of_Admit_',data=data)


# In[56]:


independent = data.iloc[:,0:7].values
dependent = data.iloc[:,7:].values


# In[57]:


independent.shape


# In[58]:


dependent.shape


# In[59]:


from sklearn.model_selection import train_test_split


# In[61]:


X_train, X_test, y_train, y_test = train_test_split(independent, dependent, random_state=0, train_size = .2)


# In[62]:


X_train


# In[63]:


X_test


# In[64]:


y_train


# In[65]:


y_test


# In[ ]:




