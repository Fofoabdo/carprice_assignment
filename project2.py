#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().system('pip install plotly')
import plotly.express as px


# In[6]:


data=pd.read_csv('CarPrice_Assignment.csv')
data


# In[7]:


data.isna().sum()


# In[8]:


data.sample(10)


# In[9]:


data.describe()


# In[10]:


data['enginelocation'].value_counts()


# In[13]:


data['doornumber'].value_counts().plot.bar()


# In[14]:


carbody=data.groupby('carbody')


# In[15]:


for carbody ,carbody_data in carbody:
    print(carbody)
    print(carbody_data)


# In[16]:




heighst_price=data.sort_values(by=['price'],ascending=False)
heighst_price=heighst_price.head(10)
heighst_price['CarName']


# In[17]:


heighst_price=data.sort_values(by=['price'],ascending=False)
heighst_price=heighst_price.head(10)
heighst_price['enginelocation']


# In[18]:


data.duplicated().sum()


# In[20]:


data1=data.loc[data['fuelsystem']=='mpfi']
data1


# In[21]:


data1['price'].max()


# In[22]:


data1['price'].min()


# In[23]:


data1.describe()


# In[25]:


data2=data.loc[data['fuelsystem']=='idi']
data2


# In[26]:


data2.describe()


# In[28]:


data3=data.loc[data['fueltype']=='gas']
data3


# In[29]:


data['carbody'].value_counts().plot.bar()


# In[31]:


data.groupby('carbody').max()['price']


# In[32]:


data.groupby('drivewheel').max()['price']


# In[33]:


data['aspiration'].value_counts().plot.bar()


# In[34]:


data.groupby('aspiration').max()['price']


# In[37]:


data.groupby('CarName').max()['price'].nlargest(10)


# In[ ]:




