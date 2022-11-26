#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 


# In[3]:


data =pd.read_csv('merc.csv')
data


# In[37]:


data['end price']=data['price']-data['tax']
data


# In[40]:


sns.relplot(data.year,data['end price'])


# In[39]:


data.describe()


# In[41]:


result1=data.groupby('year').sum()['end price']
result1


# In[8]:


data['fuelType'].unique()


# In[12]:


data['fuelType'].value_counts()


# In[42]:


plt.bar(data.fuelType,data['end price'])
plt.title('fueltype vs price ')
plt.xlabel('fueltype')
plt.ylabel('price')
plt.show()


# In[23]:


tr=data['transmission'].value_counts()
tr


# In[43]:


plt.bar(data.transmission,data['end price'])
plt.xlabel('transmission')
plt.ylabel('price')
plt.title('transmission vs price')
plt.show()


# In[25]:


sns.relplot(data.mileage,data.price)


# In[34]:


#top 20 mileage (cars)

data1=data.sort_values(by='mileage',ascending=False)
data1=data1.head(20)
print(data1['model'])


# In[44]:


data


# In[45]:


data.groupby('transmission').sum()['tax']


# In[46]:


data.groupby('year').sum()['tax']


# In[48]:


sns.relplot(data.tax,data.price,kind='line')


# In[49]:


sns.relplot(data.mpg,data.mileage)


# In[51]:


sns.relplot(data['engineSize'],data.mileage)


# In[ ]:




