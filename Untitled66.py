#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import math
import matplotlib.pyplot as plt


# In[3]:


data=pd.read_csv('CarPrice_Assignment.csv')
#data


# In[4]:


#copy data
#drop columns

data_copy=data.copy()
data_copy=data_copy.drop(['highwaympg','citympg'],axis=1)


# In[5]:


data_copy


# In[6]:


data_copy.isna().sum()


# In[11]:


data1=data.sort_values(by=['price'],ascending=False)
data1=data.head(10)
data1['CarName']


# In[12]:


data['cylindernumber'].value_counts()


# In[15]:


data.hist(rwidth=9)
plt.tight_layout()


# In[32]:


plt.subplot(2,2,1)
plt.title('length vs price')
plt.scatter(data.carlength,data.price,s=6,c='r')

plt.subplot(2,2,2)
plt.title('citympg vs price')
plt.scatter(data.citympg,data.price,s=6,c='g')

plt.subplot(2,2,3)
plt.title('horse power vs price')
plt.scatter(data.horsepower,data.price,s=6,c='y')

plt.subplot(2,2,4)
plt.title('peakrpm vs price ')
plt.scatter(data.peakrpm,data.price,s=6)
 
plt.tight_layout()    


# In[59]:


plt.subplot(3,3,1)
price =data['price'].unique()
cylindernumber=data['cylindernumber']
plt.title('cylindernumber vs price')
plt.xlabel('cylindernumber')
plt.ylabel('price')
plt.bar(cylindernumber,price)


plt.subplot(3,3,2)
price=data['price'].unique()
fuelsystem=data['fuelsystem']
plt.title('fuelsystem vs price')
plt.xlabel('fuelsystem')
plt.ylabel('price')
plt.bar(fuelsystem,price)
plt.show()


# In[ ]:




