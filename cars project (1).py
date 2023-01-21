#!/usr/bin/env python
# coding: utf-8

# In[59]:


pip install eli5


# In[63]:


get_ipython().system('pip install missingno')


# In[65]:


import eli5
from eli5.sklearn import PermutationImportance
from collections import Counter
import missingno as msno

import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# In[115]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[116]:


data=pd.read_csv('CarPrice_Assignment.csv')
data


# data explore
# 

# In[3]:


data.columns


# In[4]:


data.info()


# In[5]:


data.describe().T


# In[6]:


data.isna().sum()/data.shape[0]*100


# In[7]:


data.shape


# data cleaning
# 

# In[10]:


#first we will splitting company name from car name 
#seconed we will insert company name 
#third we will drop column carname


company_name=data['CarName'].apply(lambda x :x.split(' ')[0])
data.insert(3,'company_name',company_name)
data.drop('CarName',axis=1,inplace=True)
data.head()


# In[45]:


data.columns


# In[11]:


data.company_name.unique()


# fixing invalid values
# 

# In[12]:


data.company_name=data.company_name.str.lower()

def replace_name(a,b):
    data.company_name.replace(a,b,inplace=True)

replace_name('maxda','mazda')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('Nissan','nissan')
replace_name('vw','volkswagen')


# In[13]:


data.company_name.unique()


# data visulization
# 

# In[14]:


#lets see the distribution of price
sns.displot(data.price)
plt.title('distribtion of cars price')


# In[16]:


print(data.price.describe(percentiles=[0.25,0.50,0.75,0.90,1]))


# In[36]:


data.company_name.value_counts().plot(kind='bar')
plt.title('compines histogram')
plt.xlabel('company')
plt.ylabel('count')
plt.show()

df=pd.DataFrame(data.groupby(['company_name'])['price'].mean().sort_values(ascending=False))
df.plot.bar()
plt.title('company_name vs average price')


# In[38]:


data.fueltype.value_counts().plot(kind='bar')
plt.title('fuel type')
plt.xlabel('fuel type')
plt.ylabel('count')
plt.show()

df=pd.DataFrame(data.groupby(['fueltype'])['price'].mean().sort_values(ascending=False))
df.plot.bar()
plt.title('fuel type vs average price')


# In[39]:


data.carbody.value_counts().plot(kind='bar')
plt.title('car body histogram')
plt.xlabel('car body')
plt.ylabel('count')
plt.show()

df=pd.DataFrame(data.groupby(['carbody'])['price'].mean().sort_values(ascending=False))
df.plot.bar()
plt.title('car body vs average price ')


# inference:
# 
# 1-toyota and nissan seemed to be favored cars company---
# 2-gas fuel type is more than diesel---
# 3-saden car body in the top of car body

# In[40]:


data.aspiration.value_counts().plot(kind='bar')
plt.title('aspiration histogram')
plt.xlabel('aspiration')
plt.ylabel('count')
plt.show()

df=pd.DataFrame(data.groupby(['aspiration'])['price'].mean().sort_values(ascending=False))
df.plot.bar()
plt.title('aspiration vs average price ')


# In[42]:


data.doornumber.value_counts().plot(kind='bar')
plt.title('doornumber histogram')
plt.xlabel('door number')
plt.ylabel('count')
plt.show()

df=pd.DataFrame(data.groupby(['doornumber'])['price'].mean().sort_values(ascending=False))
df.plot.bar()
plt.title('door number vs average price ')


# In[35]:


data.drivewheel.value_counts().plot(kind='bar')
plt.title('drive wheel histogram')
plt.xlabel('drive wheel')
plt.ylabel('count')
plt.show()

df=pd.DataFrame(data.groupby(['drivewheel'])['price'].mean().sort_values(ascending=False))
df.plot.bar()
plt.title('drive wheel vs average price')


# In[34]:


data.enginelocation.value_counts().plot(kind='bar')
plt.title('enginelocation histogram')
plt.xlabel('enginelocation')
plt.ylabel('count')
plt.show()

df=pd.DataFrame(data.groupby(['enginelocation'])['price'].mean().sort_values(ascending=False))
df.plot.bar()
plt.title('enginelocation vs average price')


# In[33]:


plt.subplot(1,2,1)
data.fuelsystem.value_counts().plot(kind='bar')
plt.title('fuel system histogram')
plt.xlabel('fuel system')
plt.ylabel('count')
plt.show()

df=pd.DataFrame(data.groupby(['fuelsystem'])['price'].mean().sort_values(ascending=False))
df.plot.bar(figsize=(6,6))
plt.title('fuel system vs average price')




# insights:
# 
# 1-mpfi fuel system is top fuel system and then there is 2bbi---
# 2-front enginelocation is vastly superior rear---
# 3-fwd drive wheel is top drive wheel and thn there is rwd---
# 4-in door number The proportions are somewhat similar but four door number is much---
# 5-aspiration std is top aspiration
#     

# In[ ]:





# lets visulize numericall data 
# 

# In[51]:


def scatter(x,fig):
    plt.subplot(2,5,fig)
    plt.scatter(data[x],data['price'])
    plt.title(x+' vs price')
    plt.xlabel(x)
    plt.ylabel('price')

plt.figure(figsize=(20,8))
scatter('carlength',1)
scatter('carwidth',2)
scatter('carheight',3)
scatter('curbweight',4)

plt.tight_layout()


# In[6]:


#def plot_count(x,fig):
    #plt.subplot(4,2,fig)
    #plt.title(x+'histogram')
    #sns.countplot(data[x],palette=('magma'))
    
    #plt.subplot(4,2,(fig+1))
    #plt.title(x+' vs price')
    #sns.boxplot(x=data[x],y=data['price'],palette=('magma'))
#plt.figure(figsize=(14,10))

#plot_count('enginelocation',1)
#plot_count('cylindernumber',3)
#plot_count('fuelsystem',5)
#plot_count('drivewheel',7)

#plt.tight_layout()


# In[8]:


def pp(x,y,z):
    sns.pairplot(data,x_vars=[x,y,z],y_vars='price',kind='scatter',size=4,aspect=1)
pp('enginesize','boreratio','stroke')
pp('compressionratio','horsepower','peakrpm')
pp('wheelbase','citympg','highwaympg')


# In[37]:


data.head()


# In[92]:


data1=['fueltype','aspiration','doornumber','carbody','drivewheel','fuelsystem','enginelocation']
while i < len(data1):
    print('Best {0} : {1}'.format(data1[i],data.loc[data[data1[i]].idxmax()][1]))
    i+= 1


# In[86]:


print('total number of car body  {0} :' .format(data['carbody'].nunique()))
print(data['carbody'].value_counts())


# In[106]:


data.loc[data['price']<=15000].head()


# In[120]:


data1=['fueltype','aspiration','doornumber','carbody','drivewheel','fuelsystem','enginelocation']
while i < len(data1):
    print('Best {0} : {1}'.format(data1[i],data.loc[data[data1[i]].idxmax()][1]))
    i+= 1


# In[122]:


df1=data.price
df2=data.drop(['car_ID','symboling','CarName','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','fuelsystem','enginetype','cylindernumber'],axis=1)


# In[123]:


x_train,x_test,y_train,y_test=train_test_split(df2,df1,test_size=0.4,random_state=100)
model=LinearRegression()
model.fit(x_train,y_train)


# In[124]:


prediction=model.predict(x_test)


# In[127]:


from sklearn import metrics


# In[140]:


print('MAE :',metrics.mean_absolute_error(y_test,prediction))
print('MSE :',metrics.mean_squared_error(y_test,prediction))
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# In[ ]:





# In[153]:


plt.scatter(prediction,y_test)
plt.plot(x_train,y_train)
plt.tight_layout()


# In[ ]:




