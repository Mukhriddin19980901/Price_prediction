#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
model = LinearRegression()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import seaborn as sb


# # 1-step. Malumotni yuklab olamiz va tekshiramiz

# In[2]:


data = pd.read_csv("Tash_uy_narx.csv")
data


# # Note : data ustunlarini bir birga bog'likligini Seaborn orqali tekshiramiz

# In[60]:


sb.pairplot(data[['rooms','level','size','max_levels','price']])
plt.show()


# In[3]:


data.info()


# In[4]:


data.describe()


# # 2-step. Malumotlarni normallashtiramiz

# In[262]:


data=data.drop('location',axis=1)


# In[288]:


# malumotlarni normallashtiramiz
data[data['size']>=100]


# In[264]:


data['level'].unique()


# In[265]:


data['rooms'].unique()


# In[7]:


data['district'].unique()


# In[267]:


# malumotlarni tozalash
data=data[data['price']!='Договорная']
data=data[data['price'].astype(float)<float('200000')]
data=data[data['size']!='Площадьземли:1сот']
data


# # Tayyorlangan malumotni yuklab olamiz

# In[268]:


data.to_csv('Toshkent_uylari_cost.csv',index=False)


# # 4-step.Filterlangan datani qayta yuklash

# In[2]:


data=pd.read_csv('Toshkent_uylari_cost.csv')
data


# # 5-step.Object yani string elementlarga qayta nom beramiz raqamli beramiz
#  > Encoding
#  > Labelling

# In[3]:


data_1=data
Lab=LabelEncoder()
data_1.district = Lab.fit_transform(data_1.district)
data_1


# In[4]:


def flt(arr):
    a=[]
    for i in arr:
        a.append(float(i))
    return a

# room
dp_s=['3', ' 2', ' 1',  '4',  '5',  '8',  '6',  '7', '10']
dp=flt(dp_s)
data_1['rooms'].replace(dp_s,dp_s,inplace=True)

# size
dp_s=np.array(data_1['max_levels'].unique())
dp=flt(dp_s)
data_1['max_levels'].replace(dp_s,dp_s,inplace=True)

# level
dp_s=np.array(data_1['level'].unique())
dp=flt(dp_s)
data_1['level'].replace(dp_s,dp_s,inplace=True)
data_1


# # 7-step.Vizualizatsiya 
# > matplotlib 
# 
# > Seaborn 

# In[45]:


data_new = data_1[(data_1['price']>=4000) & (data_1['size']<=225) & (data_1['size']>=10) & (data_1['level']>0)]
x=data_new['size'].values
y=data_new['price'].values
plt.plot(x,y,".")
plt.show()


# # 8-step.Malumotlarni X va Y ga ajratib o'qitamiz

# In[48]:


datas=data_new.drop('price',axis=1)
target=data_new['price']
datas=np.array(datas)
datas


# In[49]:


ct = ColumnTransformer([("district", OneHotEncoder(), [0])], remainder = 'passthrough')
datas = ct.fit_transform(datas)
datas


# In[50]:


x_train,x_test,y_train,y_test=train_test_split(datas,target,random_state=42)


# In[51]:


model.fit(x_train,y_train)


# # Bashorat qiladi va aniqligini tekshiramiz

# In[53]:


preds=model.predict(x_test)


# In[52]:


model.score(x_test,y_test)


# In[66]:


#
model.predict([[0,0,0,0,0,0,0,0,1,0,0,0,2,50,3,5]])


# In[54]:


pred=preds.reshape(-1)
pred


# In[55]:


y_test1=np.array(y_test)


# In[56]:


new_y=y_test1.reshape(-1)
new_y


# In[58]:


A=pd.DataFrame({
    'Actual' : new_y,
    'predictions' : preds
})
A

