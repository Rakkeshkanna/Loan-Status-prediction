#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np


# In[41]:


data = pd.read_csv(r"C:\Users\rakke\Downloads\loan_approval_dataset.csv")


# In[42]:


data


# In[43]:


data.drop(columns=['loan_id'], inplace=True)


# In[44]:


data.columns


# In[45]:


data.columns = data.columns.str.strip()


# In[46]:


data.columns


# In[47]:


data['Assets'] =data.residential_assets_value + data.commercial_assets_value + data.luxury_assets_value + data.bank_asset_value


# In[48]:


data.drop(columns=['commercial_assets_value','residential_assets_value','luxury_assets_value','bank_asset_value'],inplace=True)


# In[49]:


#checking null values
data.isnull().sum()


# In[50]:


#checking distinct values
data.education.unique()


# In[51]:


#creating a function to remove extra spaces
def clean_data(st):
    st = st.strip()
    return st


# In[52]:


data.education = data.education.apply(clean_data)


# In[53]:


#now the extra spaces are removed
data.education.unique()


# In[54]:


# replacing char to numeric
data['education'] = data['education'].replace(['Graduate','Not Graduate'],[1,0])


# In[55]:


#checking distinct values
data.self_employed.unique()


# In[56]:


#applying the function
data.self_employed = data.self_employed.apply(clean_data)


# In[57]:


#now the extra spaces are removed
data.self_employed.unique()


# In[58]:


#replacing char to numeric
data['self_employed'] = data['self_employed'].replace(['No','Yes'],[0,1])


# In[59]:


#checking distinct values
data.loan_status.unique()


# In[60]:


#applying the function
data.loan_status = data.loan_status.apply(clean_data)


# In[61]:


#replacing char to numeric
data['loan_status'] = data['loan_status'].replace(['Approved','Rejected'],[1,0])


# In[64]:


#splitting train test split
from sklearn.model_selection import train_test_split


# In[69]:


#splitting target variable
X = data.drop(columns=["loan_status"])
y = data['loan_status']


# In[70]:


#splitting the train and test data 
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2)


# In[75]:


#scalling the data
from sklearn.preprocessing import StandardScaler


# In[76]:


scaler =  StandardScaler()


# In[77]:


X_train_scaled = scaler.fit_transform(X_train)


# In[78]:


X_test_scaled = scaler.transform(X_test)


# In[79]:


from sklearn.linear_model import LogisticRegression


# In[80]:


#creating the model
model = LogisticRegression()


# In[86]:


#traing the model
model.fit(X_train_scaled,y_train)


# In[87]:


# accuracy score check
model.score(X_test_scaled,y_test)


# In[98]:


import pickle as pk


# In[103]:


pk.dump(model,open('model.pkl','wb'))


# In[104]:


pk.dump(scaler, open('model.pkl', 'wb'))


# In[ ]:




