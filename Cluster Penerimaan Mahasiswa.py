#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the Relevant Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot #visualisasi
import seaborn as sns #mempercantik visualisasi
sns.set()
from sklearn.cluster import KMeans


# In[2]:


#Load the Data

data = pd.read_csv("D:\Data Science\ds_learning\ds_project\Dataset Penerimaan Mahasiswa.csv")
data


# In[13]:


import matplotlib.pyplot as plt


# In[16]:


#Plot the Data

plt.scatter(raw_data["SAT"],raw_data["Admitted"])
plt.xlim(1300,2100)
plt.ylim(-1,2)
plt.show


# In[17]:


x = raw_data.iloc[:,1:3]
x


# In[18]:


#clustering

kmeans =KMeans(2)
kmeans.fit(x)


# In[19]:


identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[20]:


data["Clusters"] = identified_clusters
data


# In[21]:


plt.scatter(data["SAT"],data["Admitted"],c=data['Clusters'],cmap='rainbow')
plt.xlim(1300,2100)
plt.ylim(-1,2)
plt.show


# In[22]:


kmeans.inertia_


# In[23]:


wcss = []

for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iteration = kmeans.inertia_
    wcss.append(wcss_iteration)


# In[24]:


wcss


# In[25]:


number_of_clusters = range(1,7)
plt.plot(number_of_clusters, wcss)
plt.xlabel("Number of clusters")
plt.ylabel("WCSS value")
plt.shw


# In[ ]:




