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

data = pd.read_csv("D:\Data Science\ds_learning\ds_project\Dataset Customer.csv")
data


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


#Plot the Data

plt.scatter(data["Satisfaction"],data["Loyalty"])
plt.xlim(-180,180) #xlim : x limit
plt.ylim(-90,90) #ylim : y limit
plt.show


# In[5]:


data.iloc[1:2,0:2]


# In[6]:


data


# In[8]:


#clustering

kmeans =KMeans(2)
kmeans.fit(data)


# In[9]:


identified_clusters = kmeans.fit_predict(data)
identified_clusters


# In[10]:


data["Clusters"] = identified_clusters
data


# In[25]:


plt.scatter(data["Satisfaction"],data["Loyalty"])
plt.xlim(0.8,11)
plt.ylim(-3,2)
plt.show


# In[26]:


plt.scatter(data["Satisfaction"],data["Loyalty"],c=data['Clusters'],cmap='rainbow')
plt.xlim(0.8,11)
plt.ylim(-3,2)
plt.show


# In[27]:


kmeans.inertia_


# In[28]:


wcss = []

for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iteration = kmeans.inertia_
    wcss.append(wcss_iteration)


# In[29]:


wcss


# In[30]:


number_of_clusters = range(1,7)
plt.plot(number_of_clusters, wcss)
plt.xlabel("Number of clusters")
plt.ylabel("WCSS value")
plt.shw


# In[36]:


#clustering

kmeans =KMeans(5)
kmeans.fit(data)


# In[38]:


identified_clusters = kmeans.fit_predict(data)
identified_clusters


# In[39]:


data["Clusters"] = identified_clusters
data


# In[40]:


plt.scatter(data["Satisfaction"],data["Loyalty"])
plt.xlim(0.8,11)
plt.ylim(-3,2)
plt.show


# In[41]:


plt.scatter(data["Satisfaction"],data["Loyalty"],c=data['Clusters'],cmap='rainbow')
plt.xlim(0.8,11)
plt.ylim(-3,2)
plt.show


# In[ ]:




