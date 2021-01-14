#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Basics of Cluster Analysis


# In[6]:


#Import the Relevant Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot #visualisasi
import seaborn as sns #mempercantik visualisasi
sns.set()
from sklearn.cluster import KMeans


# In[7]:


#Load the Data

data = pd.read_csv("D:\Data Science\ds_learning\ds_project\Dataset negara.csv")
data


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


#Plot the Data

plt.scatter(data["Longitude"],data["Latitude"])
plt.xlim(-180,180) #xlim : x limit
plt.ylim(-90,90) #ylim : y limit
plt.show


# In[10]:


#Select the Features


# In[11]:


#iloc adalah metode untuk menseleksi variabel atau fitur berdasarkan posisi

data.iloc[1:2,0:2]


# In[12]:


data


# In[13]:


x = data.iloc[:,1:3]
x


# In[14]:


#clustering

kmeans =KMeans(2)
kmeans.fit(x)


# In[15]:


#Clustering results


# In[16]:


identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[17]:


data["Clusters"] = identified_clusters
data


# In[18]:


#Plot the Clusters

plt.scatter(data["Longitude"],data["Latitude"])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show


# In[19]:


plt.scatter(data["Longitude"],data["Latitude"],c=data['Clusters'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show


# In[20]:


#Tes 3 cluster


# In[21]:


kmeans =KMeans(3)
kmeans.fit(x)


# In[22]:


identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[23]:


data["Clusters"] = identified_clusters
data


# In[24]:


plt.scatter(data["Longitude"],data["Latitude"],c=data['Clusters'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show


# In[25]:


#pemetaan data languange

data = pd.read_csv("D:\Data Science\ds_learning\ds_project\Dataset negara.csv")
data["Language"] = data["Language"].map({"English":0, "French":1, "German":2})
data


# In[26]:


x = data.iloc[:,1:4]
x


# In[27]:


kmeans =KMeans(3)
kmeans.fit(x)


# In[28]:


identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[29]:


data["Clusters"] = identified_clusters
data


# In[30]:


plt.scatter(data["Longitude"],data["Latitude"],c=data['Clusters'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show


# In[31]:


#Selectiong the number of clusters

#WCSS


# In[32]:


kmeans.inertia_


# In[34]:


wcss = []

for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iteration = kmeans.inertia_
    wcss.append(wcss_iteration)


# In[35]:


wcss


# In[36]:


#The Elbow Method


# In[40]:


number_of_clusters = range(1,7)
plt.plot(number_of_clusters, wcss)
plt.xlabel("Number of clusters")
plt.ylabel("WCSS value")
plt.shw


# In[ ]:




