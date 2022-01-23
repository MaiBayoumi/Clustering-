#!/usr/bin/env python
# coding: utf-8

# # install scikit-learn-extra

# In[3]:


get_ipython().system('pip install scikit-learn-extra')


# # import Libraries

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids


# # Read DataSet 

# In[5]:


Players= pd.read_csv('football_salaries.csv')


# In[6]:


Players.age.value_counts()


# # Delete incorrect Data , age = 0 ! , age = 2020 , age = 323 , player : 1 years old ?

# In[7]:


Players= Players[Players.age != 0]


# In[8]:


Players= Players[Players.age != 2020]
Players= Players[Players.age != 323]
Players= Players[Players.age != 1]


# In[9]:


Players


# # Divide avg_year into 1000000 to reduce the numbers

# In[10]:


Players['avg_year'] = Players['avg_year'] / 1000000


# In[11]:


Players = Players.reindex(columns=['position','player','team','age','avg_year','total_value','total_guaranteed','fully_guaranteed','free_agency'])


# In[12]:


Players


# # Determine which attribute are used in Clustering

# In[13]:


data = Players.iloc[:, 3:5].values


# # K-Medoids

# In[14]:


cluster = KMedoids(n_clusters=2, metric="manhattan",init="random", random_state=33)


# In[15]:


cluster.fit_predict(data)


# # Visualize result
# 

# In[16]:


plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.title('Clusters of Players')
plt.xlabel('age')
plt.ylabel('avg_year')


# # Visualize result to the part of data
# 

# In[17]:


Players1 = Players[0:97]


# In[18]:


Players1


# In[19]:


data = Players1.iloc[:, 3:5].values


# In[23]:


cluster = KMedoids(n_clusters=2, metric="manhattan",init="random", random_state=33)


# In[24]:


cluster.fit_predict(data)


# In[25]:


plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.title('Clusters of Players')
plt.xlabel('age')
plt.ylabel('avg_year')


# # Agglomerative Hierarchical Clustering Algorithm
# 

# In[26]:


from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


# In[27]:


data = Players.iloc[:, 3:5].values


# # Dendrogram to the part of data

# In[28]:


dendrogram= sch.dendrogram(sch.linkage(data[0:97],'single'))
plt.title('dendrogram')
plt.xlabel('X Values')
plt.ylabel('Distances')
plt.show()


# In[35]:


cluster = AgglomerativeClustering(n_clusters=2,affinity='euclidean', linkage='single')
cluster.fit_predict(data)


# # Visualize result 
# 

# In[36]:


plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.title('Clusters of Players')
plt.xlabel('age')
plt.ylabel('total_value')


# In[37]:


data = Players1.iloc[:, 3:5].values


# In[38]:


cluster = AgglomerativeClustering(n_clusters=4,affinity='euclidean', linkage='single')
cluster.fit_predict(data)


# # Visualize result to the part of data
# 

# In[39]:


plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.title('Clusters of Players')
plt.xlabel('age')
plt.ylabel('total_value')


# In[ ]:




