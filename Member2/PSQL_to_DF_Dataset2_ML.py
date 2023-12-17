#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


# In[2]:


import pandas as pd
from sqlalchemy import create_engine, text

username = 'dap'
password = 'dap'
host = '127.0.0.1'  
database_name = 'drugs'


connection_string = f'postgresql://{username}:{password}@{host}/{database_name}'

engine = create_engine(connection_string)

connection = engine.connect()

query = text("SELECT * FROM drug1 LIMIT 1000")  

result = connection.execute(query)

data = result.fetchall()

connection.close()

columns = result.keys()  
df = pd.DataFrame(data, columns=columns)

print(df)

df.to_csv('data.csv', index=False)  


# In[15]:


df.info()


# In[17]:


df['drug_name'].unique()


# In[18]:


df['drug_tier'].unique()


# In[19]:


df['prior_authorization'].unique()


# In[11]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


print("Value Counts for 'drug_name':")
print(df['drug_name'].value_counts())


print("\nValue Counts for 'drug_tier':")
print(df['drug_tier'].value_counts())


print("\nValue Counts for 'prior_authorization':")
print(df['prior_authorization'].value_counts())






# In[8]:


# Plotting count distribution for 'drug_tier'
plt.figure(figsize=(8, 5))
sns.countplot(x='drug_tier', data=df, order=df['drug_tier'].value_counts().index)
plt.title('Count Distribution of Drug Tiers')
plt.xlabel('Drug Tier')
plt.ylabel('Count')
plt.show()


# In[9]:


# Plotting count distribution for 'prior_authorization'
plt.figure(figsize=(6, 4))
sns.countplot(x='prior_authorization', data=df)
plt.title('Count Distribution of Prior Authorization')
plt.xlabel('Prior Authorization')
plt.ylabel('Count')
plt.show()


# In[10]:


# Visualizing relationships between 'drug_tier' and 'prior_authorization'
plt.figure(figsize=(8, 5))
sns.countplot(x='drug_tier', hue='prior_authorization', data=df)
plt.title('Drug Tier vs. Prior Authorization')
plt.xlabel('Drug Tier')
plt.ylabel('Count')
plt.legend(title='Prior Authorization')
plt.show()


# In[4]:


columns_to_cluster = ['drug_name', 'drug_tier', 'prior_authorization']
data_for_clustering = df[columns_to_cluster]

encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(data_for_clustering)

kmeans = KMeans(n_clusters=5)
cluster_labels = kmeans.fit_predict(encoded_data)

df['cluster'] = cluster_labels

unique_clusters = df['cluster'].unique()

for cluster_label in unique_clusters:
    cluster_records = df[df['cluster'] == cluster_label]
    
    print(f"\nCluster {cluster_label} Records:")
    print(cluster_records)
    
    cluster_records.to_csv(f'cluster_{cluster_label}.csv', index=False)

