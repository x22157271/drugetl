#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pymongo import MongoClient
client = MongoClient("mongodb://%s:%s@127.0.0.1" % ("dap", "dap"))


# In[2]:


db = client['drugdb2']


# In[3]:


collection = db['drug_desc2']


# In[4]:


pip install requests pymongo


# In[5]:


import json
import requests



database_name = "drugdb2"
collection_name = "drugdb2"


# API from where data will be fetched
api_url = "https://healthdata.demo.socrata.com/resource/jaa8-k3k2.json"


client = MongoClient("mongodb://%s:%s@127.0.0.1" % ("dap", "dap"))
db = client[database_name]
collection = db[collection_name]

# Fetch data from the API
response = requests.get(api_url)

if response.status_code == 200:
    # Parse JSON data
    data = response.json()

    # Insert data into MongoDB
    collection.insert_many(data)




# In[ ]:




