#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pymongo import MongoClient
client = MongoClient("mongodb://%s:%s@127.0.0.1" % ("dap", "dap"))


# In[2]:


db = client['drugdb']


# In[3]:


collection = db['drug_rating']


# In[4]:


import json


database_name = "drugdb"
collection_name = "drug_rating"



client = MongoClient("mongodb://%s:%s@127.0.0.1" % ("dap", "dap"))
db = client[database_name]
collection = db[collection_name]

# Load JSON data from file
with open('drug_review_test.jsonl', 'r') as f:
    for line in f:
        json_data = json.loads(line)
        # Insert JSON data into MongoDB collection
        collection.insert_one(json_data)
    





# In[ ]:




