#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from pymongo import MongoClient
client = MongoClient("mongodb://%s:%s@127.0.0.1" % ("dap", "dap"))


# In[13]:


db = client['drugcsv']


# In[14]:


collection = db['drugrating']


# In[15]:


database_name = "drugcsv"
collection_name = "drugrating"



client = MongoClient("mongodb://%s:%s@127.0.0.1" % ("dap", "dap"))
db = client[database_name]
collection = db[collection_name]
csv_file_path = 'Drug.csv'
df = pd.read_csv(csv_file_path)

# Convert DataFrame to a list of dictionaries
records = df.to_dict(orient='records')

# Insert data into MongoDB collection
collection.insert_many(records)

# Close the MongoDB connection
client.close()





# In[ ]:




