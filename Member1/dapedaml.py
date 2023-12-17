#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import psycopg2
import matplotlib.pyplot as plt
import pandas as pd
import pandas.io.sql as sqlio 
import seaborn as sns 
from sqlalchemy import create_engine, event, text, exc 
from sqlalchemy.engine.url import URL
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error#finding errors mse mae rmse
import numpy as np


# In[2]:


connection_string = "postgresql+psycopg2://dap:dap@127.0.0.1:5432/drug" 
try :
    engine = create_engine(connection_string) 
    with engine.connect() as connection:
        server_version = sqlio.read_sql_query( 
            text("SELECT VERSION();"), connection ) 
except exc.SQLAlchemyError as dbError: 
    print ("PostgreSQL Error", dbError) 
else:
    print(server_version["version"].values[0]) 
finally : 
    if engine in locals(): engine.close()


# In[3]:


sqlquery="Select * from drug_dimension;"
drug=pd.read_sql_query(sqlquery,connection_string)
drug


# In[4]:


drug.info()


# In[5]:


drug.describe()


# In[6]:


drug.shape


# In[7]:


drug.isnull().sum()


# In[8]:


drug.head()


# In[9]:


drug['patient_id'].groupby


# In[10]:


unique_ids_count = drug['patient_id'].nunique()
print(unique_ids_count)
#there are no repeated patient ids


# In[11]:


#seeing the distribution of rating in terms of their frequency
plt.hist(drug['rating'], bins=10, edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings')
plt.show()


# In[12]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(drug['rating'], bins=10, kde=False, color='skyblue', edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings')

plt.show()
#most of the rating are in 10 range


# In[13]:


#checking the corealation of features with rating
numeric_columns = drug.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = drug[numeric_columns].corr()
print(correlation_matrix)


# In[14]:


plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[15]:


drug['date'] = pd.to_datetime(drug['date'])
drug['year'] = drug['date'].dt.year

# Plotting the number of reviews over the years
plt.figure(figsize=(10, 6))
drug['year'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.title('Number of Reviews Over the Years')
plt.show()


# In[16]:


#finding top 10 drug 
top_drugs = drug['drugName'].value_counts().head(10)
#finding the top 10 medical conditions
top_conditions = drug['condition'].value_counts().head(10)


# In[17]:


top_drugs.plot(kind='bar', color='purple', edgecolor='black')
plt.xlabel('Drug Name')
plt.ylabel('Number of Reviews')
plt.title('Top 10 Reviewed Drugs')
plt.show()



# In[18]:


# Plotting top conditions
top_conditions.plot(kind='bar', color='green', edgecolor='black')
plt.xlabel('Condition')
plt.ylabel('Number of Reviews')
plt.title('Top 10 Reviewed Conditions')
plt.show()


# In[19]:


plt.hist(drug['review_length'], bins=20, edgecolor='black')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.title('Distribution of Review Lengths')
plt.show()


# In[20]:


plt.hist(drug['usefulcount'], bins=20, edgecolor='black')
plt.xlabel('Useful Count')
plt.ylabel('Frequency')
plt.title('Distribution of Useful Counts')
plt.show()


# In[21]:


drug['condition'].nunique()


# In[22]:


na_counts = drug.isna().sum()
print(na_counts)


# In[23]:


plt.figure(figsize=(10, 6))
na_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('Columns')
plt.ylabel('Number of NaN Values')
plt.title('Number of NaN Values in Each Column')
plt.show()


# In[24]:


top_conditions


# In[25]:


selected_condition = 'birth control'  
top_drugs_condition = drug[drug['condition'] == selected_condition]['drugName'].value_counts().head(10)

plt.figure(figsize=(12, 6))
top_drugs_condition.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.xlabel('Drug Name')
plt.ylabel('Number of Reviews')
plt.title(f'Top 10 Reviewed Drugs for {selected_condition}')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[26]:


selected_condition = 'depression'  
top_drugs_condition = drug[drug['condition'] == selected_condition]['drugName'].value_counts().head(10)

plt.figure(figsize=(12, 6))
top_drugs_condition.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.xlabel('Drug Name')
plt.ylabel('Number of Reviews')
plt.title(f'Top 10 Reviewed Drugs for {selected_condition}')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[27]:


selected_condition = 'acne'  
top_drugs_condition = drug[drug['condition'] == selected_condition]['drugName'].value_counts().head(10)

plt.figure(figsize=(12, 6))
top_drugs_condition.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.xlabel('Drug Name')
plt.ylabel('Number of Reviews')
plt.title(f'Top 10 Reviewed Drugs for {selected_condition}')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[28]:


selected_condition = 'pain'  
top_drugs_condition = drug[drug['condition'] == selected_condition]['drugName'].value_counts().head(10)

plt.figure(figsize=(12, 6))
top_drugs_condition.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.xlabel('Drug Name')
plt.ylabel('Number of Reviews')
plt.title(f'Top 10 Reviewed Drugs for {selected_condition}')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[29]:


top_drugs_info = drug.groupby('drugName').agg({'rating': 'mean', 'patient_id': 'count'})
top_drugs_info = top_drugs_info.rename(columns={'patient_id': 'num_reviews'})

top_drugs_info = top_drugs_info.sort_values(by='num_reviews', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_drugs_info.index, y='num_reviews', data=top_drugs_info, color='skyblue', label='Number of Reviews')
sns.lineplot(x=top_drugs_info.index, y='rating', data=top_drugs_info, color='orange', marker='o', label='Average Rating')

plt.xlabel('Drug Name')
plt.ylabel('Count / Rating')
plt.title('Top Drugs by Number of Reviews and Average Rating')
plt.xticks(rotation=45, ha='right')
plt.legend(loc='upper left', fontsize='small')
plt.show()


# In[30]:


top_drug_per_condition = drug.groupby('condition')['drugName'].agg(lambda x: x.mode().iloc[0])

top_drug_counts = top_drug_per_condition.value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_drug_counts.index, y=top_drug_counts.values, color='lightblue', edgecolor='black')
plt.xlabel('Drug Name')
plt.ylabel('Number of Occurrences as Top Medicine')
plt.title('Top Medicine for Each Condition')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[31]:


drug.head(10)


# In[32]:


#performing label encoding on categorical features
label_encoder = LabelEncoder()
drug['drugName'] = label_encoder.fit_transform(drug['drugName'])
drug['condition'] = label_encoder.fit_transform(drug['condition'])


# In[ ]:





# In[33]:


#spliting the date into separate day month and year column
drug['day'] = drug['date'].dt.day
drug['month'] = drug['date'].dt.month
drug['year'] = drug['date'].dt.year

drug = drug.drop('date', axis=1) 


# In[34]:


drug


# In[35]:


drug


# In[37]:


X = drug.drop(["rating","patient_id"], axis=1)
y = drug['rating']


# In[38]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=22157271)


# In[39]:


X_train.shape


# In[40]:


X_train.head()


# In[41]:


y_train.head()


# In[42]:


regression=LinearRegression()


# In[43]:


regression.fit(X_train,y_train)


# In[44]:


regression.coef_


# In[45]:


y_pred=regression.predict(X_test)


# In[46]:


mse=mean_squared_error(y_test,y_pred)
print(mse)
mae=mean_absolute_error(y_test,y_pred)
print(mae)
rmse=np.sqrt(mse)
print(rmse)


# In[ ]:




