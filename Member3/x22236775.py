#!/usr/bin/env python
# coding: utf-8

# # Drug Analysis Using Multiple Linear Regression

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import datasets, linear_model, metrics
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan
import scipy.stats as stats
from scipy.stats import shapiro
import warnings
# Ignore unnecessary warnings during runtime
warnings.filterwarnings('ignore')


# Establishing PostgreSQL Connection

# In[2]:


#PostgreSQL connection details
db_params = {
    'host': '127.0.0.1',
    'port': '5432',
    'database': 'drug_data',
    'user': 'dap',
    'password': 'dap'
}
 
# Establish a connection to the PostgreSQL database
connection = psycopg2.connect(**db_params)
 
#SQL query
sql_query = "SELECT * FROM drug_data1;"
 
# Using pandas to read the SQL query result into a DataFrame
df = pd.read_sql_query(sql_query, connection)
 
# Close the database connection
connection.close()


# Loading Dataset

# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# Data Seperation

# In[6]:


#seperation of dependent and independent variable
X_drug=df.drop(columns=['satisfaction'])
Y_drug=df['satisfaction']


# In[7]:


X_drug


# Descriptive Statistics

# In[8]:


X_drug.describe()


# In[9]:


Y_drug


# In[10]:


Y_drug.describe()


# In[11]:


qualitative=[]
quantitative=[]
for column in X_drug.columns:
    if X_drug[column].dtype == 'object':
        qualitative.append(column)
    else:
        quantitative.append(column)


# In[12]:


qualitative


# In[13]:


quantitative


# ### Visualisation

# In[14]:


# Plot count distribution for each qualitative variable in X_drug
for column in qualitative:
    plt.figure(figsize=(12, 8))
    sns.countplot(x =column, data = X_drug, palette='Set3')
    #plt.xticks(rotation=90)
    plt.show()


# In[15]:


# Plot histograms for each quantitative variable in X_train
for column in quantitative:
    plt.figure(figsize=(8, 6))  
    plt.hist(X_drug[column], bins=30, edgecolor='black') 
    plt.title(f'Histogram for {column}')  
    plt.xlabel(column)  
    plt.ylabel('Frequency')
    plt.show()


# In[16]:


# Plot scatter plots for each quantitative variable against Sales_Price
for column in quantitative:
    plt.figure(figsize=(8, 8))
    plt.scatter(X_drug[column], Y_drug)
    plt.ylabel('Satisfaction')
    plt.xlabel(column)
    plt.show()


# In[17]:


# Distribution of satisfaction
plt.figure(figsize=(8, 6))
sns.histplot(Y_drug, bins=10, kde=True)
plt.title('Distribution of satisfaction')
plt.show()


# Transformation on Independent Variables

# In[18]:


# Using Label Encoding to convert categorical variables to numeric format
# Initialize the LabelEncoder
label_encoder = preprocessing.LabelEncoder() 
# Encode categorical variables in the training set (X_drug)
X_drug['conditions']= label_encoder.fit_transform(X_drug['conditions'])
X_drug['drug']= label_encoder.fit_transform(X_drug['drug'])
X_drug['indication']= label_encoder.fit_transform(X_drug['indication'])
X_drug['type']= label_encoder.fit_transform(X_drug['type'])


# Handling Outliers using Z-score

# In[19]:


# Removing outliers from the 'train' DataFrame using Z-scores
z_scores = np.abs(zscore(X_drug))
threshold = 3
df = X_drug[(z_scores < threshold).all(axis=1)]


# In[20]:


# Plot boxplots for each quantitative variable in X_train
for column in X_drug:
    plt.figure(figsize=(12, 8))
    plt.boxplot(X_drug)
    plt.title(column)
    plt.ylabel('Values')
    plt.tight_layout()
    plt.show()


# In[21]:


from sklearn.model_selection import train_test_split
# Importing the train_test_split function from scikit-learn
# Splitting the dataset into features (X) and target variable (Y)
X_train, X_test, Y_train, Y_test = train_test_split(X_drug, Y_drug, test_size=0.2, random_state=22236775)


# In[22]:


# Using Min-Max Scaling to normalize features in the training set (X_train) and test set (X_test)
# Initialize the MinMaxScaler
scaler = MinMaxScaler()
# Fit and transform the training features and update the DataFrame
scaled_features_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(scaled_features_train, columns=X_train.columns, index=X_train.index)
# Initialize a new MinMaxScaler for the test set
scaler = MinMaxScaler()
# Fit and transform the test features and update the DataFrame
scaled_features_test = scaler.fit_transform(X_test)
X_test = pd.DataFrame(scaled_features_test, columns=X_test.columns, index=X_test.index)


# Implementing Model

# In[23]:


# Performing Ordinary Least Squares (OLS) regression on the training set (X_train, Y_train)
X_train = sm.add_constant(X_train)
model = sm.OLS(Y_train, X_train).fit()
print(model.summary())


# In[24]:


X_train.drop(('conditions'),axis=1,inplace=True)


# In[25]:


X_train


# In[26]:


X_test.drop(('conditions'),axis=1,inplace=True)


# In[27]:


X_test


# In[28]:


# Performing Ordinary Least Squares (OLS) regression on the training set (X_train, Y_train)
X_train = sm.add_constant(X_train)
model = sm.OLS(Y_train, X_train).fit()
print(model.summary())


# In[29]:


# Adding a constant term to the features in the X_test
X_test = sm.add_constant(X_test)


# In[30]:


Y_pred=model.predict(X_test)
Y_pred


# Diagnostics

# In[31]:


# Scatter plot of actual vs predicted satisfaction
plt.figure(figsize=(8, 6))
sns.scatterplot(x=Y_test, y=Y_pred, alpha=0.7)
plt.xlabel('Actual satisfaction')
plt.ylabel('Predicted satisfaction')
plt.title('Actual vs Predicted satisfaction')
plt.grid(True)
plt.show()


# In[32]:


# Visualizing residuals vs fitted values for the training set
residuals_train = Y_test-Y_pred   # Calculate residuals by subtracting predicted values from actual values
# Create a scatter plot of residuals against fitted values
plt.figure(figsize=(10, 6))
plt.scatter(Y_pred, residuals_train, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('Residuals vs Fitted Values (Training Set)')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()


# In[33]:


# Visualizing standardized residuals vs fitted values
standardized_residuals = model.get_influence().resid_studentized_internal
# Create a scatter plot of standardized residuals against fitted values
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(model.fittedvalues, standardized_residuals, alpha=0.5)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_title('Standardized Residuals vs Fitted Values')
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Standardized Residuals')
plt.show()


# In[34]:


residuals = Y_test  - Y_pred
sm.qqplot(residuals, line='s',fit=True)
plt.title('Normal Q-Q Plot')
plt.show()


# In[35]:


# Performing the Breusch-Pagan test for heteroscedasticity
residuals = model.resid
_, p_value, _, _ = het_breuschpagan(residuals, X_train)
print(f'p-value for Breusch-Pagan test: {p_value}')
if p_value < 0.05:
    print('Reject the null hypothesis. There is evidence of heteroscedasticity.')
else:
    print('Fail to reject the null hypothesis. There is no evidence of heteroscedasticity.')


# In[36]:


# Calculating the Durbin-Watson statistic to test for autocorrelation in residuals
durbin_watson_statistic = sm.stats.stattools.durbin_watson(model.resid)
print(f'Durbin-Watson statistic: {durbin_watson_statistic}')
if durbin_watson_statistic < 1.5:
    print('Positive autocorrelation may be present.')
elif durbin_watson_statistic > 2.5:
    print('Negative autocorrelation may be present.')
else:
    print('No significant autocorrelation detected.')


# In[37]:


# Assuming 'residuals_train' is your residuals from the regression model
statistic, p_value = shapiro(residuals_train)
 
# Check if the residuals follow a normal distribution
if p_value > 0.05:
    print("The residuals likely follow a normal distribution.")
else:
    print("The residuals may not follow a normal distribution.")


# Evaluation

# In[38]:


# Calculating Residual Standard Error (RSE)
rse = round(np.sqrt(np.sum((Y_test - Y_pred)**2) / (len(Y_test) - 2)),2)
rse


# In[39]:


# Calculating Mean Absolute Error (MAE)
round(metrics.mean_absolute_error(Y_test,Y_pred),2)


# In[40]:


# Calculating Root Mean Squared Error (RMSE)
round(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)),2)


# In[41]:


# Calculating Mean Absolute Percentage Error (MAPE)
round(metrics.mean_absolute_percentage_error(Y_test,Y_pred),2)

