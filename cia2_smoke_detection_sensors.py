#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv(r"C:\Users\PRIYANKA\Downloads\smoke_detection_iot.csv")
data.info()


# In[5]:


# Convert the UTC column to a datetime format
data['UTC'] = pd.to_datetime(data['UTC'])

# Set the UTC column as the index
data.set_index('UTC', inplace=True)

# Resample the data to hourly frequency and fill missing values with the mean
data = data.resample('H').mean().fillna(data.mean())

# Split the data into training and testing sets (80% train, 20% test)
train_data = data[:int(len(data)*0.8)]
test_data = data[int(len(data)*0.8):]

# Fit an ARIMA model to the training data
model = ARIMA(train_data, order=(1,1,1))
model_fit = model.fit()

# Make predictions on the test data
predictions = model_fit.forecast(len(test_data))[0]

# Compute the mean squared error of the predictions
mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error:', mse)


# In[6]:


# Plot the actual and predicted values
plt.figure(figsize=(10,5))
plt.plot(train_data.index, train_data.values, label='Training Data')
plt.plot(test_data.index, test_data.values, label='Test Data')
plt.plot(test_data.index, predictions, label='Predictions')
plt.legend()
plt.show()


# In[ ]:




