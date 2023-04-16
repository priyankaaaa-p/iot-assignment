# iot-assignment
CIA 2
About the dataset
This dataset is available on Kaggle: https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset
Collection of training data is performed with the help of IOT devices since the goal is to develop a AI based smoke detector device. The dataset is nearly 60.000 readings long. The sample rate is 1Hz for all sensors. To keep track of the data, a UTC timestamp is added to every sensor reading.

Steps involved
In this code, the smoke_detection_iot dataset is loaded from a CSV file and the 'UTC' column is converted to a datetime format. 
Then, the 'UTC' column is set as the index and the data is resampled to an hourly frequency, filling any missing values with the mean. 
Then split the data into training and testing sets and fit an ARIMA model to the training data using an order of (1,1,1). 
Finally, make predictions on the test data, compute the mean squared error of the predictions, and plot the actual and predicted values.

Libraries used
pandas, numpy, matplotlib, statsmodels, sklearn.metrics
