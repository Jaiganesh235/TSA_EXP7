## DEVELOPED BY: S JAIGANESH
## REGISTER NO: 212222240037
## DATE:

# Ex.No: 07                                       AUTO REGRESSIVE MODEL

## AIM:
To Implement an Auto Regressive Model using Python
## ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
## PROGRAM
```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Load the data from the CSV file
data = pd.read_csv('raw_sales.csv', index_col=0, parse_dates=True)

# Display the first few rows (GIVEN DATA)
print("GIVEN DATA:")
print(data.head())

# Perform Augmented Dickey-Fuller test for stationarity
result = adfuller(data['price'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Fit an AutoRegressive (AR) model with 13 lags
model = AutoReg(train['price'], lags=13)
model_fit = model.fit()

# Make predictions using the AR model
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(test['price'], predictions)
print('Mean Squared Error:', mse)

# Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
plt.figure(figsize=(10,6))
plt.subplot(211)
plot_pacf(train['price'], lags=13, ax=plt.gca())
plt.title("PACF - Partial Autocorrelation Function")
plt.subplot(212)
plot_acf(train['price'], lags=13, ax=plt.gca())
plt.title("ACF - Autocorrelation Function")
plt.tight_layout()
plt.show()

# PREDICTION
print("PREDICTION:")
print(predictions)

# Plot the test data and predictions (FINAL PREDICTION)
plt.figure(figsize=(10,6))
plt.plot(test.index, test['price'], label='Actual Price')
plt.plot(test.index, predictions, color='red', label='Predicted Price')
plt.title('Test Data vs Predictions (FINAL PREDICTION)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
```
## OUTPUT:

### GIVEN DATA
![Screenshot (2)](https://github.com/user-attachments/assets/9a77fc63-449a-4c7b-8860-caa0c8be3576)

### ADF-STATISTIC AND P-VALUE
![Screenshot (8)](https://github.com/user-attachments/assets/8cc34ce6-7140-4060-be93-cf7658f973ac)


### PACF - ACF
![Screenshot (3)](https://github.com/user-attachments/assets/b3edb3da-dee5-43eb-b9f8-f91894e8d5fc)

![Screenshot (4)](https://github.com/user-attachments/assets/c3b467a8-a573-4243-98a5-5a63d0f6c552)

### MSE VALUE
![Screenshot (7)](https://github.com/user-attachments/assets/c24290e7-d1a1-4bbe-806e-98fd33486c91)


### PREDICTION
![Screenshot (5)](https://github.com/user-attachments/assets/7732cbad-737a-4ff4-88ab-c29fd2318062)

### FINAL PREDICTION
![Screenshot (6)](https://github.com/user-attachments/assets/10d3db03-2359-4255-a3d5-c78f5ec4f90e)


### RESULT:
Thus, the program to implement the auto regression function using python is executed successfully.
