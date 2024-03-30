# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Google stock prices are given in trainset.csv and testset.csv files.
![image](https://github.com/Abishai95141/rnn-stock-price-prediction/assets/139335314/c7e21ac8-46be-4df3-a43e-a29b594c578c)


## Design Steps

### Step 1:
import required packages

### Step 2:
read the csv files

### Step 3:
create a sequential model

### Step 4:
compile and fit the function

### Step 5:
Plot the graph using matplotlib


## Program
#### Name: ABISHAI
#### Register Number:212223240002
```

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
dataset_train = pd.read_csv('trainset.csv')
dataset_train.columns
dataset_train.head()
train_set = dataset_train.iloc[:,1:2].values
type(train_set)
train_set.shape
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape
X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_train.shape
length = 60
n_features = 1
length = 60
n_features = 1
model = Sequential()
model.add(layers.SimpleRNN(50,input_shape=(length,n_features)))
model.add(layers.Dense(1))

model.compile(optimizer='adam',loss='mse')
print("Name:ABISHAI,Register Number:212223240002")
model.summary()

model.fit(X_train1,y_train,epochs=100, batch_size=32)
dataset_test = pd.read_csv('testset.csv')
test_set = dataset_test.iloc[:,1:2].values
test_set.shape
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
X_test.shape
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```

## Output

### True Stock Price, Predicted Stock Price vs time

![image](https://github.com/Abishai95141/rnn-stock-price-prediction/assets/139335314/f8d627eb-2933-4873-8722-7116055d86d7)


### Mean Square Error

![image](https://github.com/Abishai95141/rnn-stock-price-prediction/assets/139335314/f22b1b30-2faf-4ea4-8cd8-b4b3eab81626)


## Result
output is generated successfully
