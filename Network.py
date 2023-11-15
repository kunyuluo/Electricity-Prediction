import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Model

# Import data
data = pd.read_csv('HistoricalData.csv')

# Format 'Date' column
data['Date'] = pd.to_datetime(data.Date)

# Format prices (Get rid of the dollor sign from the string)
data['Open'] = data['Open'].replace({'\$': ''}, regex=True).astype(float)
data['High'] = data['High'].replace({'\$': ''}, regex=True).astype(float)
data['Close/Last'] = data['Close/Last'].replace({'\$': ''}, regex=True).astype(float)
data['Low'] = data['Low'].replace({'\$': ''}, regex=True).astype(float)

data.rename(columns={' Close/Last': 'Close/Last', ' Volume': 'Volume', ' Open': 'Open', ' High': 'High', ' Low': 'Low'},
            inplace=True)

# print(data.head())

# Prepare train and test data (split)
data_to_train = data[:1000]
data_to_test = data[1000:]

# Create model
# Get 'Open' values as training and validating data
training_set = data_to_train.iloc[:, 3:4].values
real_stock_price = data_to_test.iloc[:, 3:4].values

# Normalizing data, scale between 0 and 1:
sc = MinMaxScaler(feature_range=(0, 1))
training_data_scaled = sc.fit_transform(training_set)

# format data into 3D array to input in LSTM Model
X_train = []
y_train = []
for i in range(60, 1000):
    X_train.append(training_data_scaled[i-60:i, 0])
    y_train.append(training_data_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# print(X_train.shape)
# print(y_train.shape)

# Building Model:
model = Model.create_model(X_train)
print(model.summary())

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

modelo = model.fit(X_train, y_train, epochs=100, batch_size=32)

dataset_total = pd.concat([data_to_train['Open'], data_to_test['Open']], axis=0)
inputs = dataset_total[len(dataset_total) - len(data_to_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 259):
    X_test.append(inputs[i - 60:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = model.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color='black', label='History Elec Usage')
plt.plot(predicted_stock_price, color='green', label='Predicted Elec Usage')
plt.title('Elec Usage Prediction')
plt.xlabel('Hour')
plt.ylabel('kWh')
plt.legend()
plt.show()
