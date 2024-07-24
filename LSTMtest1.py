import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

df = pd.read_csv('monthly_milk_production.csv')

data = df[['Production']].dropna()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 5
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')

if X_test.shape[0] == 0 or X_test.shape[1] == 0:
    raise ValueError("X_test is empty or not correctly formed. Please check the data and time_step.")

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, batch_size=10, epochs=10)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

plt.figure(figsize=(16, 8))
plt.plot(scaler.inverse_transform(scaled_data), color='blue', label='Actual Data')
plt.plot(np.arange(time_step, len(train_predict) + time_step), train_predict, color='red', label='Train Prediction')
plt.plot(np.arange(len(train_predict) + (2 * time_step) + 1, len(train_predict) + (2 * time_step) + 1 + len(test_predict)), test_predict, color='green', label='Test Prediction')
plt.title('Prediction vs Actual')
plt.xlabel('Time')
plt.ylabel('Production')
plt.legend()
plt.show()
