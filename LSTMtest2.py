import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from keras.layers import Input



# Load the CSV file into a DataFrame
df = pd.read_csv('TestDatabase1.csv')  # Replace 'your_file.csv' with the path to your CSV file

# Select relevant columns and drop any missing values
columns = ["Date","Machine ID","Units Produced","Defects","Production Time Hours","Material Cost Per Unit","Labour Cost Per Hour","Energy Consumption kWh","Operator Count","Maintenance Hours","Down time Hours","Production Volume Cubic Meters","Scrap Rate","Rework Hours","Quality Checks Failed","Average Temperature C","Average Humidity Percent"]  # Add more features as needed
data = df[columns].dropna()

# Convert 'date' column to datetime and set as index (if applicable)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Scaling the Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Creating Training and Testing Sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Print the sizes of the datasets
print(f'Train data size: {len(train_data)}')
print(f'Test data size: {len(test_data)}')

# Create Datasets for LSTM
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), :])
        y.append(dataset[i + time_step, 0])  # Predicting the 'production' column
    return np.array(X), np.array(y)

# Choose a reasonable time_step based on the dataset size
time_step = min(10, len(train_data) // 2)  # Adjust the time_step to fit within the dataset size

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Check the shapes of X_train and X_test
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')

# Ensure X_test has the correct shape
if X_test.shape[0] == 0 or X_test.shape[1] == 0:
    raise ValueError("X_test is empty or not correctly formed. Please check the data and time_step.")

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Build the LSTM Model
model = Sequential()
model.add(Input(shape=(time_step, X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))  # Final layer for predicting 'production'

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the Model
model.fit(X_train, y_train, batch_size=1000, epochs=1000)  # Adjust batch_size and epochs as needed

# Make Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Create a DataFrame with same shape as scaled_data for inverse transform
train_predict_extended = np.zeros((len(train_predict), scaled_data.shape[1]))
test_predict_extended = np.zeros((len(test_predict), scaled_data.shape[1]))

# Place train and test predictions in the first column
train_predict_extended[:, 0] = train_predict.flatten()
test_predict_extended[:, 0] = test_predict.flatten()

# Inverse transform to get actual values
train_predict = scaler.inverse_transform(train_predict_extended)[:, 0]
test_predict = scaler.inverse_transform(test_predict_extended)[:, 0]

# Inverse transform the scaled_data for plotting actual data
actual_data = scaler.inverse_transform(scaled_data)[:, 0]

# Plotting
plt.figure(figsize=(8, 4))
plt.plot(actual_data, color='blue', label='Actual Data')
plt.plot(np.arange(time_step, len(train_predict) + time_step), train_predict, color='red', label='Train Prediction')
plt.plot(np.arange(len(train_predict) + (2 * time_step) + 1, len(train_predict) + (2 * time_step) + 1 + len(test_predict)), test_predict, color='green', label='Test Prediction')
plt.title('Prediction vs Actual')
plt.xlabel('Time')
plt.ylabel('Production')
plt.legend()
plt.show()
