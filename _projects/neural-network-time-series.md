---
title: "Neural Network for Time Series Prediction"
date: 2025-06-13
categories:
  - machine-learning
  - deep-learning
tags:
  - neural-networks
  - time-series
  - prediction
  - tensorflow
  - keras
header:
  image: "/images/KerasTensorflow.jpg"
  teaser: "/images/KerasTensorflow.jpg"
excerpt: "A project demonstrating how to build and train a neural network model for time series prediction using TensorFlow and Keras."
---

# Neural Network for Time Series Prediction

This project demonstrates how to build and train a neural network model for time series prediction using TensorFlow and Keras. We'll use a combination of LSTM (Long Short-Term Memory) and Dense layers to create a model that can predict future values based on historical data.

## Project Overview

Time series prediction is crucial in many applications, including:
- Stock price prediction
- Weather forecasting
- Energy consumption prediction
- Sales forecasting

In this project, we'll build a model that can predict future values of a time series based on past observations.

## Technologies Used

- Python 3.8
- TensorFlow 2.6
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Data Preparation

First, we need to prepare our data in a format suitable for time series prediction:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Load data
df = pd.read_csv('time_series_data.csv')
data = df['value'].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Set sequence length (lookback period)
sequence_length = 60

# Create sequences
X, y = create_sequences(scaled_data, sequence_length)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
```

## Model Building

Now, let's build our LSTM model:

```python
# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)
```

## Model Evaluation

Let's evaluate our model on the test data:

```python
# Make predictions
predicted = model.predict(X_test)

# Inverse transform to get actual values
predicted_actual = scaler.inverse_transform(predicted)
y_test_actual = scaler.inverse_transform(y_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_actual, predicted_actual))
print(f'Test RMSE: {rmse}')

# Plot the results
plt.figure(figsize=(16, 8))
plt.plot(y_test_actual, label='Actual')
plt.plot(predicted_actual, label='Predicted')
plt.title('Time Series Prediction')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(16, 8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
```

## Future Predictions

Now, let's predict future values:

```python
# Prepare the input for future predictions
last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)

# Number of future steps to predict
future_steps = 30

# Make predictions for future steps
future_predictions = []
current_sequence = last_sequence.copy()

for _ in range(future_steps):
    # Predict the next value
    next_pred = model.predict(current_sequence)[0]
    
    # Append to our predictions
    future_predictions.append(next_pred[0])
    
    # Update the sequence by removing the first value and adding the prediction
    current_sequence = np.append(current_sequence[:, 1:, :], 
                                 [[next_pred]], 
                                 axis=1)

# Convert to original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Plot with historical data
plt.figure(figsize=(16, 8))
plt.plot(data, label='Historical Data')
plt.plot(np.arange(len(data), len(data) + future_steps), future_predictions, label='Future Predictions')
plt.title('Time Series Prediction with Future Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
```

## Conclusion

In this project, we've demonstrated how to build and train a neural network model for time series prediction. The model can be further improved by:

1. Tuning hyperparameters (learning rate, batch size, number of layers, etc.)
2. Trying different architectures (GRU, CNN-LSTM, etc.)
3. Adding more features (multivariate time series)
4. Implementing attention mechanisms

Feel free to experiment with the code and adapt it to your specific time series prediction task!

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf)
- [Keras Documentation](https://keras.io/api/)
- [Time Series Forecasting Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
