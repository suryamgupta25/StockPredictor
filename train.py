from datetime import datetime
import pandas as pd
import math
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense


if __name__ == "__main__":
    reader = open('apple_data.csv', 'r')
    data = pd.read_csv('apple_data.csv')

    dataset = data[['Date', 'Open', 'Close/Last']]
    # Convert date strings to Datetime
    for i, j in dataset.iterrows():
        j['Date'] = datetime.strptime(j['Date'], '%m/%d/%Y') # %Y is for 4-digit years, %y is for 2-digit years
    dataset = dataset.sort_values(by=['Date']).rename(columns={'Close/Last': 'Close'})
    dataset['Open'] = dataset['Open'].str[1:].astype(float).round(2)
    dataset['Close'] = dataset['Close'].str[1:].astype(float).round(2)

    num_rows_training = math.floor(0.8 * len(dataset))
    num_rows_testing = len(dataset) - num_rows_training
    train_set = dataset.head(num_rows_training)
    test_set = dataset.tail(num_rows_testing)


    # Normalize training data, where lowest price in the time period is closer to 0, and the highest is closer to 1
    train_set_scaled = train_set.copy()
    max_val = max(train_set['Open'].max(), train_set['Close'].max())
    min_val = min(train_set['Open'].min(), train_set['Close'].min())
    train_set_scaled['Open'] = (train_set_scaled['Open'] - min_val) / (max_val - min_val)
    train_set_scaled['Close'] = (train_set_scaled['Close'] - min_val) / (max_val - min_val)
    
    #train_set_scaled['Open'].plot(kind='line')
    #train_set_scaled['Close'].plot(kind='line')
    #plt.show()

    #print(train_set_scaled.head().to_numpy())
    X_train = []
    Y_train = []
    for i,j in train_set_scaled.iterrows():
        X_train.append([len(dataset) - i, j['Open']])
        Y_train.append(j['Close'])

    X_train = np.array(X_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    Y_train = np.array(Y_train)

    # Building model
    num_units = 75
    dropout_val = 0.1
    model = Sequential()
    model.add(LSTM(units=num_units,return_sequences=True,input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(dropout_val))
    model.add(LSTM(units=num_units,return_sequences=True))
    model.add(Dropout(dropout_val))
    model.add(LSTM(units=num_units,return_sequences=True))
    model.add(Dropout(dropout_val))
    model.add(LSTM(units=num_units))
    model.add(Dropout(dropout_val))
    model.add(Dense(units=1))
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(X_train,Y_train,epochs=100,batch_size=8)

    # Testing the model
    test_dataset = pd.concat((train_set['Open'], test_set['Open']), axis=0)
    inputs = []
    idx = 1

    # Normalize testing data
    max_test_val = test_dataset.max()
    min_test_val = test_dataset.min()
    for val in test_dataset:
        inputs.append([idx, (val - min_test_val) / (max_test_val - min_test_val)])
        idx += 1
    inputs = np.array(inputs)
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))

    testing_results = model(inputs)
    testing_results = testing_results.numpy()
    testing_results = np.array(testing_results)

    actual_values = dataset['Close'].tolist()
    final_predictions = []
    # Reverting back to original scale prices
    for val in testing_results:
        final_predictions.append(val * max_test_val)


    # Calculating Model Accuracy - does the prediction model correctly predict relative up/down patterns?
    totalAccurateTrends = 0
    for i in range(len(dataset) - 1):
        actualUp = actual_values[i + 1] - actual_values[i] >= 0
        predictUp = final_predictions[i+1] - final_predictions[i] >= 0
        if actualUp == predictUp:
            totalAccurateTrends += 1
    print(totalAccurateTrends)
    print("Trend Accuracy: {accuracy:.2f}%\n".format(accuracy=(totalAccurateTrends / (len(dataset) - 1) * 100)))
    
    # Displaying results
    x_axis = [i for i in range(1, len(dataset) + 1)]
    plt.plot(x_axis, actual_values, label='Actual Stock Price')
    plt.plot(final_predictions, label='Predicted Price')
    plt.title("Apple Stock Price Prediction")
    plt.xlabel("Days since August 7, 2023")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.show()