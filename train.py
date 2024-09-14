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

from matplotlib.backend_bases import MouseButton

trained_models = {}

def train_stock(name:str):
    data = open(name + ".csv", 'r')
    pass

def test_stock(name:str, dataset):
    pass

"""
def on_move(event):
    if event.inaxes:
        print(f'data coords {event.xdata} {event.ydata},',
"""

if __name__ == "__main__":
    data = pd.read_csv('AAPL.csv')

    dataset = data[['Date', 'Open', 'Close/Last', 'Volume']]
    # Convert date strings to Datetime
    for i, j in dataset.iterrows():
        j['Date'] = datetime.strptime(j['Date'], '%m/%d/%Y') # %Y is for 4-digit years, %y is for 2-digit years
    dataset = dataset.sort_values(by=['Date']).rename(columns={'Close/Last': 'Close'})
    dataset['Open'] = dataset['Open'].str[1:].astype(float).round(2)
    dataset['Close'] = dataset['Close'].str[1:].astype(float).round(2)
    dataset['Volume'] = dataset['Volume'].astype(float)

    num_rows_training = math.floor(0.8 * len(dataset))
    num_rows_testing = len(dataset) - num_rows_training
    train_set = dataset.head(num_rows_training)
    test_set = dataset.tail(num_rows_testing)


    # Normalize training data, where lowest price in the time period is closer to 0, and the highest is closer to 1
    train_set_scaled = train_set.copy()
    max_val = max(train_set['Open'].max(), train_set['Close'].max())
    min_val = min(train_set['Open'].min(), train_set['Close'].min())
    max_volume = train_set['Volume'].max()
    min_volume = train_set['Volume'].min()
    train_set_scaled['Open'] = (train_set_scaled['Open'] - min_val) / (max_val - min_val)
    train_set_scaled['Close'] = (train_set_scaled['Close'] - min_val) / (max_val - min_val)
    train_set_scaled['Volume'] = (train_set_scaled['Volume'] - min_volume) / (max_volume - min_volume)
    

    print(train_set_scaled.head().to_numpy())
    X_train = []
    Y_train = []
    for i,j in train_set_scaled.iterrows():
        X_train.append([j['Open'], j['Volume']])
        Y_train.append(j['Close'])

    X_train = np.array(X_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    Y_train = np.array(Y_train)
    
    
    if 'AAPL' not in trained_models:
        # Building model
        unit_size = 120
        dropout_val = 0.10
        model = Sequential()
        model.add(LSTM(units=unit_size,return_sequences=True,input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(dropout_val))
        model.add(LSTM(units=unit_size,return_sequences=True))
        model.add(Dropout(dropout_val))
        model.add(LSTM(units=unit_size,return_sequences=True))
        model.add(Dropout(dropout_val))
        model.add(LSTM(units=unit_size, return_sequences=True))
        model.add(Dropout(dropout_val))
        model.add(LSTM(units=unit_size))
        model.add(Dropout(dropout_val))
        model.add(Dense(units=1))
        model.compile(optimizer='adam',loss='mean_absolute_error')
        model.fit(X_train,Y_train,epochs=100,batch_size=8)
        trained_models['AAPL'] = model
    
    trained_model = trained_models['AAPL']

    # Testing the model
    test_dataset = pd.concat((train_set, test_set), axis=0)
    test_dataset.drop(columns=['Date', 'Close'])
    inputs = []


    # Normalize testing data
    max_test_val = test_dataset['Open'].max()
    min_test_val = test_dataset['Open'].min()
    max_test_volume = test_dataset['Volume'].max()
    min_test_volume = test_dataset['Volume'].min()
    for i, val in test_dataset.iterrows():
        inputs.append([(val['Open'] - min_test_val) / (max_test_val - min_test_val), (val['Volume'] - min_test_volume) / (max_test_volume - min_test_volume)])
    inputs = np.array(inputs)
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))

    testing_results = trained_model(inputs)
    testing_results = testing_results.numpy()
    testing_results = np.array(testing_results)
    print(testing_results)

    actual_values = dataset['Close'].tolist()
    final_predictions = []
    max_close = dataset['Close'].max()
    min_close = dataset['Close'].min()
    # Reverting back to original scale prices
    for val in testing_results:
        final_predictions.append(val * (max_close - min_close) + min_close)


    # Calculating Model Accuracy - does the prediction model correctly predict relative up/down patterns?
    totalAccurateTrends = 0
    for i in range(len(dataset) - 1):
        actualUp = (actual_values[i + 1] - actual_values[i]) > 0
        predictUp = (final_predictions[i+1] - final_predictions[i]) > 0
        if actualUp == predictUp:
            totalAccurateTrends += 1
    print("Trend Accuracy (all data): {accuracy:.2f}%\n".format(accuracy=(totalAccurateTrends / (len(dataset) - 1) * 100)))

    # Calculating Trend Accuracy in only the testing data (as the model was tested with all data points)
    testingAccurateTrends = 0
    for i in range(num_rows_training, len(dataset) - 1, 1):
        actualUp = (actual_values[i + 1] - actual_values[i]) > 0
        predictUp = (final_predictions[i+1] - final_predictions[i]) > 0
        if actualUp == predictUp:
            testingAccurateTrends += 1
    print("Testing Accuracy: {accuracy:.2f}%\n".format(accuracy=(testingAccurateTrends / (len(dataset) - num_rows_training - 1) * 100)))

    # Calculating Ending Price Accuracy
    actualFinal = actual_values[len(actual_values) - 1]
    predictedFinal = final_predictions[len(final_predictions) - 1][0]
    print("Final Price Accuracy: {accuracy:.2f}%\n".format(accuracy=(100 - (abs(predictedFinal - actualFinal)) / actualFinal * 100)))
    
    # Displaying results
    x_axis = [i for i in range(1, len(dataset) + 1)]
    plt.plot(x_axis, actual_values, label='Actual Stock Price')
    plt.plot(final_predictions, label='Predicted Price')
    plt.title("Apple Stock Price Prediction")
    plt.xlabel("Days since {date:s}".format(date=dataset['Date'][len(dataset) - 1]))
    plt.ylabel("Stock Price (USD)")
    plt.legend()

    #binding_id = plt.connect('motion_notify_event', on_move)

    plt.show()