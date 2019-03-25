import lstm1
import daccuracy
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt

def perf_stats(y_pred, y_test):

    #For normalised values
    forecast_errors = [y_test[i]-y_pred[i] for i in range(len(y_test))]

    ##Mean forecast error or bias
    mean_forecast_error = np.mean(forecast_errors)
    print('Bias: %s' % mean_forecast_error)

    ##RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    print('RMSE: %f' % rmse)

    # Making the Confusion Matrix
    #from sklearn.metrics import confusion_matrix
    #cm = confusion_matrix(y_test, y_pred)

    #For denormalised values
    y_test = lstm1.denormalise_windows(y_test, actual)
    y_pred = lstm1.denormalise_windows(y_pred, actual)

    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    # try:
    forecast_errors = [y_test[i] - y_pred[i] for i in range(len(y_test))]

    ##Mean forecast error or bias
    mean_forecast_error = np.mean(forecast_errors)
    print('Bias: %s' % mean_forecast_error)

    #except:
        ##RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    print('RMSE: %f' % rmse)


def plot_results(predicted_data, true_data):
    tick_spacing = 0.05
    fig = plt.figure(facecolor='white', figsize=(10,10))
    fig.set_size_inches
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.plot(true_data, label='True Data', color = 'blue')
    plt.plot(predicted_data, label='Prediction', color = 'green')
    #plt.xlim(0,505)
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white', figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data', color = 'blue')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction', color = 'magenta')

    plt.xlim(0,150)
    plt.show()


def grid_search():

    # Applying Grid Search to find the best model and the best parameters
    batch_size = [50, 100, 200, 500]
    #epochs =  [1, 5, 10]
    parameters = dict(batch_size=batch_size)
    m = KerasClassifier(build_fn = lstm1.build_model, layers = [1, window_size, 200, 1])
    grid_search = GridSearchCV(estimator = m,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           n_jobs = -1)

    grid_result = grid_search.fit(X_train, y_train)
    best_parameters = grid_result.best_params_
    print("Best params - ", best_parameters)


#Main Function
if __name__=='__main__':

    #declare global variables
    start_time = time.time()
    epochs  = 50
    window_size = 50
    prediction_length = 5

    seed = 7
    np.random.seed(seed)

    #higher the number, higher the space required
    batch = 256

    #Training and testing data
    X_train, y_train, X_test, y_test, actual = lstm1.get_data('bse_data.csv', window_size, True)

    # for TECHM stock data
    #X_train, y_train, X_test, y_test, actual = lstm1.get_data('techm.csv', window_size, True)
    print('>>> Data processed...')

    #for daily predictions
    model = lstm1.build_model([1, window_size, 200, 1])

    #if val loss isnt decreasing, early callback to stop training
    early_stopping = EarlyStopping(monitor='val_loss', patience = 3)

    #Fit model with batch size and epochs
    history = model.fit(X_train, y_train, batch_size=batch, nb_epoch=epochs, validation_split=0.1, callbacks=[early_stopping])

    #Metrics which are stored for each epoch
    #print(history.history.keys())

    #summarize history
    fig = plt.figure(facecolor='white', figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.plot(history.history['loss'], color= 'blue')
    plt.plot(history.history['val_loss'], color = 'red')
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    #Call predict function with window size and prediction length
    predictions = lstm1.predict_sequences_multiple(model, X_test, window_size, prediction_length)

    #Call point by point predict method
    y_pred = lstm1.predict_point_by_point(model, X_test)

    print('Training time : ', time.time() - start_time)

    #Plot multiple preds
    plot_results_multiple(predictions, y_test, prediction_length)
    #************************

    #Plot daily preds
    plot_results(y_pred, y_test)

    #Get Performance Stats
    perf_stats(y_pred, y_test)

    #Get directional accuracy of n days ahead prediction
    daccuracy.calc_directional_accuracy(y_test, predictions, prediction_length)
