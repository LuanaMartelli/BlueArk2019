from datetime import datetime
import numpy as np
from keras.models import Sequential
from keras.models import Model

from keras.constraints import nonneg
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.metrics import mean_squared_error
from keras.layers.merge import concatenate

import pandas as pd


# from keras.models import model_from_json

# multistep multivariate multi-headed 1d cnn example

def readData(filename):
    inputData = []
    time = []
    for x in open(filename, 'r'):
        line = x.strip().split(",")
        # if (line[0]!= "" and line[0]!="Horodatage" ): $
        if (line[1] != ""):
            inputData.append(float(line[1]))
            cur_time = datetime.strptime(line[0], "%Y-%m-%d  %H:%M:%S")
            time.append(cur_time)
    inputData = np.array((inputData), dtype=np.float64)

    return (inputData, np.array(time))


# split a multivariate sequence into samples
def split_sequences_range(sequences, n_steps, n_steps_out, PredictTemp, indices):
    X, y = list(), list()
    # for i in range(0, len(sequences)):
    if PredictTemp:
        ind_output = range(0, 2)
    else:
        ind_output = 0
    for i in indices:

        # find the end of this pattern
        end_ix = i + n_steps
        out_end_ix = end_ix + n_steps_out

        # check if we are beyond the dataset
        # if end_ix > len(sequences)-1:
        #	break
        if out_end_ix > len(sequences):
            continue
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, ind_output]
        # gather input and output parts of the pattern
        # seq_x, seq_y = sequences[i:end_ix], sequences[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# split a multivariate sequence into samples
def split_sequences_rand(sequences, n_steps, n_steps_out, PredictTemp, indices=np.array([])):

    if indices.size == 0:
        indices = range(0, len(sequences))
    X, y = split_sequences_range(sequences, n_steps, n_steps_out, PredictTemp, indices)
    return X, y


def shape_train_data(Debit, Temp, cur_year, nSamples, n_steps, nstepout, PredictTemp):
    # indicesToTrain = np.where(np.logical_and(np.array([i.year for i in time]) == cur_year, np.array([i.month for i in time]) >5))[0].tolist()
    indicesToTrain = np.where(np.array([i.year for i in time]) == cur_year)[0].tolist()

    Debit_train1 = Debit[indicesToTrain].reshape((len(indicesToTrain), 1))
    Temp_train1 = Temp[indicesToTrain].reshape((len(indicesToTrain), 1))
    dataset1 = np.hstack((Debit_train1, Temp_train1))
    indices = (np.random.randint(0, len(indicesToTrain), size=nSamples))
    X, y = split_sequences_rand(dataset1, n_steps, nstepout, PredictTemp, indices)
    return X, y


np.random.seed(7)

###takes into accout n_steps previous measurements
n_steps = 300 #300
###
# predict nstepout next steps nstep=12 3 hours
nstepout = 96  # 12
modelname = "24hours_woTemp_sep"
##
# select nb sample for training
nSamples = 3000  #3500
PredictTemp = False
SeparateInputs = False

#####Load Data
Debit, time = readData("dataset/clean_data/debitTsijiore.csv")
Temp, time = readData("dataset/clean_data/arollaTemp.csv")
## Debit and Temp nparray of float, time nparray of datetime


## select the 4 first years to train  and the last year 2019 to test


##split sequences for the 4 years and then stack them.
nSampleperyear = int(nSamples / len(range(2015, 2019)))

X = np.array([])
y = np.array([])
for cur_year in range(2015, 2019):
    Xperyear, yperyear = shape_train_data(Debit, Temp, cur_year, nSampleperyear, n_steps, nstepout, PredictTemp)

    ##stack all the year to obtain only one Xy
    if X.size == 0:
        X = Xperyear
    else:
        X = np.vstack((X, Xperyear))
    if y.size == 0:
        y = yperyear
    else:
        y = np.vstack((y, yperyear))

# dataset_Test =data for test (here is it the 5th year)

print(X.shape, y.shape)

###process data for test
indicesToTest = np.where(np.array([i.year for i in time]) == 2019)[0].tolist()
# indicesToTest = np.where(np.logical_and(np.array([i.year for i in time]) == 2019, np.array([i.month for i in time]) > 5))[0].tolist()

Debit_test = Debit[indicesToTest].reshape((len(Debit[indicesToTest]), 1))
Temp_test = Temp[indicesToTest].reshape((len(Temp[indicesToTest]), 1))
time_test = time[indicesToTest].reshape((len(time[indicesToTest]), 1))

dataset_Test = np.hstack((Debit_test, Temp_test))
# convert into input/output
X_test, y_test = split_sequences_rand(dataset_Test, n_steps,  nstepout, PredictTemp)
print(X_test.shape, y_test.shape)
## extract time of y_test
dataset_timeTest = np.hstack((time_test, Debit_test))
X_timetest, y_timetest = split_sequences_rand(dataset_timeTest, n_steps, nstepout, PredictTemp)
if PredictTemp:
    y_timetest = y_timetest[:, :, 0]
else:
    y_timetest = y_timetest[:, :]

##reshape for use in the model
if PredictTemp:
    n_output = y.shape[1] * y.shape[2]
else:
    n_output = y.shape[1]
y = y.reshape((y.shape[0], n_output))
if PredictTemp:
    n_output = y_test.shape[1] * y_test.shape[2]
else:
    n_output = y_test.shape[1]
y_test = y_test.reshape((y_test.shape[0], n_output))


if SeparateInputs:
    n_features = 1
    # separate input data for training
    X1 = X[:, :, 0].reshape(X.shape[0], X.shape[1], n_features)
    X2 = X[:, :, 1].reshape(X.shape[0], X.shape[1], n_features)
    # first input model
    visible1 = Input(shape=(n_steps, n_features))
    cnn1 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible1)
    cnn1 = MaxPooling1D(pool_size=2)(cnn1)
    cnn1 = Flatten()(cnn1)
    # second input model
    visible2 = Input(shape=(n_steps, n_features))
    cnn2 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible2)
    cnn2 = MaxPooling1D(pool_size=2)(cnn2)
    cnn2 = Flatten()(cnn2)
    # merge input models
    merge = concatenate([cnn1, cnn2])
    dense = Dense(150, activation='relu')(merge)
    dense2 = Dense(100, activation='relu')(dense)
    output = Dense(n_output)(dense2)
    model_cnn = Model(inputs=[visible1, visible2], outputs=output)
    model_cnn.compile(optimizer='adam', loss='mse')
    # fit model
    model_cnn.fit([X1, X2], y, epochs=1000, verbose=0)
else:
    n_features = X.shape[2]

    ##Create the model
    model_cnn = Sequential()
    model_cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_steps, n_features)))
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(150, activation='relu'))
    model_cnn.add(Dense(100, activation='relu'))
    model_cnn.add(Dense(n_output))
    model_cnn.compile(optimizer='adam', loss='mse')

    # fit_cnn model
    model_cnn.fit(X, y, epochs=1000, verbose=0)

# demonstrate prediction

# predict for  training data
x_input = X

if SeparateInputs:
    ##evaluate on training
    yhat = model_cnn.predict([X1, X2], verbose=0)
else:
    ##evaluate on training
    yhat = model_cnn.predict(x_input, verbose=0)
# print(yhat[0])
# print(y[0])

print("mse on training set is " + str(mean_squared_error(y, yhat)))
if PredictTemp:
##change shape yhat y  to extract flows
    yhat = yhat.reshape(y.shape[0], nstepout, n_features)
    y = y.reshape(y.shape[0], nstepout, n_features)
# loss=model_cnn.evaluate(yhat,y)
    print("mse on training set (flow only) is " + str(mean_squared_error(y[:, :, 0], yhat[:, :, 0])))



###store in file the flows predictions
fhnModel = open("TruePredicted_ontraining_" + modelname + ".csv", 'w')
fhnModel.write("timestep,real,predicted\n")
for i in range(0, len(x_input)):
    for j in range(0, nstepout):
        if PredictTemp:
            fhnModel.write(str(j) + "," + str(y[i, j, 0]) + "," + str(yhat[i, j, 0]) + '\n')
        else:
            fhnModel.write(str(j) + "," + str(y[i, j]) + "," + str(yhat[i, j]) + '\n')

fhnModel.close()

#####predict on the test set
if SeparateInputs:
    x1 = X_test[:, :, 0].reshape((X_test.shape[0], X_test.shape[1], n_features))
    x2 = X_test[:, :, 1].reshape((X_test.shape[0], X_test.shape[1], n_features))
    ##evaluate on training
    yhat = model_cnn.predict([ x1, x2], verbose=0)
else:
    ##evaluate on training
    yhat = model_cnn.predict(X_test, verbose=0)



# loss=model_cnn.evaluate(yhat,y_test)
print("mse on test set is " + str(mean_squared_error(y_test, yhat)))
##change shape yhat y_test to extract flows

if PredictTemp:
    yhat = yhat.reshape(yhat.shape[0], nstepout, n_features)
    y_test = y_test.reshape(y_test.shape[0], nstepout, n_features)
    print("mse on test set (flow only) is " + str(mean_squared_error(y_test[:, :, 0], yhat[:, :, 0])))

###store in file the flows predictions
fhnModel = open("TruePredicted_ontest_" + modelname + "_step.csv", 'w')
fhnModel.write("time,timestep,real,predicted\n")
##find time for y_test
for i in range(0, len(y_test)):
    for j in range(0, nstepout):
        timeFormat = y_timetest[i, j].strftime("%Y-%m-%d %H:%M:%S")
        # timeFormat = str(time_test[i]).split(" ")[4] + "  " + str(time_test[i]).split(" ")[5].split("\n")[0]
        if PredictTemp:
            fhnModel.write(
            str(timeFormat) + "," + str(j + 1) + "," + str(y_test[i, j, 0]) + "," + str(yhat[i, j, 0]) + '\n')
        else:
            fhnModel.write(
                str(timeFormat) + "," + str(j + 1) + "," + str(y_test[i, j]) + "," + str(yhat[i, j]) +  '\n')
fhnModel.close()

###Save ave Model 
# serialize model to JSON
model_json = model_cnn.to_json()
with open(modelname + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_cnn.save_weights(modelname + ".h5")
print("Saved model to disk")
