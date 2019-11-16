def readData(filename):
	inputData=[]
	time=[]
	for x in open(filename, 'r'):
		line =x.strip().split(",")
		#if (line[0]!= "" and line[0]!="Horodatage" ): $
		if(line[1]==""): 
			inputData.append(np.nan)
			time.append(np.nan)
		else:
			inputData.append(float(line[1]))
			time.append(line[0])
	return(np.array((inputData),dtype=np.float64),time)


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps, n_steps_out,nlag):
	X, y = list(), list()
	for i in range(0,len(sequences),nlag):
		# find the end of this pattern
		end_ix = i + n_steps
		out_end_ix = end_ix + n_steps_out

		# check if we are beyond the dataset
		#if end_ix > len(sequences)-1:
		#	break
		if out_end_ix > len(sequences):
			continue
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		# gather input and output parts of the pattern
		#seq_x, seq_y = sequences[i:end_ix], sequences[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)



# split a multivariate sequence into samples
def split_sequences_rand(sequences, n_steps, n_steps_out,indices):
	X, y = list(), list()
	for i in indices:
		# find the end of this pattern
		end_ix = i + n_steps
		out_end_ix = end_ix + n_steps_out

		# check if we are beyond the dataset
		#if end_ix > len(sequences)-1:
		#	break
		if out_end_ix > len(sequences):
			continue
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# multistep multivariate multi-headed 1d cnn example

import numpy as np
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.metrics import mean_squared_error
import pandas as pd
from keras.models import model_from_json

np.random.seed(7)

#####Load Data
############ to modify
Debit, time = readData("dataset/clean_data/debitTsijiore.csv")
Debit=pd.DataFrame(Debit).dropna()
#print(Debit[0])
##add pluie 
Temp,time=readData("dataset/clean_data/arollaTemp.csv")

#time=time[  Temp != np.nan]

Temp=pd.DataFrame(Temp).dropna()


#Temp=Temp[Debit>0.1]
Temp=np.array(Temp)
#Debit=Debit[Debit>0.1]
Debit=np.array(Debit)

time=pd.DataFrame(time).dropna()


OneYear=14687

timeToTrain=4*OneYear
test=range(timeToTrain,5*OneYear)


#Dataset1 =data for training
Debit_train1 = Debit[0:timeToTrain].reshape((len(Debit[0:timeToTrain]), 1))
Temp_train1 = Temp[0:timeToTrain].reshape((len(Temp[0:timeToTrain]), 1))


dataset1 = hstack((Temp_train1, Debit_train1))
###takes into accout n_steps previous measurements
n_steps = 300

###
#predict nstepout next steps
nstepout=4
modelname="1hour"
## 
#select nb sample for training
nSamples= 2000
indices=(np.random.randint(0,timeToTrain,size=nSamples))
X, y = split_sequences_rand(dataset1, n_steps,nstepout,indices)


#dataset_Test =data for test (here is it the 4th year)
###process data for test

Debit_test = Debit[test].reshape((len(Debit[test]), 1))
Temp_test = Temp[test].reshape((len(Temp[test]), 1))
time_test=time.iloc[test]





#print(Temp[0])
#Debit_test=np.array(Debit_test)
dataset_Test = hstack((Temp_test, Debit_test))
# convert into input/output
X_test, y_test = split_sequences(dataset_Test, n_steps,nstepout,1)

print(X.shape, y.shape)
print(X_test.shape, y_test.shape)

##reshape for use in the model
n_output = y.shape[1]* y.shape[2]
y = y.reshape((y.shape[0], n_output))

n_output = y_test.shape[1]* y_test.shape[2]
y_test = y_test.reshape((y_test.shape[0], n_output))

n_features = X.shape[2]

##Create the model
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(n_output))
model_cnn.compile(optimizer='adam', loss='mse')

# fit_cnn model 
model_cnn.fit(X, y, epochs=300, verbose=0)


# demonstrate prediction

#predict for  training data
x_input = X

##evaluate on training
yhat = model_cnn.predict(x_input, verbose=0)
#print(yhat[0])
#print(y[0])

print("mse on training set is "+ str(mean_squared_error(y,yhat)))
##change shape yhat y  to extract flows
yhat = yhat.reshape(y.shape[0],nstepout, n_features)
y = y.reshape(y.shape[0],nstepout, n_features)
#loss=model_cnn.evaluate(yhat,y)

print("mse on training set (flow only) is "+ str(mean_squared_error(y[:,:,1],yhat[:,:,1])))
###store in file the flows predictions
fhnModel =open("TruePredicted_ontraining_"+modelname+".csv",'w')
fhnModel.write("timestep,real,predicted\n") 
for i in range(0,len(x_input)):
	for j in range(0,nstepout):
		fhnModel.write(str(j)+","+str(y[i,j,1]) +","+ str(yhat[i,j,1]) +'\n')
fhnModel.close()


#####predict on the test set
yhat = model_cnn.predict(X_test, verbose=0)
#loss=model_cnn.evaluate(yhat,y_test)
print("mse on test set is "+ str(mean_squared_error(y_test,yhat)))
##change shape yhat y_test to extract flows

yhat=yhat.reshape(yhat.shape[0],nstepout, n_features)
y_test=y_test.reshape(y_test.shape[0],nstepout, n_features)

print("mse on test set (flow only) is "+ str(mean_squared_error(y_test[:,:,1],yhat[:,:,1])))
###store in file the flows predictions
fhnModel =open("TruePredicted_ontest_"+modelname+".csv",'w')
fhnModel.write("time,timestep,real,predicted\n") 
for i in range(0,len(X_test)):
	for j in range(0,nstepout):

		timeFormat = str(time_test.iloc[i]).split( " ")[4] + "  "+ str(time_test.iloc[i]).split( " ")[5].split("\n")[0]
		fhnModel.write(timeFormat+","+str(j)+","+str(y_test[i,j,1])+ ","+ str(yhat[i,j,1]) +'\n')
fhnModel.close()

###Save ave Model 
# serialize model to JSON
model_json = model_cnn.to_json()
with open(modelname+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_cnn.save_weights(modelname+".h5")
print("Saved model to disk")


