import tensorflow as tf
from keras.layers import Dense, Activation, Flatten, Conv1D, Dropout, MaxPooling1D, Convolution1D, UpSampling1D
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_features = 60001 
n_sensors = 10

###########
###model###
###########

model = Sequential()
model.add(Conv1D(filters = 8, kernel_size = 11, input_shape = (data_features, n_sensors)))
model.add(MaxPooling1D(pool_size = 2))
model.add(Convolution1D(filters = 16, kernel_size = 9))
model.add(MaxPooling1D(pool_size = 2))
model.add(Convolution1D(filters = 16, kernel_size = 7))
model.add(MaxPooling1D(pool_size = 2))
model.add(Convolution1D(filters = 32, kernel_size = 7))
model.add(MaxPooling1D(pool_size = 2))
model.add(Convolution1D(filters = 32, kernel_size = 5))
model.add(MaxPooling1D(pool_size = 2))
model.add(Convolution1D(filters = 64, kernel_size = 5))
model.add(MaxPooling1D(pool_size = 2))
model.add(Convolution1D(filters = 64, kernel_size = 3))
model.add(MaxPooling1D(pool_size = 2))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1, activation = 'sigmoid'))

###########
#optimizer#
###########

#optimizer = keras.optimizers.Adadelta(lr = 1.0, rho = 0.95, epsilon = None, decay = 1e-6)	
optimizer = optimizers.SGD(lr = 0.0001, momentum = 0.0, decay=0.0, nesterov=False)
#optimizer = keras.optimizers.Nadam(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, schedule_decay = 0.004)
#optimizer = keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
model.compile(loss = 'mse', optimizer = 'sgd', metrics = ['accuracy'])

#early stop
earlyStopping = EarlyStopping(monitor = 'loss', patience = 20, verbose = 1, mode = 'max')

# reduce learning rate when accuracy keep stable
reduce_lr_loss = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.1,patience = 7, verbose = 1, min_delta = 1e-4,mode = 'max')
model.summary()

#----load training data----

label = np.array(pd.read_csv('data/train.csv'))


for i in range(10):
    #initialize training dataset
    x = np.zeros((443, data_features, n_sensors))
    y_l = []

    #split dataset to 10 set, apply 2 epoch train on 1 set 
    count = 0
    for j in range(443):
        #read sensor readings and fill the missing reading with zero
        file = pd.read_csv('data/train/' + str(label[i*443 + j][0]) +'.csv').fillna(0)
        file_i = np.array(file)

        #Normalization
        nor = Normalizer()
        file_s = nor.fit_transform(file_i)
        
        x[count] = file_s
        y_l.append(label[i*443 + j][1])
        count+=1

    y = np.array(y_l)
    y_max = np.max(y)
    y_min = np.min(y)
    y = (y - y_min)/(y_max - y_min)

    h = model.fit(x, y, epochs = 2, batch_size = 10, verbose = 1, validation_split = 0.2)

model.save("cnn_model")




############
###evalue###
############
l_max = np.max(label)
l_min = np.min(label)

result = np.zeros(len(label))
y_t = np.column_stack((label[:,0], result.T))

for i in range(4431):
    #load test data
    file = pd.read_csv('data/train/' + str(int(y_t[i][0])) +'.csv').fillna(0)
    file_i = np.matrix(file)

    #standardization
    nor = Normalizer()
    x_test = nor.fit_transform(file_i)
    x_test = np.expand_dims(x_test, 0)
    #predict
    test_softmax_output = model.predict(x_test)		
    test_predictions = test_softmax_output[0][0]

    y_t[i][1] = test_predictions*(l_max - l_min) + y_min

print('MAE:')
print(mean_absolute_error(label[:,1], y_t[:,1]))



#############
###predict###
#############

y_p = np.array(pd.read_csv('data/sample_submission.csv'))

for i in range(4520):
    #load test data
    file = pd.read_csv('data/test/' + str(int(y_p[i][0])) +'.csv').fillna(0)
    file_i = np.matrix(file)

    #standardization
    nor = Normalizer()
    x_p = nor.fit_transform(file_i)
    x_p = np.expand_dims(x_p, 0)
    #predict
    p_softmax_output = model.predict(x_p)		
    p_predictions = p_softmax_output[0][0]

    y_p[i][1] = p_predictions*(l_max - l_min) + y_min

sub = pd.DataFrame(y_p, columns = ['segment_id', 'time_to_eruption'])
sub.to_csv('submission.csv')
