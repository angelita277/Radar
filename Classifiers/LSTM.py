"""
Bidirectional LSTMS on VOXELS

Usage:

- extract_path is the where the extracted data samples are available.
- checkpoint_model_path is the path where to checkpoint the trained models during the training process


EXAMPLE: SPECIFICATION

extract_path = '/Users/lianqi/Angelita/MyResearch/radar/RadHAR-master/Data/extract/Train_Data_voxels_'
checkpoint_model_path="/Users/lianqi/Angelita/MyResearch/radar/RadHAR-master/model/LSTM"
"""
import tensorflow
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Bidirectional, TimeDistributed
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.layers.convolutional import *
from keras.layers import BatchNormalization
from keras.metrics import categorical_crossentropy
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import optimizers
from keras import backend as K
from keras.layers.core import Permute, Reshape
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Activation
from keras.models import Sequential
import keras
# from tensorflow import set_random_seed
from numpy.random import seed
import numpy as np
import os
import glob
from keras.utils import to_categorical

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#tensorflow.config.set_visible_devices([], 'GPU')

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
print("TensorFlow Version: ", tensorflow.__version__)


def one_hot_encoding(y_data, sub_dirs, categories=4):
    # Mapping = dict()
    #
    # count = 0
    # for i in sub_dirs:
    #     Mapping[i] = count
    #     count = count + 1

    y_features2 = []
    for i in range(len(y_data)):
        if y_data[i] == 'fall':
            lab = 0
        else:
            lab = 1
        # Type = y_data[i]
        # lab = Mapping[Type]
        y_features2.append(lab)

    y_features = np.array(y_features2)
    y_features = y_features.reshape(y_features.shape[0], 1)
    #print(y_features)
    
    y_features = to_categorical(y_features)

    return y_features


def full_3D_model(summary=False):
    print('building the model ... ')
    model = Sequential()

    model.add(Bidirectional(LSTM(64, return_sequences=False,
              stateful=False, input_shape=(20, 10*1024))))
    model.add(Dropout(.5, name='dropout_1'))
    model.add(Dense(128, activation='relu', name='DENSE_1'))
    model.add(Dropout(.5, name='dropout_2'))
    model.add(Dense(2, activation='softmax', name='output'))

    return model

if __name__=="__main__":

    extract_path = 'D:/研究生/研一/跌倒检测/1017/data_6.16_dynamic_txt/train_voxel_'
    checkpoint_model_path = "D:/研究生/研一/跌倒检测/1017/Radar/Classifiers/LSTM_dynamic"
    test_path = "D:/研究生/研一/跌倒检测/1017/data_6.16_dynamic_txt/test_voxel_data"
    sub_dirs = ['fall', 'lie', 'sit', 'walk']
    


    # random seed.
    rand_seed = 1
    seed(rand_seed)
    tensorflow.random.set_seed(rand_seed)



    # loading the train data
    Data_path = extract_path + sub_dirs[0]

    data = np.load(Data_path+'.npz')
    train_data = data['arr_0']
    train_data = np.array(train_data, dtype=np.dtype(np.int32))
    train_label = data['arr_1']

    del data
    print(train_data.shape, train_label.shape)

    Data_path = extract_path+sub_dirs[1]
    data = np.load(Data_path+'.npz')
    train_data = np.concatenate((train_data, data['arr_0']), axis=0)
    train_label = np.concatenate((train_label, data['arr_1']), axis=0)


    del data

    print(train_data.shape, train_label.shape)


    Data_path = extract_path + sub_dirs[2]
    data = np.load(Data_path+'.npz')
    train_data = np.concatenate((train_data, data['arr_0']), axis=0)
    train_label = np.concatenate((train_label, data['arr_1']), axis=0)

    del data
    print(train_data.shape, train_label.shape)

    Data_path = extract_path+sub_dirs[3]
    data = np.load(Data_path+'.npz')
    train_data = np.concatenate((train_data, data['arr_0']), axis=0)
    train_label = np.concatenate((train_label, data['arr_1']), axis=0)

    del data

    train_label = one_hot_encoding(train_label, sub_dirs, categories=4)

    train_data = train_data.reshape(
        train_data.shape[0], train_data.shape[1], train_data.shape[2] * train_data.shape[3] * train_data.shape[4])

    print('Training Data Shape is:')
    print(train_data.shape, train_label.shape)

    # loading the test data
    test = np.load(test_path + '.npz')
    test_data = test['arr_0']
    test_data = np.reshape(test_data, (-1, 20, 10 * 32 * 32))
    test_label = test['arr_1']


    # X_train, X_val, y_train, y_val = train_test_split(
    #     train_data, train_label, test_size=0.20, random_state=1)
    # del train_data, train_label


    model = full_3D_model()


    print("Model building is completed")


    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                        decay=0.0, amsgrad=False)
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=adam,
                metrics=['accuracy'])

    checkpoint = ModelCheckpoint(
        checkpoint_model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    callbacks_list = [checkpoint]


    # Training the model
    learning_hist = model.fit(train_data, train_label,
                            batch_size=20,
                            epochs=10,
                            verbose=1,
                            shuffle=True,
                            callbacks=callbacks_list,
                            )

    model.save(checkpoint_model_path)
