import os
import re
import time
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
import sklearn.metrics as skm
from keras import backend as Kb
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from keras.utils import to_categorical
import sklearn.metrics as skm
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Reshape
from keras.models import Model
from keras import metrics
import tensorflow as tf
from numpy import array
import seaborn as sns
from math import sqrt
from numpy import argmax
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.layers import Dropout
import keras
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier

param_grid = {  'param_grid_DT1': {
                    'model__class_weight': ['balanced'], #{0: 1, 1: 50, 2: 100},
                    'model__criterion': ['gini'],  # entropy
                    'model__max_depth': np.arange(8, 15),
                    'model__min_samples_split': np.arange(0.1, 1),
                    'model__min_samples_leaf': np.arange(1, 6),
                    'model__max_features': ['log2', 'auto', 'sqrt'],
                                },

                'param_grid_LR1': {
                    'model__class_weight': [{0: 1, 1: 50, 2: 100}],
                    "model__C": [0.001, 0.01], 
                    "model__penalty": ["l1", "l2"],
                    'model__solver': ['saga']},

                'param_grid_LSTM1': {
                    
                    'model__batch_size': [64, 32, 128, 512],
                    'model__epochs':  [20, 50, 100, 200],
                    'model__learn_rate': [0.001, 0.01, 0.0001, 0.1],
                    'model__weight_constraint': [1, 2, 3, 4, 5],
                    'model__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    'model__neurons': [1, 3, 6, 12, 24, 32, 64],
                    'model__activation':  ['softmax', 'relu', 'tanh', 'sigmoid']
                },

                'param_grid_NN1': {
                    
                    'model__batch_size': [64, 32, 128, 512],
                    'model__epochs':  [20, 50, 100, 200],
                    'model__learn_rate': [0.001, 0.01, 0.0001, 0.1],
                    'model__weight_constraint': [1, 2, 3, 4, 5],
                    'model__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    'model__neurons': [1, 3, 6, 12, 24, 32, 64],
                    'model__activation':  ['softmax', 'relu', 'tanh', 'sigmoid']
                },

                'param_grid_SVC1': {
                    'model__C': [10, 100],
                    'model__class_weight': [{0: 1, 1: 50, 2: 100},'balanced'],
                    'model__gamma': [1e-4, 0.01, 0.1,'scale']
                },

                'param_grid_RF1': {
                    'model__base_estimator__class_weight': [{0: 1, 1: 50, 2: 100},'balanced', 'balanced subsample'], 
                    # Maximum number of levels in tree
                    'model__base_estimator__max_depth': [30, 50, 60, 100, None],
                    # Number of trees in random forest
                    'model__base_estimator__n_estimators': (30, 100),
                    # Number of features to consider at every split
                    'model__base_estimator__max_features': ['auto', 'sqrt'],
                    # Minimum number of samples required to split a node
                    'model__base_estimator__min_samples_split': [5, 10],
                    # Minimum number of samples required at each leaf node
                    'model__base_estimator__min_samples_leaf': [1, 2, 4],
                    # Method of selecting samples for training each tree
                    'model__base_estimator__bootstrap': [True, False]
                },
                'param_grid_DT0': {
                    'model__criterion': ['mae', 'mse', 'friedman_mse'],
                    'model__max_depth': np.arange(8, 15),
                    'model__min_samples_split': np.arange(0.1, 1),
                    'model__min_samples_leaf': np.arange(1, 6),
                    'model__max_features': ['log2', 'auto', 'sqrt'],
                                },

                'param_grid_LR0': {
                    'model__fit__alpha':[550, 580, 600, 620, 650]
                },

                'param_grid_LSTM0': {
                    
                    'model__batch_size': [64, 32, 128, 512],
                    'model__epochs':  [20, 50, 100, 200],
                    'model__learn_rate': [0.001, 0.01, 0.0001, 0.1],
                    'model__weight_constraint': [1, 2, 3, 4, 5],
                    'model__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    'model__neurons': [1, 3, 6, 12, 24, 32, 64],
                    'model__activation':  ['relu', 'tanh', 'sigmoid']
                },

                'param_grid_NN0': {
                    
                    'model__batch_size': [64, 32, 128, 512],
                    'model__epochs':  [20, 50, 100, 200],
                    'model__learn_rate': [0.001, 0.01, 0.0001, 0.1],
                    'model__weight_constraint': [1, 2, 3, 4, 5],
                    'model__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    'model__neurons': [1, 3, 6, 12, 24, 32, 64],
                    'model__activation':  ['relu', 'tanh', 'sigmoid']
                },

                'param_grid_SVC0': {
                    'model__C': [1.5,10, 100],
                    'model__gamma': [1e-4, 0.01, 0.1],
                    'model__epsilon': [0.01, 0.1, 0.2, 0.5]
                },

                'param_grid_RF0': {
                    # Maximum number of levels in tree
                    'model__max_depth': [3, 30, 50, 60, 100, None],
                    # Number of trees in random forest
                    'model__n_estimators': (30, 100,200, 400),
                    # Number of features to consider at every split
                    'model__max_features': ['auto', 'sqrt'],
                    # Minimum number of samples required to split a node
                    'model__min_samples_split': [2, 5, 10],
                    # Minimum number of samples required at each leaf node
                    'model__min_samples_leaf': [1, 2, 4],
                    # Method of selecting samples for training each tree
                    'model__bootstrap': [True, False]
                }
}
trained_param_grid = { 
    'param_grid_RF1' : {
        # max_depth,  n_estimators, max_features, min_samples_split, min_samples_leaf, bootstrap, n_steps, class_weights
        'param_DOcategory_1': [100, 100, 'sqrt',5,4, True, 24,{0: 1, 1: 50, 2: 100}],
        'param_DOcategory_3': [60,30,'auto',10,4,False,24,{0: 1, 1: 50, 2: 100}],
        'param_DOcategory_6': [30,100,'auto',5,4,True,24,{0: 1, 1: 50, 2: 100}],
        'param_DOcategory_12': [100,100,'auto',5,1,True,12,{0: 1, 1: 50, 2: 100}],
        'param_DOcategory_24': [100,100,'auto',5,1,True,24,{0: 1, 1: 50, 2: 100}],
        'param_pHcategory_1': [100,100,'auto',10,4,True,24,'balanced'],
        'param_pHcategory_3': [100,100,'auto',10,4,True,24,'balanced'],
        'param_pHcategory_6': [100,100,'auto',10,4,True,24,'balanced'],
        'param_pHcategory_12': [60,100,'sqrt',10,1,False,6,'balanced'],
        'param_pHcategory_24': [100,100,'auto',10,4,True,24,'balanced'],
    },
    'param_grid_RF0' : {
        'param_dissolved_oxygen_1': [],
        'param_dissolved_oxygen_3': [],
        'param_dissolved_oxygen_6': [],
        'param_dissolved_oxygen_12': [],
        'param_dissolved_oxygen_24': [],
        'param_ph_1': [None, 100,'sqrt',10,2,False,12],
        'param_ph_3': [50,200,'sqrt',5,2,True,12],
        'param_ph_6': [50,200,'sqrt',5,2,True,12],
        'param_ph_12': [50,200,'sqrt',5,2,True,6],
        'param_ph_24': [3,400,'sqrt',5,4,True,3],
        },

    'param_grid_dummy1' : {
        'param_DOcategory_1': [],
        'param_DOcategory_3': [],
        'param_DOcategory_6': [],
        'param_DOcategory_12': [],
        'param_DOcategory_24': [],

        'param_pHcategory_1': [],
        'param_pHcategory_3': [],
        'param_pHcategory_6': [],
        'param_pHcategory_12': [],
        'param_pHcategory_24': [],
        },
    'param_grid_dummy0' : {
        'param_dissolved_oxygen_1': [],
        'param_dissolved_oxygen_3': [],
        'param_dissolved_oxygen_6': [],
        'param_dissolved_oxygen_12': [],
        'param_dissolved_oxygen_24': [],

        'param_ph_1': [],
        'param_ph_3': [],
        'param_ph_6': [],
        'param_ph_12': [],
        'param_ph_24': [],
        },
    }

def algofind(modelname, input_dim,n_steps, cat):
    if cat ==1 :
        if modelname == 'LSTM':
            model = KerasClassifier(build_fn=create_LSTM_model, input_dim=input_dim,
                                    epochs=20, batch_size=64,  nsteps=int(n_steps), verbose=0)
        elif modelname == 'DT':
            model = DecisionTreeClassifier() #OneVsRestClassifier(
        elif modelname == 'RF':
            model = RandomForestClassifier()
        elif modelname == 'LR':
            model = LogisticRegression(multi_class='multinomial', max_iter=2000)
        elif modelname == 'SVC':
            model = SVC()
        elif modelname == 'NN':
            model = KerasClassifier(build_fn=create_NN_model,
                                    input_dim=input_dim, verbose=0)
        
    elif cat==0:
        if modelname == 'LSTM':
            model = KerasRegressor(build_fn=create_reg_LSTM_model, input_dim=input_dim,
                                    epochs=20, batch_size=64,  nsteps=int(n_steps), verbose=0)
        elif modelname == 'DT':
            model = DecisionTreeRegressor()
        elif modelname == 'RF':
            model = RandomForestRegressor()
        elif modelname == 'LR':
            model = Pipeline([('poly', PolynomialFeatures()),('fit', Ridge())])
        elif modelname == 'SVC':
            model = SVR()
        elif modelname == 'NN':
            model = KerasRegressor(build_fn=create_reg_NN_model,
                                    input_dim=input_dim, verbose=0)

    return model


# def custom_score(test_y, predictions):
#     test_y = np.argmax(test_y, axis=-1)
#     predictions = np.argmax(predictions, axis=-1)
#     F1_01 = skm.f1_score(test_y, predictions, labels=[1, 2], average='micro')
#     return F1_01


def forecast_accuracy(predictions, test_y, cat):
    if cat ==1:
        F1 = skm.f1_score(test_y, predictions, labels=[1, 2], average=None).ravel()

        P = skm.precision_score(test_y, predictions, labels=[
                                1, 2], average=None).ravel()

        R = skm.recall_score(test_y, predictions, labels=[
                            1, 2], average=None).ravel()

        tp, fn, fp, tn = confusion_matrix(
            test_y, predictions, labels=[1, 2]).ravel()
        print(tp, fn, fp, tn)
        acc = (tp+tn)/(tp+fp+fn+tn)
        F1_0_1 = skm.f1_score(test_y, predictions, labels=[1, 2], average='micro')
        F1_all = skm.f1_score(test_y, predictions,average='micro')
        fbeta = skm.fbeta_score(test_y,predictions,labels=[1, 2],  beta=0.5, average='weighted')

        return(F1[0], F1[1], P[0], P[1], R[0], R[1], acc, F1_0_1, F1_all, fbeta)
    else:
        test_y = test_y + 0.00001
        mape = np.mean(np.abs(predictions - test_y)/np.abs(test_y))  # MAPE
        me = np.mean(predictions - test_y)             # ME
        mpe = np.mean((predictions - test_y)/test_y)   # MPE

        # mae = np.mean(np.abs(predictions - test_y))    # MAE
        mae = skm.mean_absolute_error(test_y, predictions)

        # rmse = np.sqrt(np.mean((predictions - test_y)**2))  # RMSE
        rmse = np.sqrt(skm.mean_squared_error(test_y,predictions))

        corr = np.corrcoef(predictions, test_y)[0, 1]   # corr
        
        r2 = skm.r2_score(test_y,predictions)       # R2
        # 1-(sum((predictions - test_y)**2)/sum((test_y-np.mean(test_y))**2))

        return(mape, me, mae,mpe,rmse,corr, r2)

def inverseTransform(predictions, test_y, method, file, path):
    y_scaler_path = path
    y_scaler_filename = re.sub('.csv', '_'+method+'_y.save', file)
    y_scaler = joblib.load(y_scaler_path+y_scaler_filename)
    inv_y = y_scaler.inverse_transform(test_y.reshape(-1, 1))
    inv_yhat = y_scaler.inverse_transform(predictions.reshape(-1, 1))
    return inv_yhat, inv_y

def transform(predictions, test_y, method, target, file):
    print(predictions.shape)
    path = 'Sondes_data/train/train_data_normalized/' + \
        method+'/'+target+'/'
    if method == 'MinMaxScaler' or method == 'StandardScaler':
        inv_yhat, inv_y = inverseTransform(
            predictions, test_y, method, file, path)
    else:
        inv_yhat, inv_y =predictions, test_y
    return inv_yhat, inv_y


def split_sequences(data, n_steps):
    data = data.values
    X, y = list(), list()

    for i in range(len(data)):
        end_ix = i + n_steps*6
        if end_ix > len(data):
            break

        Kx = np.empty((1, 12))
        for index in np.arange(i, i+(n_steps*6), step=6, dtype=int):
            eachhour = index + 6
            if eachhour > len(data) or i+(n_steps*6) > len(data):
                break

            a = data[index: eachhour, : -1]
            hourlymean_x = np.mean(a, axis=0)
            hourlymean_y = data[eachhour-1, -1]

            hourlymean_x = hourlymean_x.reshape((1, hourlymean_x.shape[0]))
            if index != i:
                Kx = np.append(Kx, hourlymean_x, axis=0)
            else:
                Kx = hourlymean_x

        X.append(Kx)
        y.append(hourlymean_y)
    print(np.array(X).shape)
    return np.array(X), np.array(y)


def temporal_horizon(df, pd_steps, target):
    pd_steps = pd_steps * 6
    target_values = df[[target]]
    target_values = target_values.drop(
        target_values.index[0: pd_steps], axis=0)
    target_values.index = np.arange(0, len(target_values[target]))

    df = df.drop(
        df.index[len(df.index)-pd_steps: len(df.index)], axis=0)
    df['Target_'+target] = target_values
    print('Target_'+target)
    return df

######################################
# Creating custom datasets
# By choosing a random minutes from each hour to represent that hour
######################################


def custom_cv_2folds(X, kfolds):
    n = X.shape[0]
    print('******** creating custom CV:')
    i = 1
    while i <= kfolds:
        np.random.seed(i)
        idx = np.empty(0, dtype=int)
        for index in np.arange(0, n-6, step=6, dtype=int):
            randwindowpoint = np.random.randint(0, 6, size=1)
            idx = np.append(idx, [randwindowpoint+index])
            # print(idx)
        print(idx[0:10])
        yield idx[:int(len(idx)*0.7)], idx[int(len(idx)*0.7):]
        i = i+1

def custom_cv_kfolds_testdataonly(X, kfolds):
    n = X.shape[0]
    print('******** creating custom CV:')
    i = 1
    while i <= kfolds:
        np.random.seed(i)
        idx = np.empty(0, dtype=int)
        for index in np.arange(0, n-6, step=6, dtype=int):
            randwindowpoint = np.random.randint(0, 6, size=1)
            idx = np.append(idx, [randwindowpoint+index])
            # print(idx)
        print(idx[0:10])
        yield idx[:int(len(idx))]
        i = i+1

def create_LSTM_model(neurons=1, learn_rate=0.01, dropout_rate=0.0, weight_constraint=0, activation='relu', input_dim=None, nsteps=1):
    model = Sequential()
    model.add(Reshape(target_shape=(
        nsteps, input_dim[2]), input_shape=(nsteps*input_dim[2],)))
    model.add(LSTM(neurons, activation=activation, return_sequences=True,
                   kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(neurons, activation=activation,
                   kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dense(3, activation='softmax'))
    opt = Adam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['acc', keras.losses.categorical_crossentropy])
    print('model: ' + str(model))
    return model

def create_reg_LSTM_model(neurons=1, learn_rate=0.01, dropout_rate=0.0, weight_constraint=0, activation='sigmoid', input_dim=None, nsteps=1):
    model = Sequential()
    model.add(Reshape(target_shape=(
        nsteps, input_dim[2]), input_shape=(nsteps*input_dim[2],)))
    model.add(LSTM(neurons, activation=activation, return_sequences=True,
                   kernel_constraint=maxnorm(weight_constraint)))

    model.add(Dropout(dropout_rate))
    model.add(LSTM(neurons, activation=activation))
    model.add(Dense(1))
    opt = Adam(lr=learn_rate)
    model.compile(loss='mae', optimizer=opt)
    print('model: ' + str(model))
    return model

def create_NN_model(neurons=1, learn_rate=0.01, dropout_rate=0.0, weight_constraint=0, activation='relu', input_dim=None):
    model = Sequential()
    model.add(Dense(neurons, activation=activation,
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(3, activation='softmax'))
    opt = Adam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[
        'acc', keras.losses.categorical_crossentropy])
    print('model: ' + str(model))
    return model


def create_reg_NN_model(neurons=1, learn_rate=0.01, dropout_rate=0.0, weight_constraint=0, activation='sigmoid', input_dim=None, nsteps=1):
    model = Sequential()
    model.add(Dense(neurons, activation=activation,kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1))
    opt = Adam(lr=learn_rate)
    model.compile(loss='mae', optimizer=opt)
    print('model: ' + str(model))
    return model



def setfeatures(currentlist, n_steps):
    # Creating name of the new columns of data:
    columns = list(currentlist)
    for i in range(1, n_steps):
        w = '+'+str(i)+'h'
        a = [w + str(item) for item in currentlist]
        columns.extend(a)
    print(columns)
    return columns

def getlags_window(model_name,params, cat):
    if cat ==1:
        if model_name == 'RF':
            max_depth,  n_estimators, max_features, min_samples_split, min_samples_leaf, bootstrap,  n_steps, class_weight = params
    if cat ==0:
        if model_name == 'RF':
            max_depth,  n_estimators, max_features, min_samples_split, min_samples_leaf, bootstrap,  n_steps = params
    return n_steps

def getModel(model,input_dim, params,n_jobs, cat):
    if cat ==1:
        if model == 'RF':
            max_depth,  n_estimators, max_features, min_samples_split, min_samples_leaf, bootstrap,  n_steps, class_weight = params
            clf = RandomForestClassifier(max_depth=max_depth,
                                        bootstrap=bootstrap,
                                        n_estimators=n_estimators,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        max_features=max_features,
                                        class_weight=class_weight, n_jobs=n_jobs)
    else:
        if model == 'RF':
            max_depth,  n_estimators, max_features, min_samples_split, min_samples_leaf, bootstrap,  n_steps = params
            clf = RandomForestRegressor(max_depth=max_depth,
                                        bootstrap=bootstrap,
                                        n_estimators=n_estimators,
                                        min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf,
                                        max_features=max_features,n_jobs=n_jobs)

    return clf

def preparedata(dataset, PrH_index, lags, target,cat):
    dataset = dataset.drop(
                    columns=['time', 'depth', 'lat', 'lon', 'year'])
    # dataset = dataset.drop(['ysi_turbidity'], axis=1)  # lorain test
    dataset = dataset.dropna()
    print(dataset.head())
    dataset = temporal_horizon(
        dataset, PrH_index, target)

    train_X_grid, train_y_grid = split_sequences(
        dataset, lags)

    input_dim = train_X_grid.shape

    inds = np.where(np.isnan(train_X_grid))
    train_X_grid[inds] = 0

    train_X_grid = train_X_grid.reshape(
        train_X_grid.shape[0], train_X_grid.shape[1]*train_X_grid.shape[2])

    train_y_grid = train_y_grid.reshape(len(train_y_grid),) # get a warning if 1 is out in as second parameter: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    if cat ==1:
        train_y_grid = train_y_grid.astype(int)
    inds = np.where(np.isnan(train_y_grid))
    train_y_grid[inds] = 0
    return train_X_grid, train_y_grid, input_dim, list(dataset.columns[:-1])


