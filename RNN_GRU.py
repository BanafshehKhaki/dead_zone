import numpy as np
import tensorflow as tf
import pandas as pd
import keras as k
from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense
from keras.models import Model
from keras.utils import to_categorical
from math import sqrt
from keras.models import model_from_json
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
from numpy import array
import re
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
import gc
from numpy import argmax


# class myCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self,epoch ,logs={}):
#   		print (self.model.layers[0].get_weights())
     

#using TD 1 files and different PrH files to create new TD collection
def split_sequences(data, n_steps):
	X, y = list(), list()
	for i in range(len(data)):
		end_ix = i + n_steps
		if end_ix > len(data):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = data[i:end_ix, :-1], data[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

ScaledfolderName = 'MinMaxScaler'
# for targetvariable in (['dissolved_oxygen','ph','DOcategory','pHcategory']):
for targetvariable in (['pHcategory']):
	dfheader = pd.DataFrame( columns =['Score','TD','PrH','exp'])
	result_filename ='Results/test/'+targetvariable+'/sonde1_2018-06-01-2018-07-01_GRU30_v1june19.csv'
	dfheader.to_csv(result_filename)

# 	for n_steps in ([1,3,5,6,7,12,24,36]):
# 		for PrH_index in ([1,3,5,6,7,12,24,36,144,288,1008]):
	for n_steps in ([3]):
		for PrH_index in ([288]):
		
			training_whiten_filename = 'Sondes_data/train_data/train_data_normalized/'+ScaledfolderName+'/'+targetvariable+'/PrH_TD/sonde1_wo_2018-06-01-2018-07-01_TD1_PrH'+str(PrH_index)+'.csv'
			testing_whiten_filename=  'Sondes_data/test_data/test_data_normalized/'+ScaledfolderName+'/'+targetvariable+'/PrH_TD/sonde1_2018-06-01-2018-07-01_TD1_PrH'+str(PrH_index)+'.csv'
			print (testing_whiten_filename)
			
			training_whiten = pd.read_csv(training_whiten_filename)
			testing_whiten = pd.read_csv(testing_whiten_filename)
			
			training_whiten = training_whiten.drop(columns=['time'])
			testing_whiten = testing_whiten.drop(columns=['time'])
			
			if (targetvariable == 'dissolved_oxygen' or targetvariable == 'ph'): 
				training_whiten = training_whiten.drop(columns=['pHcategory','DOcategory'])
				testing_whiten = testing_whiten.drop(columns=['pHcategory','DOcategory'])
			elif (targetvariable == 'DOcategory'):
				training_whiten = training_whiten.drop(columns=['pHcategory'])
				testing_whiten = testing_whiten.drop(columns=['pHcategory'])
			elif (targetvariable == 'pHcategory'):
				training_whiten = training_whiten.drop(columns=['DOcategory'])
				testing_whiten = testing_whiten.drop(columns=['DOcategory'])
		
			training_whiten_values = training_whiten.values
			testing_whiten_values = testing_whiten.values
			
			# split into input and outputs
			train_X, train_y = split_sequences(training_whiten_values, n_steps)
			test_X, test_y = split_sequences(testing_whiten_values, n_steps)
			test_y = test_y.reshape((len(test_y), 1))
			
# 			print(train_X[0:5], train_y[0:5])
			
			if (targetvariable == 'DOcategory' or targetvariable == 'pHcategory'):
				start_time = time.time()
				
				model = Sequential()
				model.add(GRU(128,activation='sigmoid', return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
				model.add(GRU(128,activation='sigmoid'))

				if(targetvariable == 'DOcategory'):
					model.add(Dense(4, activation='softmax'))
					y_binary = to_categorical(train_y,4)
					test_y = to_categorical(test_y,4)
				else:
					model.add(Dense(12, activation='softmax'))
					y_binary = to_categorical(train_y,12)										
					test_y = to_categorical(test_y,12)
					
					
				model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
				# checkpoint
				filepath='Results/test/'+targetvariable+'/weights_checkpoints/sonde1_2018-06-01-2018-07-01_TD'+str(n_steps)+'_PrH'+str(PrH_index)+'_weights_checkpoint_epoch{epoch:02d}_june19.hdf5'
				checkpoint = ModelCheckpoint(filepath, save_weights_only=True)#, save_best_only=True)
				
				History = model.fit(train_X, y_binary, epochs=30, batch_size=32, validation_split=0.1 ,shuffle=False, callbacks=[checkpoint])


				elapsed_time = time.time() - start_time
				print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
			
				# serialize model to JSON
				model_json = model.to_json()
				with open('Scripts/models/'+targetvariable+'/sonde1_2018-06-01-2018-07-01_TD'+str(n_steps)+'_PrH'+str(PrH_index)+'_GRU30_v1'+".json", "w") as json_file:
					json_file.write(model_json)
				# serialize weights to HDF5
				model.save_weights('Scripts/models/'+targetvariable+'/sonde1_2018-06-01-2018-07-01_TD'+str(n_steps)+'_PrH'+str(PrH_index)+'_GRU30_v1'+".h5")
				print("Saved model to disk")

				
				# Final evaluation of the model
				scores = model.evaluate(test_X, test_y, verbose=0)
				

				catcrossEntValue = [scores[1]]
				catcrossEntNames = [str(n_steps)+','+str(PrH_index)+",acc"+",30june19"]

				data = {'Accuracy': catcrossEntValue,'Method':catcrossEntNames}
				df = pd.DataFrame(data=data)
				df.to_csv(result_filename, index=False, mode='a', header=False)
				
					# Plot training & validation accuracy values
				plt.plot(History.history['acc'])
				plt.plot(History.history['val_acc'])
				plt.title('Model accuracy')
				plt.ylabel('Accuracy')
				plt.xlabel('Epoch')
				plt.legend(['Train', 'Test'], loc='upper left')
				plt.savefig('Results/test/'+targetvariable+'/sonde1_2018-06-01-2018-07-01_TD'+str(n_steps)+'_PrH'+str(PrH_index)+'_GRU30_v1_accjune19.jpg')
				plt.close()

				# Plot training & validation loss values
				plt.plot(History.history['loss'])
				plt.plot(History.history['val_loss'])
				plt.title('Model loss')
				plt.ylabel('Loss')
				plt.xlabel('Epoch')
				plt.legend(['Train', 'Test'], loc='upper left')
				plt.savefig('Results/test/'+targetvariable+'/sonde1_2018-06-01-2018-07-01_TD'+str(n_steps)+'_PrH'+str(PrH_index)+'_GRU30_v1_lossjune19.jpg')
				plt.close()
				
				yhat = model.predict(test_X)
# 				print(test_y.shape)
# 				print(test_y)
				actual_y = argmax(test_y, axis=1)
# 				print(actual_y.shape)
# 				print(actual_y)
# 				print(argmax(yhat, axis=1))
				# Plot Predictions
				plt.scatter(np.arange(0,len(test_y)),actual_y)
				plt.scatter(np.arange(0,len(yhat)),argmax(yhat, axis=1))
				plt.title('Actual and Prediction ')
				plt.legend(['Actual', 'Prediction'], loc='upper left')
				plt.savefig('Results/test/'+targetvariable+'/sonde1_2018-06-01-2018-07-01_TD'+str(n_steps)+'_PrH'+str(PrH_index)+'_GRU30_v1_june19.jpg')
				plt.close()
				
				
# 				k.clear_session()
				gc.collect()
				del model



			elif(targetvariable == 'dissolved_oxygen' or targetvariable == 'ph'):
				y_scaler_filename = 'Sondes_data/train_data/train_data_normalized/'+ScaledfolderName+'/'+targetvariable+'/sonde1_wo_2018-06-01-2018-07-01_'+ScaledfolderName+'_y'+'.save'
				y_scaler = joblib.load(y_scaler_filename)
				start_time = time.time()
				
				model = Sequential()
				model.add(GRU(128, activation='sigmoid', recurrent_activation='sigmoid', return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2]),reset_after=True))
				model.add(GRU(128, activation='sigmoid',recurrent_activation='sigmoid',reset_after=True))
				model.add(Dense(1))
				model.compile(loss='mse', optimizer='adam')
				
				# checkpoint
				filepath='Results/test/'+targetvariable+'/weights_checkpoints/sonde1_2018-06-01-2018-07-01_TD'+str(n_steps)+'_PrH'+str(PrH_index)+'_weights_checkpoint_epoch{epoch:02d}_june19.hdf5'
				checkpoint = ModelCheckpoint(filepath, save_weights_only=True)

				callbacks_list = [checkpoint]
				
				# fit network 
				History = model.fit(train_X, train_y, epochs=30, batch_size=32, validation_split=0.1,shuffle=False, callbacks=callbacks_list)

				elapsed_time = time.time() - start_time
				print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
			
				# serialize model to JSON
				model_json = model.to_json()
				with open('Scripts/models/'+targetvariable+'/sonde1_2018-06-01-2018-07-01_TD'+str(n_steps)+'_PrH'+str(PrH_index)+'_GRU30_june19'+".json", "w") as json_file:
					json_file.write(model_json)
				# serialize weights to HDF5
				model.save_weights('Scripts/models/'+targetvariable+'/sonde1_2018-06-01-2018-07-01_TD'+str(n_steps)+'_PrH'+str(PrH_index)+'_GRU30_june19'+".h5")
				print("Saved model to disk")

				# make a prediction
				yhat = model.predict(test_X)
# 				print(np.any(np.isnan(yhat)))
# 				print(np.all(np.isfinite(yhat)))

				test_y = test_y.reshape((len(test_y), 1))
				inv_y = y_scaler.inverse_transform(test_y)

				inv_yhat = y_scaler.inverse_transform(yhat)
				
				# calculate RMSE
				rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
				GRU_r2_score = r2_score(inv_y, inv_yhat)

				r2_list = [GRU_r2_score,rmse]
				r2_names = [str(n_steps)+','+str(PrH_index)+',_r2'+'30june19', str(n_steps)+','+str(PrH_index)+',_RMSE'+',30june19']
				data = {'R2-score':r2_list,'Method':r2_names}
				df = pd.DataFrame(data=data)
				df.to_csv(result_filename, index=False, mode='a', header=False)
				
				# evaluate the model
				# scores = model.evaluate(test_X, test_y, verbose=0)
# 				print(scores)
# 				print(model.metrics_names)
					
				# Plot Predictions
				plt.plot(inv_y)
				plt.plot(inv_yhat)				
				plt.title('Actual and Prediction ')
				plt.legend(['Actual', 'Prediction'], loc='upper left')
				plt.savefig('Results/test/'+targetvariable+'/sonde1_2018-06-01-2018-07-01_TD'+str(n_steps)+'_PrH'+str(PrH_index)+'_GRU30_v1_june19.jpg')
				plt.close()


				# Plot training & validation loss values
				plt.plot(History.history['loss'])
				plt.plot(History.history['val_loss'])
				plt.title('Model loss')
				plt.ylabel('Loss')
				plt.xlabel('Epoch')
				plt.legend(['Train', 'Test'], loc='upper left')
				plt.savefig('Results/test/'+targetvariable+'/sonde1_2018-06-01-2018-07-01_TD'+str(n_steps)+'_PrH'+str(PrH_index)+'_GRU30_v1_lossjune19.jpg')
				plt.close()
				
# 				k.clear_session()
				gc.collect()
				del model

