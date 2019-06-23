import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate , cross_val_score
from sklearn.metrics import recall_score
from sklearn.svm import SVR
from sklearn import svm
import sys


#**************************************************
# Main Function
# Arguments: File path from prepared data
# trains SVR / SVM model
# Saves the trained model in SVM_models
#**************************************************
def main(FilePath, parameters='M'):

	df = pd.read_csv(FilePath)
	df = df[['water_conductivity','Water_Temperature_at_Surface','dissolved_oxygen','ysi_blue_green_algae','ph','ysi_chlorophyll']]
	
	
	svm_reg=svm.SVR()
	svm_reg.fit(df.drop(columns=['dissolved_oxygen']),df.dissolved_oxygen)
	y1_svm=svm_reg.predict(df.drop(columns=['dissolved_oxygen']))
	r2_svm = r2_score(y_true, y1_svm)
	print (r2_svm)
	
	

if __name__ == "__main__":
	if len(sys.argv) >= 2:
		path = sys.argv[1]
# 		parameters = sys.argv[2]
		
		main(path)
	else:
		print ("Please enter < directory path to the files> <PW> for pair-wise plots")
	
	
	