import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#####################################################################
# Script 11 : Training Linear Regression TO BE ADDED
# Parameter: 
# Returns: 
#####################################################################	
def main(train_filePath, PrH_Steps, td_steps, PrH_name):
	fileName = filePath.split("/")
# 	df = pd.read_csv(filePath)
	
	
	# Linear
#     reg_model = LinearRegression().fit(train_data,train_y)
#     y_pred = reg_model.predict(test_data)
#     regression_coefficient = pd.DataFrame({'Feature': test_data.columns, 'Coefficient': reg_model.coef_}, columns=['Feature', 'Coefficient'])




if __name__ == "__main__":
	if len(sys.argv) ==5:
		filePath = sys.argv[1]
		
		main(filePath)
	else:
		print("Please enter the path to the file <PrH value> <Temporal depth_steps>")
		
		
		
		
		