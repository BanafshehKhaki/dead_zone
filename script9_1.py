#Making Prediction Horizon(PrH( and Temporal Depth (TD) files
import pandas as pd
import numpy as np
import sys
import re

#####################################################################
# Prediction horizon function
# Parameters PrH steps 
# Returns  Y target value of corresponding PrH
#####################################################################
def prediction_horizon(data, pd_steps, target):
# 	target = data.columns[len(data.columns)-1]
	target_values = data[[target]]
	target_values = target_values.drop(target_values.index[0:pd_steps],axis=0)
	target_values.index= np.arange(0,len(target_values[target]))
    
	return target_values

#####################################################################
# Temporal depth function
# Parameters TD steps 
# Returns X values of corresponding TD
#####################################################################
def temporal_depth(data, n_steps, features_count):
 X = list()
 for i in range(len(data)):
	 end_ix = i + n_steps
# 	 check for dataset boundary
	 if end_ix > len(data):
		 break
	 seq_X = data[i:end_ix, :].reshape(1,features_count*n_steps)
	 X.append(seq_X)
 return np.array(X)


#####################################################################
# Script 9 : Temporal depth and Prediction Horizon files from whiten files
# Parameter: File path method 
# Returns: whiten file TD and PrH stored in the whiten folder > PrH_TD
#####################################################################	
def main(filePath, PrH_Steps, td_steps, PrH_name, target):
	fileName = filePath.split("/")
	df = pd.read_csv(filePath)
	
	target_values = prediction_horizon(df, PrH_Steps, target)
	
	df = df.drop(df.index[len(df.index)-PrH_Steps : len(df.index)], axis=0)
	
	new_data = df.drop(columns=['station_name','depth','lat','lon'])
	features_count = len(new_data.columns)
	columns = new_data.columns
	
	new_data = temporal_depth(new_data.values, td_steps, features_count)
	new_data = new_data.reshape(new_data.shape[0],new_data.shape[2])

	# Creating name of the new columns of data:
# 	columns = list()
# 	timestamp = 1
# 	index =1 
# 	for i in range(1,features_count*td_steps+1):
# 		if index<features_count:
# 			columns.append('f'+str(timestamp)+"_"+str(index))
# 			index= index+1
# 		else:
# 			columns.append('f'+str(timestamp)+"_"+str(index))
# 			index = 1
# 			timestamp = timestamp +1
# 
# 		if timestamp > td_steps:
# 			break;
# 	
	# Creating dataFrame and putting the target values + category data back in the file		
	new_datapd = pd.DataFrame(new_data, columns=columns)
	new_datapd['Target_'+target] = target_values
	
	# Saving in /PrH_TD folder
	filePath = re.sub('sonde','PrH_TD/sonde',filePath)
	filePath = re.sub('.csv','_TD'+str(td_steps)+'_PrH'+str(PrH_name)+'.csv',filePath)
	new_datapd.to_csv(filePath,index= False)


if __name__ == "__main__":
	if len(sys.argv) ==6:
		filePath = sys.argv[1]
		PrH_Steps = int(sys.argv[2])
		td_steps = int(sys.argv[3])
		PrH_name = int(sys.argv[4])
		target   = int(sys.argv[5])
		main(filePath, PrH_Steps,td_steps,PrH_name, target)
	else:
		print("Please enter the path to the file <PrH value> <Temporal depth_steps>")
		
		
		
		
		
		