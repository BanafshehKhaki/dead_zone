import pandas as pd
import numpy as np
import sys
import re
import random

#####################################################################
# Main function: Create validation train and test file for a sonde using train TD, PrH data
# Parameter: File path start and end date 
# Returns: validation set
# - Uncomment to use a directory for specific dates
#####################################################################
def main(FilePath):

	for number in range(1,31):
		df_train_valid = pd.read_csv(FilePath)	
		index_range = range(0, int((len(df_train_valid.index) * 0.1)))
		df_test_valid = pd.DataFrame(columns=df_train_valid.columns)
		
		for i in index_range:
			loc = int(random.uniform(df_train_valid.index[0], len(df_train_valid.index)-1 ))
			#print (loc)
			
			df_test_valid = df_test_valid.append(df_train_valid.iloc[[loc]], ignore_index=True)
			#print (df_test_valid.tail(1))

			df_train_valid = df_train_valid.drop(loc).reset_index(0,len(df_train_valid.index)-i)
			#print (df_train_valid.iloc[[loc]])


		FilePath1 = re.sub('PrH_TD/','Validation_train/',FilePath)
		FilePath1 = re.sub('.csv',"_"+str(number)+'.csv',FilePath1)		
		df_train_valid.to_csv(FilePath1)
		print(FilePath1)
		
		FilePath2 = re.sub('PrH_TD/','Validation_test/',FilePath)	
		FilePath2 = re.sub('.csv',"_"+str(number)+'_test.csv',FilePath2)	
		df_test_valid.to_csv(FilePath2)



if __name__ == "__main__":

	if len(sys.argv) == 2:
		path  = sys.argv[1]
		main(path) 

	else:
		print("Please provide a correct path")
		print("Example: python make_cv.py Sondes_data/train_data/train_data_normalized/MinMaxScaler/dissolved_oxygen/PrH_TD/sonde1_wo_2018-06-01-2018-07-01_TD1_PrH1.csv ")
		
		
		
		
		
		
	