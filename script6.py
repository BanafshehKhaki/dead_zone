import pandas as pd
import numpy as np
import sys
import re
import datetime
import time

#####################################################################
# Main function: Create train and test file for a sonde using dates given
# Parameter: File path start and end date 
# Returns: a test file in test_data > test_data , a train file in train_data > train_data for the given file
# - Uncomment to use a directory for specific dates
#####################################################################
def main(FilePath, start_date, end_date):
	
	fileName = FilePath.split('/')
	df = pd.read_csv(FilePath, index_col='time')
	df.index = df.index.map(np.int64)


	start_dt = time.mktime(datetime.datetime.strptime(start_date, '%Y-%m-%d').timetuple())
	end_dt = time.mktime(datetime.datetime.strptime(end_date, '%Y-%m-%d').timetuple())

	index1 = df.index.get_loc(start_dt,"NEAREST")
	index2 = df.index.get_loc(end_dt,"NEAREST")
		
	df_test = df.loc[slice(df.index[index1], df.index[index2])]
	df_train = df.drop(df_test.index, axis=0)
	
	Name = re.sub('.csv','',fileName[len(fileName)-1])
	df_train.to_csv('Sondes_data/train_data/train_data/'+Name+"_wo_"+str(start_date)+"-"+str(end_date)+".csv")
	df_test.to_csv('Sondes_data/test_data/test_data/'+Name+"_"+str(start_date)+"-"+str(end_date)+'.csv')



if __name__ == "__main__":

	if len(sys.argv) == 4:
		path  = sys.argv[1]
		date1 = sys.argv[2]
		date2 = sys.argv[3]
		main(path, date1,date2) 

	else:
		print("Please provide a correct path  start and end data")
		print("Example: python script6.py Sondes_data/data_withCategories/sonde1.csv 2018-06-01 2018-07-01 ")
		
		
		
		
		
		
	