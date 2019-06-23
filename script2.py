import sys;
import time;
import csv;
import pandas as pd
import numpy as np

# TDS GLOS time is saved from 2001-01-01, converting it to current time UTC
# dt = datetime.datetime.strptime('2001-01-01', '%Y-%m-%d')
# time.mktime(dt.timetuple())  #Out[45]: 978325200.0
# time.ctime(978325200.0 + 457220400) # Out[46]: 'Sun Jun 28 22:40:00 2015'


#**************************************************
# Main Function
# Arguments: File path from downloaded raw data
# Retrieves data and corrects the time in UTC
# Saves the file in local folder
#**************************************************
def main(FilePath):
		df = pd.read_csv(FilePath)
		df['time'] = df['time'] + 978325200.0
			
		fileName = FilePath.split('/')
		df.to_csv("Sondes_data/data_timecorrected/"+fileName[len(fileName)-1], index=False)
		print("file created: " + (fileName[len(fileName)-1]))
		
		

if __name__ == "__main__":
	if len(sys.argv) == 2:
		path = sys.argv[1]
		files = [f for f in os.listdir(path) if f.endswith(".csv")]

		for file in files:
			main(path+file)
	else:
		print("Please provide a directorty path")

	