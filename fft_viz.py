## FFT 
## 
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import re
import datetime
import time
import seaborn as sns



def main(FilePath, start_date, end_date):
	
	fileName = FilePath.split('/')
	df = pd.read_csv(FilePath, index_col='time')
	df.index = df.index.map(np.int64)


	start_dt = time.mktime(datetime.datetime.strptime(start_date, '%Y-%m-%d').timetuple())
	end_dt = time.mktime(datetime.datetime.strptime(end_date, '%Y-%m-%d').timetuple())

	index1 = df.index.get_loc(start_dt,"NEAREST")
	index2 = df.index.get_loc(end_dt,"NEAREST")
		
	df_summer = df.loc[slice(df.index[index1], df.index[index2])]
# 	df_rest = df.drop(df_summer.index, axis=0)
	
	df_summer = df_summer.drop(columns=['station_name','lat','lon','depth','DOcategory','pHcategory'])
# 	df_rest = df_rest.drop(columns=['station_name','lat','lon','depth','DOcategory','pHcategory'])
	
	df_data = df_summer
	name = start_date +"-"+ end_date#'Except_summer'
	
	freq = np.fft.fftfreq(len(df_data), df_data.index[1] - df_data.index[0])

	plt.subplot(321)
	f = np.fft.fft(df_data['dissolved_oxygen'].values)	
	plt.plot(freq, (f))
	

	plt.subplot(322)
	f = np.fft.fft(df_data['ph'].values)
	plt.plot(freq, (f))
	
	plt.subplot(323)
	f = np.fft.fft(df_data['Water_Temperature_at_Surface'].values)
	plt.plot(freq, (f))
	
	plt.subplot(324)
	f = np.fft.fft(df_data['ysi_blue_green_algae'].values)	
	plt.plot(freq, (f))
	
	plt.subplot(325)
	f = np.fft.fft(df_data['ysi_chlorophyll'].values)	
	plt.plot(freq, (f))
	
	plt.subplot(326)
	f = np.fft.fft(df_data['water_conductivity'].values)
	plt.plot(freq, (f))
	
	plt.savefig("Data_vis/FFT/"+fileName[len(fileName)-2]+"_fft_"+fileName[len(fileName)-1]+"_"+name+"_.jpg")
	
	
	
	
if __name__ == "__main__":

	if len(sys.argv) == 4:
		path  = sys.argv[1]
		date1 = sys.argv[2]
		date2 = sys.argv[3]
		main(path, date1,date2) 

	else:
		print("Please provide a correct path  start and end data")
		print("Example: python fft_viz.py Sondes_data/data_withCategories/sonde1.csv 2018-06-01 2018-09-01 ")