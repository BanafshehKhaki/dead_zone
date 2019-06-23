import sys
import re
import pandas as pd

def main(FilePath):
	fileName = FilePath.split('/')
	
	if fileName[len(fileName)-1] =='sonde1.csv':
		df = pd.read_csv(FilePath)
		df = df[(df[['water_conductivity']] > -100).all(1)]
		df = df[(df[['Water_Temperature_at_Surface']] > -100).all(1)]
		df = df[(df[['ph']] > -100).all(1)]

		df = df[ (df['ph'] !=0)  | (df['Water_Temperature_at_Surface'] !=0) 
								 | (df['water_conductivity'] !=0)
								 | (df['dissolved_oxygen'] !=0)
								 | (df['ysi_chlorophyll'] !=0)
								 | (df['ysi_blue_green_algae'] !=0)]
	
	df.to_csv('Sondes_data/data_corrected/'+fileName[len(fileName)-1],index= False)
    	
	
	
	
	
if __name__ == "__main__":

	if len(sys.argv) == 2:
		path  = sys.argv[1]
		main(path) 

	else:
		print("Please provide a correct path")
		print("Example: python script4.py Sondes_data/data_organized/sonde1.csv")
		
		