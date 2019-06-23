# Using PYDAP LIBRARY to get following sonde data
from pydap.client import open_url
import sys;
import time;
import csv;
import pandas as pd
import numpy as np

#**************************************************
# Main Function
# Arguments: File path from TDS GLOS
# Retrieves data from TDS GLOST
# Saves the file in local folder
#**************************************************
def main(FilePath):
		dataset = open_url(FilePath)
		fieldnames = list(dataset.keys())
		var = np.arange(0,len(dataset['time']))
		print(var.shape)
		
		df = pd.DataFrame(index=var, columns = fieldnames) #
		for col in fieldnames:
			print(col)
			var = dataset[col]
			df[col] = var[:].data
	
		df.index.name = "index"
		fileName = FilePath.split('/')
		df.to_csv("Sondes_data/raw_data/"+fileName[len(fileName)-2]+".csv", columns = fieldnames, index=False)
		print("file created: " + (fileName[len(fileName)-2]))
		
		

if __name__ == "__main__":
	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/leavon/leavon.ncml')
	# main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/45164/45164.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/45165/45165.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/45165_1/45165_1.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/45167/45167.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/45169/45169.ncml')
	# main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/45176/45176.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/45176b/45176b.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/leash/leash.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/leash_1/leash.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/lebiww/lebiww.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/leelyria/leelyria.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/lelorain/lelorain.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/lementor/lementor.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/lementor_1/lementor.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/lemrbhd/lemrbhd.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/leoc/leoc.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/leoc_1/leoc.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/leorgn/leorgn.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/tolcrib/tolcrib.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/tollsps/tollsps.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/bgsusd2/bgsusd2.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/bgsusd/bgsusd.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/osugi/osugi.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/45176b/45176b.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/ESF3/ESF3.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/glerlwe13/glerlwe13.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/glerlwe4/glerlwe4.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/glerlwe2/glerlwe2.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/glerlwe8/glerlwe8.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/utlcp/utlcp.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/OMOECC_E1/OMOECC_E1.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/bgsdb/bgsdb.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/osuss/osuss.ncml')
# 	main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/sbedison/sbedison.ncml')
# 	
# 	
	
	