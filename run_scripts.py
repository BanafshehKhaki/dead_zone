import os
# scripts 1 - 9 : File Preparations
import script1 as sc1
import script2 as sc2
import script3 as sc3
import script4 as sc4
import script5 as sc5
import script6 as sc6
import script7 as sc7
import script8 as sc8


# import script10 as sc10




#####################################################################
# Script 1 gets the data from TDS GLOS server
# Parameter: Link to the TDS GLOS server for each Buoy
# Returns: Saves the data in Sondes_data > raw_data
# - Uncomment each line for the specific buoy you need:
#####################################################################

# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/leavon/leavon.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/45164/45164.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/45165/45165.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/45165_1/45165_1.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/45167/45167.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/45169/45169.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/45176/45176.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/45176b/45176b.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/leash/leash.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/leash_1/leash.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/lebiww/lebiww.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/leelyria/leelyria.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/lelorain/lelorain.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/lementor/lementor.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/lementor_1/lementor.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/lemrbhd/lemrbhd.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/leoc/leoc.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/leoc_1/leoc.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/leorgn/leorgn.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/tolcrib/tolcrib.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/tollsps/tollsps.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/bgsusd2/bgsusd2.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/bgsusd/bgsusd.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/osugi/osugi.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/45176b/45176b.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/ESF3/ESF3.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/glerlwe13/glerlwe13.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/glerlwe4/glerlwe4.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/glerlwe2/glerlwe2.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/glerlwe8/glerlwe8.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/utlcp/utlcp.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/OMOECC_E1/OMOECC_E1.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/bgsdb/bgsdb.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/osuss/osuss.ncml')
# sc1.main('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/sbedison/sbedison.ncml')



#####################################################################
# Script 2 changes TDS GLOS time to current time UTC
# Parameter: Link to the GLOS local files directory saved from script1
# Returns: Saves the data in Sondes_data > data_timecorrected
# - Uncomment to send the directory where all raw data are saved
#name of files:
				# leavon.csv
				# sbedison.csv
				# osuss.csv
				# bgsdb.csv
				# OMOECC_E1.csv
				# utlcp.csv
				# glerlwe8.csv
				# glerlwe2.csv
				# glerlwe4.csv
				# glerlwe13.csv
				# ESF3.csv
				# 45176b.csv
				# osugi.csv
				# bgsusd.csv
				# bgsusd2.csv
				# tollsps.csv
				# tolcrib.csv
				# leorgn.csv
				# leoc_1.csv
				# leoc.csv
				# lemrbhd.csv
				# lementor_1.csv
				# lementor.csv
				# lelorain.csv
				# leelyria.csv
				# lebiww.csv
				# leash_1.csv
				# leash.csv
				# 45176.csv
				# 45169.csv
				# 45167.csv
				# 45165_1.csv
				# 45165.csv
				# 45164.csv
#####################################################################

# path = 'Sondes_data/raw_data/'
# files = [f for f in os.listdir(path) if f.endswith(".csv")]
# for file in files:
# 	sc2.main(path+file)

# sc2.main('Sondes_data/raw_data/leavon.csv')


#####################################################################
# Script 3 : Changing file names
# Parameter: File path 
# Returns: File Name
# - Uncomment the loop to use a directory or just uncomment the one line for a file
# NOTE: 45165, lementor, leash and leoc are not used as they are not updated
#####################################################################
# path = 'Sondes_data/data_timecorrected/' 
# files = [f for f in os.listdir(path) if f.endswith(".csv")]
# for file in files:
# 	sc3.main(path+file)

# sc3.main('Sondes_data/data_timecorrected/lementor_1.csv')
# sc3.main('Sondes_data/data_timecorrected/45165_1.csv')
# sc3.main('Sondes_data/data_timecorrected/leoc_1.csv')
# sc3.main('Sondes_data/data_timecorrected/leavon.csv')



#####################################################################
# Script 4 : Remove wrong values 
# Parameter: File path method for normalization
# Returns: a file in data_corrected
# - Uncomment the line for a file
# NOTE: Currently only checks for Sonde 1 
#####################################################################

# sc4.main('Sondes_data/data_organized/sonde1.csv')


#####################################################################
# Script 5 : Create Category files 
# Parameter: File path 
# Returns: a 
# - Uncomment the loop to use a directory or just uncomment the one line for a file
#####################################################################

# path = 'Sondes_data/data_corrected/'
# files = [f for f in os.listdir(path) if f.endswith(".csv")]
# for file in files:
# 	sc5.main(path+file,['ph'])

# sc5.main('Sondes_data/data_corrected/sonde1.csv',['dissolved_oxygen','ph'])



#####################################################################
# Script 6 : Create train and test file for a sonde using dates given
# Parameter: File path start and end date 
# Returns: a test file in test_data > test_data , a train file in train_data > train_data for the given file
# - Uncomment the loop to use a directory or just uncomment the one line for a file
#####################################################################

# start_date = '2018-06-01'
# end_date   = '2018-07-01' 
# path = 'Sondes_data/data_withCategories/'
# 
# files = [f for f in os.listdir(path) if f.endswith(".csv")]
# for file in files:
# 	sc6.main(path+file, start_date, end_date)

# sc6.main('Sondes_data/data_withCategories/sonde1.csv', '2018-06-01', '2018-07-01')




#####################################################################
# Script 7 : Normalize train and test data
# Parameter: File path method for normalization
# Returns: a test file in test_data > test_data_normalized >method , a train file in train_data > train_data_normalized>method for the given file
# - Uncomment the loop to use a directory or just uncomment the one line for a file
#####################################################################

# trainpath = 'Sondes_data/train_data/train_data/'
# method ='SS'
# category = 'train'
# target = 'dissolved_oxygen'
# files = [f for f in os.listdir(trainpath) if f.endswith(".csv")]
# for file in files:
# 	sc7.main(trainpath+file,method,category)

# testpath = 'Sondes_data/test_data/test_data/'
# method ='SS'
# category = 'test'
# target = 'dissolved_oxygen'
# files = [f for f in os.listdir(testpath) if f.endswith(".csv")]
# for file in files:
# 	sc7.main(testpath+file,method,category, target)

# sc7.main('Sondes_data/train_data/train_data/sonde1_wo_2018-06-01-2018-07-01.csv','MM','train', 'ph')
# sc7.main('Sondes_data/test_data/test_data/sonde1_2018-06-01-2018-07-01.csv','MM','test', 'ph')
# sc7.main('Sondes_data/train_data/train_data/sonde1_wo_2018-06-01-2018-07-01.csv','MM','train', 'dissolved_oxygen')
# sc7.main('Sondes_data/test_data/test_data/sonde1_2018-06-01-2018-07-01.csv','MM','test', 'dissolved_oxygen')



#####################################################################
# Script 8 : PCA whiten data
# Parameter: File path method 
# Returns: an uncorrelated data stored in whiten_data folder
# - Uncomment the loop to use a directory or just uncomment the one line for a file
#####################################################################

# trainpath = 'Sondes_data/train_data/train_data_normalized/StandardScaler/dissolved_oxygen/'
# files = [f for f in os.listdir(trainpath) if f.endswith(".csv")]
# for file in files:
# 	sc8.main(trainpath+file)

# testpath = 'Sondes_data/test_data/test_data_normalized/StandardScaler/dissolved_oxygen/'
# files = [f for f in os.listdir(testpath) if f.endswith(".csv")]
# for file in files:
# 	sc8.main(testpath+file)
	
# sc8.main('Sondes_data/train_data/train_data_normalized/StandardScaler/ph/sonde1_wo_2018-06-01-2018-07-01.csv')
# sc8.main('Sondes_data/test_data/test_data_normalized/StandardScaler/ph/sonde1_wo_2018-06-01-2018-07-01.csv')



#####################################################################
# Script 9 : Temporal depth and Prediction Horizon files from whiten files
# Parameter: File path method 
# Returns: an uncorrelated data stored in whiten_data folder
# - Uncomment the loop to use a directory or just uncomment the one line for a file
#####################################################################


# pd_names = [1,3,5,6,7,12,24,36,144,288,1008]
# td 1  :[1,3,5,6,7,12,24,36,144,288,1008] 
# td 3  :[3,5,7,8,9,14,26,38,146,290,1010] 
# td 5  :[5,7,9,10,11,16,28,40,148,292,1012] 
# td 6  :[6,8,10,11,12,17,29,41,149,293,1013] 
# td 7  :[7,9,11,12,13,18,30,42,150,294,1014] 
# td 12 :[12,14,16,17,18,23,35,47,155,299,1019] 
# td 24 :[24,26,28,29,30,35,47,59,167,311,1031] 
# td 36 :[36,38,40,41,42,47,59,71,179,323,1043]

# dissolved_oxygen_test_files_path  = 'Sondes_data/test_data/test_data_normalized/StandardScaler/dissolved_oxygen/whiten/'
# ph_test_files_path  = 'Sondes_data/test_data/test_data_normalized/StandardScaler/ph/whiten/'
# dissolved_oxygen_train_files_path = 'Sondes_data/train_data/train_data_normalized/StandardScaler/dissolved_oxygen/whiten/'
# ph_train_files_path =  'Sondes_data/train_data/train_data_normalized/StandardScaler/ph/whiten/'
# 
# path = dissolved_oxygen_test_files_path
# pd_names = [36]
# files = [f for f in os.listdir(path) if f.endswith(".csv")]
# for file in files: 
# 	counter =0
# 	PrH_Steps = [36]
# 	td_steps  = 1
# 	for prh_step in PrH_Steps:
# 		pd_name = pd_names[counter]
# 		counter = counter +1
# 		sc9.main(path+file,prh_step, td_steps,pd_name)
# 
# 	counter =0
# 	PrH_Steps = [38]
# 	td_steps  = 3
# 	for prh_step in PrH_Steps:
# 		pd_name = pd_names[counter]
# 		counter = counter +1
# 		sc9.main(path+file,prh_step, td_steps, pd_name)
# 
# 	counter =0
# 	PrH_Steps = [40]
# 	td_steps  = 5
# 	for prh_step in PrH_Steps:
# 		pd_name = pd_names[counter]
# 		counter = counter +1
# 		sc9.main(path+file,prh_step, td_steps, pd_name)
# 
# 	counter =0
# 	PrH_Steps = [41]
# 	td_steps  = 6
# 	for prh_step in PrH_Steps:
# 		pd_name = pd_names[counter]
# 		counter = counter +1
# 		sc9.main(path+file,prh_step, td_steps, pd_name)
# 
# 	counter =0
# 	PrH_Steps = [42]
# 	td_steps  = 7
# 	for prh_step in PrH_Steps:
# 		pd_name = pd_names[counter]
# 		counter = counter +1
# 		sc9.main(path+file,prh_step, td_steps, pd_name)
# 
# 	counter =0
# 	PrH_Steps = [47]
# 	td_steps  = 12
# 	for prh_step in PrH_Steps:
# 		pd_name = pd_names[counter]
# 		counter = counter +1
# 		sc9.main(path+file,prh_step, td_steps, pd_name)
# 		
# 		
# 	counter =0
# 	PrH_Steps = [59]
# 	td_steps  = 24
# 	for prh_step in PrH_Steps:
# 		pd_name = pd_names[counter]
# 		counter = counter +1
# 		sc9.main(path+file,prh_step, td_steps, pd_name)
# 	
# 	
# 	counter =0
# 	PrH_Steps = [71]
# 	td_steps  = 36
# 	for prh_step in PrH_Steps:
# 		pd_name = pd_names[counter]
# 		counter = counter +1
# 		sc9.main(path+file,prh_step, td_steps, pd_name)
# 		


##################################################
##### Script 9_1 variation to script 9 for TD and PrH for Non-Whiten data
#####
##################################################
import script9_1 as sc9_1
# prh_step = 3
# td_steps = 1
# pd_name = 3
# target = 'pHcategory'
# scalerFolder = 'MinMaxScaler'
# sc9_1.main('Sondes_data/train_data/train_data_normalized/'+scalerFolder+'/'+target+'/sonde1_wo_2018-06-01-2018-07-01.csv', prh_step, td_steps, pd_name,target)
# sc9_1.main('Sondes_data/test_data/test_data_normalized/'+scalerFolder+'/'+target+'/sonde1_2018-06-01-2018-07-01.csv',prh_step, td_steps, pd_name, target)
# 
# pd_names = [1,3,5,6,7,12,24,36,144,288,1008]
# td 1  :[1,3,5,6,7,12,24,36,144,288,1008] 
# td 3  :[3,5,7,8,9,14,26,38,146,290,1010] 
# td 5  :[5,7,9,10,11,16,28,40,148,292,1012] 
# td 6  :[6,8,10,11,12,17,29,41,149,293,1013] 
# td 7  :[7,9,11,12,13,18,30,42,150,294,1014] 
# td 12 :[12,14,16,17,18,23,35,47,155,299,1019] 
# td 24 :[24,26,28,29,30,35,47,59,167,311,1031] 
# td 36 :[36,38,40,41,42,47,59,71,179,323,1043]

# dissolved_oxygen_test_files_path  = 'Sondes_data/test_data/test_data_normalized/StandardScaler/dissolved_oxygen/whiten/'
# ph_test_files_path  = 'Sondes_data/test_data/test_data_normalized/StandardScaler/ph/whiten/'
# dissolved_oxygen_train_files_path = 'Sondes_data/train_data/train_data_normalized/StandardScaler/dissolved_oxygen/whiten/'
# ph_train_files_path =  'Sondes_data/train_data/train_data_normalized/StandardScaler/ph/whiten/'

# # 
# for target in ['pHcategory', 'ph', 'DOcategory', 'dissolved_oxygen']:
# 	trainpath ='Sondes_data/train_data/train_data_normalized/'+scalerFolder+'/'+target+'/'
# 	testpath = 'Sondes_data/test_data/test_data_normalized/'+scalerFolder+'/'+target+'/'
# 	trainfile = 'sonde1_wo_2018-06-01-2018-07-01.csv'
# 	testfile = 'sonde1_2018-06-01-2018-07-01.csv'
# 
# 	counter =0
# 	PrH_Steps = [1,3,5,6,7,12,24,36,144,288,1008] 
# 	td_steps  = 1
# 	for prh_step in PrH_Steps:
# 		pd_name = pd_names[counter]
# 		counter = counter +1
# 		sc9_1.main(trainpath+trainfile,prh_step, td_steps,pd_name, target)
# 		sc9_1.main(testpath+testfile,prh_step, td_steps,pd_name, target)
 	
# 	counter =0
# 	PrH_Steps = [3,5,7,8,9,14,26,38,146,290,1010] 
# 	td_steps  = 3
# 	for prh_step in PrH_Steps:
# 		pd_name = pd_names[counter]
# 		counter = counter +1
# 		sc9_1.main(trainpath+trainfile,prh_step, td_steps,pd_name, target)
# 		sc9_1.main(testpath+testfile,prh_step, td_steps,pd_name, target)
# 		
# 	counter =0
# 	PrH_Steps = [5,7,9,10,11,16,28,40,148,292,1012] 
# 	td_steps  = 5
# 	for prh_step in PrH_Steps:
# 		pd_name = pd_names[counter]
# 		counter = counter +1
# 		sc9_1.main(trainpath+trainfile,prh_step, td_steps,pd_name, target)
# 		sc9_1.main(testpath+testfile,prh_step, td_steps,pd_name, target)
# 		
# 	counter =0
# 	PrH_Steps = [6,8,10,11,12,17,29,41,149,293,1013] 
# 	td_steps  = 6
# 	for prh_step in PrH_Steps:
# 		pd_name = pd_names[counter]
# 		counter = counter +1
# 		sc9_1.main(path+file,prh_step, td_steps, pd_name, target)
# 
# 	counter =0
# 	PrH_Steps = [7,9,11,12,13,18,30,42,150,294,1014] 
# 	td_steps  = 7
# 	for prh_step in PrH_Steps:
# 		pd_name = pd_names[counter]
# 		counter = counter +1
# 		sc9_1.main(trainpath+trainfile,prh_step, td_steps,pd_name, target)
# 		sc9_1.main(testpath+testfile,prh_step, td_steps,pd_name, target)
# 		
# 	counter =0
# 	PrH_Steps = [12,14,16,17,18,23,35,47,155,299,1019] 
# 	td_steps  = 12
# 	for prh_step in PrH_Steps:
# 		pd_name = pd_names[counter]
# 		counter = counter +1
# 		sc9_1.main(trainpath+trainfile,prh_step, td_steps,pd_name, target)
# 		sc9_1.main(testpath+testfile,prh_step, td_steps,pd_name, target)		
# 		
# 	counter =0
# 	PrH_Steps = [24,26,28,29,30,35,47,59,167,311,1031] 
# 	td_steps  = 24
# 	for prh_step in PrH_Steps:
# 		pd_name = pd_names[counter]
# 		counter = counter +1
# 		sc9_1.main(trainpath+trainfile,prh_step, td_steps,pd_name, target)
# 		sc9_1.main(testpath+testfile,prh_step, td_steps,pd_name, target)	
# 	
# 	counter =0
# 	PrH_Steps = [36,38,40,41,42,47,59,71,179,323,1043]
# 	td_steps  = 36
# 	for prh_step in PrH_Steps:
# 		pd_name = pd_names[counter]
# 		counter = counter +1
# 		sc9_1.main(trainpath+trainfile,prh_step, td_steps,pd_name, target)
# 		sc9_1.main(testpath+testfile,prh_step, td_steps,pd_name, target)		



#####################################################################
# Script make_cv : create cross validation sets
# Parameter: File path method 
# Returns: 
# - Uncomment the loop to use a directory or just uncomment the one line for a file
#####################################################################
# import make_cv as cv;
# 
# cv.main('Sondes_data/train_data/train_data_normalized/MinMaxScaler/dissolved_oxygen/PrH_TD/sonde1_wo_2018-06-01-2018-07-01_TD1_PrH1.csv')
# 
# trainpath = 'Sondes_data/train_data/train_data_normalized/MinMaxScaler/dissolved_oxygen/PrH_TD/'
# files = [f for f in os.listdir(trainpath) if f.endswith(".csv")]
# for file in files:
# 	cv.main(trainpath+file)

#####################################################################
# Script 10 : Stats, accl, jerk .. 
# Parameter: File path method 
# Returns: 
# - Uncomment the loop to use a directory or just uncomment the one line for a file
#####################################################################





##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
## MODELS :
#####################################################################
# Script 11 : Linear Regression 
# Parameter: File path method 
# Returns: 
# - Uncomment the loop to use a directory or just uncomment the one line for a file
#####################################################################
































