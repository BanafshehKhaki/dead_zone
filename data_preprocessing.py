from pydap.client import open_url
import sys
import time
import csv
import pandas as pd
import numpy as np
import time
import datetime
import script8_NormalizeData as sc8
import os
from sklearn.model_selection import TimeSeriesSplit
import re
# **************************************************
# collect_dataset Function
# Arguments: File path from TDS GLOS
# Retrieves data from TDS GLOST
# Saves the file in local folder
# links: http://tds.glos.us/thredds/dodsC/buoy_agg_standard/leavon/leavon.ncml.html
# links: http://tds.glos.us/thredds/catalog/catalog.html
# **************************************************


def collect_dataset(FilePath):
    dataset = open_url(FilePath)
    fieldnames = list(dataset.keys())
    var = np.arange(0, len(dataset['time']))
    print(var.shape)

    df = pd.DataFrame(index=var, columns=fieldnames)
    for col in fieldnames:
        print(col)
        var = dataset[col]
        df[col] = var[:].data

    df.index.name = "index"
    fileName = FilePath.split('/')
    df.to_csv("Sondes_data/raw_data/" +
              fileName[len(fileName)-2]+".csv", columns=fieldnames, index=False)
    print("file created: " + (fileName[len(fileName)-2]))
    return df, fileName[len(fileName)-2]


# **************************************************
# time_Correction Function
# Arguments: df and file name
# Retrieves data and corrects the time in UTC
# Saves the file in local folder
# **************************************************


def time_Correction(df, fileName):
    df['time'] = df['time'] + 978321600.0
    df.to_csv("Sondes_data/data_timecorrected/" +
              fileName+'.csv', index=False)
    print("file created: " + (fileName))
    return df


def data_Cleaning(df, fileName):
    num = df._get_numeric_data()
    num[num < -100] = 0.0001
    df = df.dropna()
    df = df.drop(columns=['station_name'])
    df['time'] = [datetime.datetime.strptime(
        time.ctime(i), "%a %b %d %H:%M:%S %Y") for i in df['time']]
    df['month'] = [i.month for i in df['time']]
    df['day'] = [i.day for i in df['time']]
    df['hour'] = [i.hour for i in df['time']]
    df['year'] = [i.year for i in df['time']]
    df['time'] = pd.DatetimeIndex(df.time)
    df.set_index('time', inplace=True)
    df = df.resample('10T').mean().fillna(method='ffill')
    df.to_csv('Sondes_data/data_corrected/'+fileName+'.csv', index=True)
    return df


def hourly_Freq(df, fileName):
    df = df.drop(columns=['station_name'])
    df['time'] = [datetime.datetime.strptime(
        time.ctime(i), "%a %b %d %H:%M:%S %Y") for i in df['time']]
    df['month'] = [i.month for i in df['time']]
    df['day'] = [i.day for i in df['time']]
    df['hour'] = [i.hour for i in df['time']]
    df['year'] = [i.year for i in df['time']]
    df.set_index('time', inplace=True)
    df = df.resample('H').mean().fillna(method='ffill')
    df.to_csv('Sondes_data/data_withHourlyMean/' +
              fileName+'.csv', index=True)
    return df


def data_Categorize(df, fileName):
    
    df.loc[df['dissolved_oxygen'] > 2, 'DOcategory'] = 1  # Problematic
    # df.loc[df['dissolved_oxygen'] > 4, 'DOcategory'] = 2  # Trending
    df.loc[df['dissolved_oxygen'] > 4, 'DOcategory'] = 0  # All good
    df.loc[df['dissolved_oxygen'] <= 2, 'DOcategory'] = 2  # Dead Zone

    # pH for distribution:	< 6.9 ,  6.9-7.2 , 7.2-7.5 , 7.5-8.0 ,  > 8.0 
    df.loc[df['ph'] >= 7.9, 'pHcategory'] = 1  # Trending
    df.loc[df['ph'] > 8.2, 'pHcategory'] = 0  # OK
    df.loc[df['ph'] < 7.9, 'pHcategory'] = 2  # Bad
    df.to_csv('Sondes_data/data_withCategories/' +
              fileName+'.csv', index=True) #False if we just run this function with a our df
    return df


#####################################################################
# Main function: Create train and test file for a sonde using dates given
# Parameter: File path start and end date
# Returns: a test file in test_data > test_data , a train file in train_data > train_data for the given file
# - Uncomment to use a directory for specific dates
#####################################################################


def split_Test_Train(df, fileName, start_date, end_date, dataFOlder=''):

    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    df['time'] = pd.DatetimeIndex(df.time)
    df.set_index('time', inplace=True)
    index1 = df.index.get_loc(start_dt, "NEAREST")
    index2 = df.index.get_loc(end_dt, "NEAREST")

    df_test = df.loc[slice(df.index[index1], df.index[index2])]
    df_train = df.drop(df_test.index, axis=0)

    directory = 'Sondes_data/train'+dataFOlder+'/train_data/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    df_train.to_csv(directory +
                    fileName+"_wo_"+str(start_date)+"-"+str(end_date)+".csv")
    
    directory = 'Sondes_data/test'+dataFOlder+'/test_data/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    df_test.to_csv(directory +
                   fileName+"_"+str(start_date)+"-"+str(end_date)+'.csv')
    train_fileName = fileName+"_wo_"+str(start_date)+"-"+str(end_date)
    return df_train, df_test, train_fileName


def df_normalize(fileName, valSet=''):  # put train_val for validation data sets
    folderName = ''
    categories = ['test'] # remove  lorain data
    methods = ['diff_StandardScaler','StandardScaler']#,'MinMaxScaler', 'diff', 'diff_MinMaxScaler'] #'ewm', 'diff_ewm',
    
    targets = ['dissolved_oxygen', 'ph']
    for category in categories:
        categoryPath = category
        # if len(valSet)>1:
        #     valSet = re.sub('train',category,valSet)
        #     categoryPath = 'train'
        for method in methods:
            for target in targets:
                path = 'Sondes_data/'+categoryPath+'/'+categoryPath+'_data/'+valSet
                print(path)
                files = [f for f in os.listdir(path) if f.endswith(
                    ".csv") and f.startswith(fileName)]
                for file in files:
                    sc8.main(path+file, method, category,
                             target, folderName, valSet)


def TimeSeriesCVmain(train_data, fileName, FOlderName='allData'):
    print(train_data.head())
    i = 0
    tscv = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tscv.split(train_data):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = train_data.iloc[train_index], train_data.iloc[test_index]

        FilePath1 = 'Sondes_data/'+'train_data_'+FOlderName + \
            '/train_data/train_val/'+fileName+'_'+str(i+1)+'.csv'
        df_train_valid = pd.DataFrame(data=X_train, columns=train_data.columns)
        df_train_valid.to_csv(FilePath1, index=True)
        print(FilePath1)

        FilePath2 = re.sub('train_val/', 'test_val/', FilePath1)
        df_test_valid = pd.DataFrame(data=X_test, columns=train_data.columns)
        df_test_valid.to_csv(FilePath2, index=True)
        print(FilePath2)
        i += 1

import script10_1_TD_Prh_forNormalizedData as sc10_1

def makeTemporalHorizon(fileName, valSet=''):
    pd_names = [1,2,3,4,5,6,7,8,9,10,11,12]
    # folderName = 'allData'
    categories = ['train', 'test']
    methods = ['MinMaxScaler', 'StandardScaler', 'diff', 'ewm', 'diff_ewm',
               'diff_StandardScaler', 'diff_MinMaxScaler']
    targets = ['pHcategory', 'ph', 'DOcategory', 'dissolved_oxygen']

    for category in categories:
        categoryPath = category
        if len(valSet)>1:
            valSet = re.sub('train',category,valSet)
            categoryPath = 'train'
        for method in methods:
            for target in targets:
                path ='Sondes_data/'+categoryPath+'/'+categoryPath+'_data_normalized/'+method+'/'+target+'/'+ valSet
                files = [f for f in os.listdir(path) if f.endswith(".csv")and f.startswith(fileName)]
                for file in files:
                    counter =0
                    PrH_Steps = [1,2,3,4,5,6,7,8,9,10,11,12] 
                    td_steps  = 1
                    for prh_step in PrH_Steps:
                        pd_name = pd_names[counter]
                        counter = counter +1
                        sc10_1.main(path+file,prh_step, td_steps,pd_name, target)


if __name__ == "__main__":
    # df, fileName = collect_dataset('http://tds.glos.us/thredds/dodsC/buoy_agg_standard/lelorain/lelorain.ncml')
        # 'http://tds.glos.us/thredds/dodsC/buoy_agg_standard/leavon/leavon.ncml')
    # df = time_Correction(df, fileName)
    # df = data_Cleaning(df, fileName)   
    fileName= 'lelorain'
    # df = pd.read_csv('Sondes_data/data_corrected/'+fileName+'.csv')
    
    # df = data_Categorize(df, fileName)
    # df_train, df_test, train_fileName = split_Test_Train(
    #     df, fileName, '2019-07-01', '2020-01-15')
   
    df_normalize(fileName)  # for big train and test files
    # makeTemporalHorizon(fileName)

    # TimeSeriesCVmain(df_train, train_fileName)
    # df_normalize(fileName, 'train_val/')
    # makeTemporalHorizon(fileName,  'train_val/')
