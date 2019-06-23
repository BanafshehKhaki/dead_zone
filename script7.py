import sys, time, csv, pandas as pd, numpy as np, os, sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import re

def standardizeData(df, file, category, target):
    file = re.sub('.csv', '', file)
    if category.lower() == 'train':
        if target.lower() == 'dissolved_oxygen':
            SavePath = 'Sondes_data/train_data/train_data_normalized/StandardScaler/dissolved_oxygen/'
            y_scaler = StandardScaler()
            y_scale = y_scaler.fit(df[['dissolved_oxygen']].values)
            y_scaler_filename = SavePath + file + '_StandardScaler_y.save'
            joblib.dump(y_scale, y_scaler_filename)
            y_normalized = y_scale.transform(df[['dissolved_oxygen']].values)
            df = df.drop(columns=['dissolved_oxygen'])
            scaler_filename = SavePath + file + '_StandardScaler_X.save'
        else:
            if target.lower() == 'ph':
                SavePath = 'Sondes_data/train_data/train_data_normalized/StandardScaler/ph/'
                y_scaler = StandardScaler()
                y_scale = y_scaler.fit(df[['ph']].values)
                y_scaler_filename = SavePath + file + '_StandardScaler_y.save'
                joblib.dump(y_scale, y_scaler_filename)
                y_normalized = y_scale.transform(df[['ph']].values)
                df = df.drop(columns=['ph'])
                scaler_filename = SavePath + file + '_StandardScaler_X.save'
        scaler = StandardScaler()
        scale = scaler.fit(df.values)
        joblib.dump(scale, scaler_filename)
        normalized = scale.transform(df.values)
    else:
        if category.lower() == 'test':
            file = re.sub('_', '_wo_', file)
            if target.lower() == 'dissolved_oxygen':
                SavePath = 'Sondes_data/train_data/train_data_normalized/StandardScaler/dissolved_oxygen/'
                y_scale = joblib.load(SavePath + file + '_StandardScaler_y.save')
                y_normalized = y_scale.transform(df[['dissolved_oxygen']].values)
                df = df.drop(columns=['dissolved_oxygen'])
                scaler = joblib.load(SavePath + file + '_StandardScaler_X.save')
                SavePath = 'Sondes_data/test_data/test_data_normalized/StandardScaler/dissolved_oxygen/'
            else:
                if target.lower() == 'ph':
                    SavePath = 'Sondes_data/train_data/train_data_normalized/StandardScaler/ph/'
                    y_scale = joblib.load(SavePath + file + '_StandardScaler_y.save')
                    y_normalized = y_scale.transform(df[['ph']].values)
                    df = df.drop(columns=['ph'])
                    scaler = joblib.load(SavePath + file + '_StandardScaler_X.save')
                    SavePath = 'Sondes_data/test_data/test_data_normalized/StandardScaler/ph/'
            normalized = scaler.transform(df.values)
    return (normalized, y_normalized, SavePath)


def minMaxNormalizeData(df, file, category, target):
    file = re.sub('.csv', '', file)
    if category.lower() == 'train':
    
        if target.lower() == 'dissolved_oxygen':
            SavePath = 'Sondes_data/train_data/train_data_normalized/MinMaxScaler/dissolved_oxygen/'
            y_scaler = MinMaxScaler(feature_range=(0, 1))
            y_scale = y_scaler.fit(df[['dissolved_oxygen']].values)
            y_scaler_filename = SavePath + file + '_MinMaxScaler_y.save'
            joblib.dump(y_scale, y_scaler_filename)
            y_normalized = y_scale.transform(df[['dissolved_oxygen']].values)
            df = df.drop(columns=['dissolved_oxygen'])
            scaler_filename = SavePath + file + '_MinMaxScaler_X.save'
        else:
            if target.lower() == 'ph':
                SavePath = 'Sondes_data/train_data/train_data_normalized/MinMaxScaler/ph/'
                y_scaler = MinMaxScaler(feature_range=(0, 1))
                y_scale = y_scaler.fit(df[['ph']].values)
                y_scaler_filename = SavePath + file + '_MinMaxScaler_y.save'
                joblib.dump(y_scale, y_scaler_filename)
                y_normalized = y_scale.transform(df[['ph']].values)
                df = df.drop(columns=['ph'])
                scaler_filename = SavePath + file + '_MinMaxScaler_X.save'
                
                
        scaler = MinMaxScaler(feature_range=(0, 1))
        scale = scaler.fit(df.values)
        joblib.dump(scale, scaler_filename)
        normalized = scale.transform(df.values)
    else:
        if category.lower() == 'test':
            file = re.sub('_', '_wo_', file)
            if target.lower() == 'dissolved_oxygen':
                SavePath = 'Sondes_data/train_data/train_data_normalized/MinMaxScaler/dissolved_oxygen/'
                y_scale = joblib.load(SavePath + file + '_MinMaxScaler_y.save')
                y_normalized = y_scale.transform(df[['dissolved_oxygen']].values)
                df = df.drop(columns=['dissolved_oxygen'])
                scaler = joblib.load(SavePath + file + '_MinMaxScaler_X.save')
                SavePath = 'Sondes_data/test_data/test_data_normalized/MinMaxScaler/dissolved_oxygen/'
            else:
                if target.lower() == 'ph':
                    SavePath = 'Sondes_data/train_data/train_data_normalized/MinMaxScaler/ph/'
                    y_scale = joblib.load(SavePath + file + '_MinMaxScaler_y.save')
                    y_normalized = y_scale.transform(df[['ph']].values)
                    df = df.drop(columns=['ph'])
                    scaler = joblib.load(SavePath + file + '_MinMaxScaler_X.save')
                    SavePath = 'Sondes_data/test_data/test_data_normalized/MinMaxScaler/ph/'
                    
                    
            normalized = scaler.transform(df.values)
    return (normalized, y_normalized, SavePath)


def main(FilePath, method='SS', category='train', target='dissolved_oxygen'):
    fileName = FilePath.split('/')
    df = pd.read_csv(FilePath)
    newdf_x = df.drop(['time', 'depth', 'lat', 'station_name', 'lon', 'DOcategory', 'pHcategory'], axis=1)
    
    
    if method == 'MM':
        norm_x, norm_y, SavePath = minMaxNormalizeData(newdf_x, fileName[(len(fileName) - 1)], category, target)
    elif method == 'SS':
        norm_x, norm_y, SavePath = standardizeData(newdf_x, fileName[(len(fileName) - 1)], category, target)
            
            
    if target.lower() == 'dissolved_oxygen':
        newdf_y = pd.DataFrame(data=norm_y, columns=['dissolved_oxygen'])
        newdf_x = newdf_x.drop(columns=['dissolved_oxygen'])
    else:
        if target.lower() == 'ph':
            newdf_y = pd.DataFrame(data=norm_y, columns=['ph'])
            newdf_x = newdf_x.drop(columns=['ph'])
            
    newdf_x = pd.DataFrame(data=norm_x, columns=(newdf_x.columns))
    
    
    frame = [df[['station_name', 'time', 'depth', 'lat', 'lon', 'DOcategory', 'pHcategory']], newdf_x, newdf_y]
    newdf = pd.concat(frame, axis=1)
    newdf.to_csv((SavePath + fileName[(len(fileName) - 1)]), index=False)


if __name__ == '__main__':
    if len(sys.argv) == 5:
        path = sys.argv[1]
        method = sys.argv[2]
        category = sys.argv[3]
        target = sys.argv[4]
        if method == 'MM' or method == 'SS':
            if category.lower() == 'test' or category.lower() == 'train':
                if target.lower() == 'dissolved_oxygen' or target.lower() == 'ph':
                    main(path, method, category, target)
                else:
                    print('target: dissolved_oxygen or ph')
            else:
                print('Category: train or test')
        else:
            print('Current available methods: MM or SS')
    else:
        print('\n ******************************************************')
        print('Please provide <File path> <Normalization Method> <test || train data> <dependant variables: dissolved_oxygen || ph>\n')
        print('Normalization methods:  MinMaxScaler = MM \n')
        print('Standardization method with mean zero and standard deviation of one = SS \n')
        print('\n ******************************************************\n')