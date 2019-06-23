import pandas as pd
import sys
from sklearn.decomposition import PCA
import re

def main(filePath):
	fileName = filePath.split('/')
	df = pd.read_csv(filePath)
	new_df = df.drop(columns=['station_name','time','depth','lat','lon','DOcategory','pHcategory','dissolved_oxygen_saturation'])
# 	new_df = new_df.drop(new_df.columns[len(new_df.columns)-1], axis=1)
	pca = PCA(whiten = True)
	
	transform_X = pca.fit_transform(new_df)
	

	columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5','feature_6']
	transform_df = pd.DataFrame.from_records(transform_X)
	transform_df.columns = columns


	transform_df['pHcategory'] = df['pHcategory'].values
	transform_df['DOcategory'] = df['DOcategory'].values
	transform_df['Target'] = df[df.columns[len(df.columns)-1]].values
	
	filePath = re.sub('DO/','DO/whiten/',filePath)
	filePath = re.sub('pH/','pH/whiten/',filePath)
	
	transform_df.to_csv(filePath,index= False)
	

	
	
if __name__ == "__main__":
	if len(sys.argv) == 2:
		path  = sys.argv[1]
		main(path) 

	else:
		print("Please provide a correct path")
		print("Example: python script8.py Sondes_data/train_data/train_data_normalized/StandardScaler/DO/sonde1_wo_2018-06-01-2018-07-01.csv")
		
		