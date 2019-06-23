import pandas as pd
import sys

def main(filePath, target):
	fileName = filePath.split('/')
	df = pd.read_csv(filePath)
	if target[0].lower() == 'dissolved_oxygen' or (len(target)>1 and target[1].lower() == 'dissolved_oxygen'):
		df.loc[df['dissolved_oxygen'] >2, 'DOcategory'] = 1
		df.loc[df['dissolved_oxygen'] >4, 'DOcategory'] = 2
		df.loc[df['dissolved_oxygen'] >5, 'DOcategory'] = 3
		df.loc[df['dissolved_oxygen'] <=2, 'DOcategory'] = 0
		
	if target[0].lower() == 'ph' or (len(target)>1 and target[1].lower() == 'ph'):
		df.loc[df['ph'] >7, 'pHcategory'] 	= 1
		df.loc[df['ph'] >7.2, 'pHcategory'] = 2
		df.loc[df['ph'] >7.4, 'pHcategory'] = 3
		df.loc[df['ph'] >7.6, 'pHcategory'] = 4
		df.loc[df['ph'] >7.8, 'pHcategory'] = 5
		df.loc[df['ph'] >8, 'pHcategory'] 	= 6
		df.loc[df['ph'] >8.2, 'pHcategory'] = 7
		df.loc[df['ph'] >8.4, 'pHcategory'] = 8
		df.loc[df['ph'] >8.6, 'pHcategory'] = 9
		df.loc[df['ph'] >8.8, 'pHcategory'] = 10
		df.loc[df['ph'] >9, 'pHcategory'] 	= 11
		df.loc[df['ph'] <=7, 'pHcategory']	= 0
	
	
	df.to_csv('Sondes_data/data_withCategories/'+fileName[len(fileName)-1],index= False)
	

	
	
if __name__ == "__main__":
	if len(sys.argv) >= 3:
		path  = sys.argv[1]
		target  = sys.argv[2:]		
		main(path, target) 

	else:
		print("Please provide a correct path")
		print("Example: python script5.py Sondes_data/data_corrected/sonde1.csv DO pH")
		
		