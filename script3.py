import sys
import re
import pandas as pd

def main(FilePath):
	fileName = FilePath.split('/')
	df = pd.read_csv(FilePath)
	
	Sondes_names= {	
	"leavon":	1,
	"45164":	2,
	"45176":	3,
	"45176b":	4,
	"45169":	5,
	"lementor_1":6,
	"leash_1":	7,
	"lelorain":	8,
	"leelyria":	9,
	"bgsdb":	10,
	"bgsusd":	11,
	"bgsusd2":	12,
	"lebiww":	13,
	"sbedison":	14,
	"osuss":	15,
	"lemrbhd":	16,
	"leoc_1":	17,
	"osugi":	18,
	"tolcrib":	19,
	"tollsps":	20,
	"45165_1":	21,
	"utlcp":	22,
	"leorgn":	23,
	"glerlwe2": 24,
	"glerlwe4": 25,
	"glerlwe8": 26,
	"glerlwe13": 27,
	"45167":	28,
	"ESF3":	29,
	"OMOECC_E1":	30,
	}
	
	Name = re.sub('.csv','',fileName[len(fileName)-1])	
	count = 0 
	for key,val in Sondes_names.items():
		if Name == key:
			df.to_csv('Sondes_data/data_organized/sonde'+str(val)+'.csv', index=False)
			count = count + 1
		
	if count ==0:
		print("Coudn't find the correct Key for: "+ Name)
    	
	
	
	
	
if __name__ == "__main__":

	if len(sys.argv) == 2:
		path  = sys.argv[1]
		main(path) 

	else:
		print("Please provide a correct path")
		print("Example: python script3.py Sondes_data/data_timecorrected/leavon.csv ")
		
		
		