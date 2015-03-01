from emg_automation import feature_extr,get_rawdata
from emg_autoweka import autoWEKA

def rawpick(pick,ctype,filename): # pick = constance of each number of record each type 
	arff = open('2arff/%s-%d.arff'%(filename,ctype),"a+")
	# head
	arff.write("@relation '%s'\n"%(filename))
	for x in xrange(1,9):
		arff.write("@attribute FD%d numeric\n"%x)

	if ctype == 0 :
		arff.write("@attribute State {1,2,3,4,5}\n\n")
	else :
		arff.write("@attribute Rest {0,1}\n")
		arff.write("@attribute Flex {0,1}\n")
		arff.write("@attribute Ext {0,1}\n")
		arff.write("@attribute Cir_Right {0,1}\n")
		arff.write("@attribute Cir_Left {0,1}\n\n")
	
	# data
	arff.write("@data\n")
	import glob
	stores = glob.glob('1store/*.csv')
	for store in stores:
		print store
		print pick
		lines = open(store,'r').readlines()
		for p in xrange(0,pick):
			arff.write(lines[p])
	arff.close()

	return 1


def rawtoarff(ctype,filename): #type : 0=one variable, 1=one hot
	elementFFT = [0,128,4,8,0]
	rawdata = get_rawdata('0raw/'+filename+'.txt')
	features = feature_extr(elementFFT, rawdata, False)

	store = open('1store/%s-%d.csv'%(filename,ctype),"a+")

	for fs,o in features :
		for f in fs :
			store.write("%f,"%(f))
		if ctype == 0 :
			store.write("%.00f\n"%(o))
		else :
			for x in xrange(1,6):
				if x == o and x == 5 :
					store.write("1\n")
				elif x == o and x != 5 :
					store.write("1,")
				elif x != o and x == 5 :
					store.write("0\n")
				elif x != o and x != 5 :
					store.write("0,")
	store.close()



if __name__ == '__main__':
	import glob
	ctype = 0

	#rawToStore /
	# txt = glob.glob('0raw/*.txt')
	# for x in txt:
	# 	filename = x.split('.')[0].split('/')[1] # for mac
	# 	# filename = x.split('.')[0].split('\\')[1] # for windows
	# 	rawtoarff(ctype, filename)

	#pickDataset /
	# picks = [5000,10000,15000,20000,25000,30000]
	# for pick in picks:
	# 	rawpick(pick/25,ctype,"data"+str(pick))
	
	#autoDataTest
	# autoWEKA('Exp0','2arff/data10000-0.arff')

	# autoWEKA('Exp1','2arff/data10000-0.arff')
	
	autoWEKA('Exp2','2arff/data10000-0.arff')
	# autoWEKA('Exp2','2arff/data15000-0.arff')
	# autoWEKA('Exp2','2arff/data20000-0.arff')
	# autoWEKA('Exp2','2arff/data25000-0.arff')
	# autoWEKA('Exp2','2arff/data30000-0.arff')
	# autoWEKA('Exp2','2arff/data35000-0.arff')
	# autoWEKA('Exp2','2arff/data5000-0.arff')


	# autoWEKA('Exp3','2arff/data10000-0.arff')
	# autoWEKA('Exp3','2arff/data15000-0.arff')
	# autoWEKA('Exp3','2arff/data20000-0.arff')
	# autoWEKA('Exp3','2arff/data25000-0.arff')
	# autoWEKA('Exp3','2arff/data30000-0.arff')
	# autoWEKA('Exp3','2arff/data35000-0.arff')
	# autoWEKA('Exp3','2arff/data5000-0.arff')

