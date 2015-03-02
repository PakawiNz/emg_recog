from emg_autoweka import autoWEKA
from emg_fft import get_supervised_fd
import re,time

def getPath_raw(filename):
	return '0raw/%s.txt'%(filename)

def getPath_csv(ctype,filename):
	return '1store/%s-%d.csv'%(filename,ctype)

def getPath_arff(ctype,filename):
	return '2arff/%s-%d.arff'%(filename,ctype)

def getPath_train(filename):
	return '4train/%s.txt'%(filename)

def storepick_arff(pick,ctype,filename): # pick = constance of each number of record each type 
	arfffile = getPath_arff(ctype, filename)
	arff = open(arfffile,"w")
	data = []
	import glob
	stores = glob.glob('1store/*.csv')
	for store in stores:
		print store
		print pick
		lines = open(store,'r').readlines()
		for p in xrange(0,min(pick,len(lines))):
			data.append(lines[p])

		fd_seperate = zip(*map(lambda x: map(float,x.split(',')[:-1]), data))
		minfd = map(min,fd_seperate)
		maxfd = map(max,fd_seperate)

	# head
	arff.write("%% min: %s \n"%(minfd))
	arff.write("%% max: %s \n"%(maxfd))
	
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

	for line in data:
		arff.write('%s\n'%line)


	arff.close()

	return arfffile

def fd_store(ctype,filename): #type : 0=one variable, 1=one hot
	elementFFT = [0,128,4,8,0]
	rawdata = get_supervised_td(getPath_raw(filename))
	features = get_supervised_fd(elementFFT, rawdata, False)

	outfile = getPath_csv(ctype,filename)
	store = open(outfile,"w")

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
	return features

def __tuplation(x):
	try :
		if ',' in x :
			return tuple(map(float,x.split(',')))
		else :
			return float(x)
	except :
		return None

def get_supervised_td(filename):
	lines = open(filename,'r').readlines()
	result = []
	for line in lines :
		line = re.split(r'\s+',line)
		result += map(__tuplation,line)
	return result

if __name__ == '__main__':
	import glob
	ctype = 0

	#rawToStore /
	# txt = glob.glob('0raw/*.txt')
	# for x in txt:
	# 	filename = x.split('.')[0].split('/')[1] # for mac
	# 	# filename = x.split('.')[0].split('\\')[1] # for windows
	# 	fd_store(ctype, filename)

	#pickDataset /
	# picks = [5000,10000,15000,20000,25000,30000]
	# for pick in picks:
	# 	storepick_arff(pick/25,ctype,"data"+str(pick))
	storepick_arff(10,ctype,"datatestpick"+str(10))
	
	#autoDataTest
	# autoWEKA('Exp0','2arff/data10000-0.arff')

	# autoWEKA('Exp1','2arff/data10000-0.arff')
	
	# autoWEKA('Exp2','2arff/data10000-0.arff')
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

