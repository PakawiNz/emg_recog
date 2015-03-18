from emg_utils import getPath_arff,getPath_csv,getPath_raw
import re,sys,glob

def arffToData(filename,inline=False):
	arff = open(getPath_arff(filename),"r").read()
	data = re.search(r'@data\s+([\d.,\s]+).*', arff).group(1).splitlines()
	data = map(lambda x : x.split(','),data)
	data = map(lambda x : map(float, x),data)
	if not inline :
		data = map(lambda x : (x[:-1],x[-1]), data)

	return data

def fd_store(filename,ctype=0): #type : 0=one variable, 1=one hot
	from emg_fft import get_supervised_fd
	elementFFT = [0,128,4,8,0]
	rawdata = get_supervised_td(getPath_raw(filename))
	features = get_supervised_fd(elementFFT, rawdata, False)

	outfile = getPath_csv(filename)
	store = open(outfile,"w")

	for fs,o in features :
		for f in fs :
			store.write("%0.6f,"%(f))
		if ctype == 0 :
			store.write("%d\n"%(o))
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

def storepick_arff(pick,filename,ctype=0): # pick = constance of each number of record each type 
	# gather csv
	data = []
	if pick == 0 :
		pick = 1000000
		stores = glob.glob(getPath_csv(filename))
	else :
		stores = glob.glob(getPath_csv('*'))

	for store in stores:
		print store
		print pick
		lines = open(store,'r').readlines()
		for p in xrange(0,min(pick,len(lines))):
			data.append(lines[p])

		fd_seperate = zip(*map(lambda x: map(float,x.split(',')[:-1]), data))
		minfd = map(min,fd_seperate)
		maxfd = map(max,fd_seperate)

	# write header
	input_size = len(minfd)
	output_size = 5

	arfffile = getPath_arff( filename)
	arff = open(arfffile,"w")
	arff.write("%% min: %s \n"%(minfd))
	arff.write("%% max: %s \n"%(maxfd))
	arff.write("%% i/o: %s \n"%([input_size,output_size]))
	
	arff.write("@relation '%s'\n"%(filename))
	for x in xrange(1,input_size+1):
		arff.write("@attribute FD%d numeric\n"%x)

	if ctype == 0 :
		arff.write("@attribute State {%s}\n\n"%(
			",".join(map(str,range(1,output_size+1))))) # {1,2,3,4,5}
	else :
		arff.write("@attribute Rest {0,1}\n")
		arff.write("@attribute Flex {0,1}\n")
		arff.write("@attribute Ext {0,1}\n")
		arff.write("@attribute Cir_Right {0,1}\n")
		arff.write("@attribute Cir_Left {0,1}\n\n")
	
	# write data
	arff.write("@data\n")

	for line in data:
		arff.write('%s'%line)

	arff.close()

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

def getFlienameFromPaths(pathList):
	for path in pathList:
		if sys.platform.startswith('win'):
			sign = '\\' # for windows
		elif sys.platform.startswith('darwin') or sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
			sign = '/' # for mac
		else:
			raise EnvironmentError('Unsupported class path')
		filename = map(lambda x : x.split('.')[0].split(sign)[-1],pathList)
	return filename	

if __name__ == '__main__':

	#rawToStore /
	# import glob
	# pathList = glob.glob('0raw/*.txt')
	# for filename in getFlienameFromPaths(pathList):
	# 	fd_store(filename)


	#pickDataset /
	# picks = [5000,10000,15000,20000,25000,30000]
	# picks = [500,1000,1500,2000,2500,3000,3500,4000]
	# for pick in picks:
	# 	storepick_arff(pick/25,"data"+str(pick))
	# storepick_arff(10,"datatestpick"+str(10))

	# arffToData /
	# arffToData('data5000')

	pass

