from emg_arff import getPath_arff,getPath_train,fd_store,storepick_arff,arffToData
import re, subprocess, sys
import datetime
import subprocess

def getWekaPath() :
	if sys.platform.startswith('win'):
		WEKA_PATH = '-classpath "C:\Program Files\Weka-3-7\weka.jar"' #for windows
	elif sys.platform.startswith('linux') :
		WEKA_PATH = '-classpath "/home/pakawinz/Workspace/weka.jar"' #for linux
	elif sys.platform.startswith('darwin'):
		WEKA_PATH = '-classpath "/Users/Gift/Downloads/meka-1.7.5/lib/weka.jar"' #for mac
	else:
		raise EnvironmentError('Unsupported class path')
	return WEKA_PATH

def calcMinMax(sample):
	sample = zip(*sample)
	if len(sample) != input_size :
		raise Exception("input_size and sample_size missmatch")
	minarray = map(np.min, sample)
	maxarray = map(np.max, sample)
	return minarray,maxarray

DATA = [[],[],[],[],[],[],[],[],[]]

def getStat_WEKA(resultString,convertToList=False,verbose=True):
	ACCU = re.search(r'Correctly Classified Instances\s+(\d+)\s+([\d.]+).*', resultString).group(2)
	EPE	 = ''
	MAE	 = re.search(r'Mean absolute error\s+([\d.]+).*', resultString).group(1)
	RMSE = re.search(r'Root mean squared error\s+([\d.]+).*', resultString).group(1)
	RAE	 = re.search(r'Relative absolute error\s+([\d.]+).*', resultString).group(1)
	RRSE = re.search(r'Root relative squared error\s+([\d.]+).*', resultString).group(1)

	table = re.search(r'=== Detailed Accuracy By Class ===\s+([\w\t -]+)\s+([\d.\s?]+).*', resultString)

	
	lines = re.split('\s+',table.group(2))
	for i,word in enumerate(lines):
		DATA[i%len(DATA)].append(word)

	listStat = [ACCU]+[EPE]+[MAE]			\
			+[RMSE]+[RAE]+[RRSE]			\
			+DATA[2]+DATA[3]+DATA[4]

	dictStat = {'ACC':ACCU,'EPE':EPE,'MAE':MAE,									\
				'RMS':RMSE,'RAE':RAE,'RRS':RRSE,	\
				'1PS':map(float,DATA[2]),'2RC':map(float,DATA[3]),'3FM':map(float,DATA[4])}

	if verbose :
		print '\t\t '+table.group(2)

	if convertToList : 
		return listStat
	else :
		return dictStat

readarray = lambda x,fn : map(fn,re.search(r'\[([^\]]+)]', x).group(1).split(','))

class WekaTrainer(object):

	W0 = 0 # initial bias weight

	def __init__(self,
		N_FOLD = 10,
		EPOCH = 500,
		MOMENTUM = 0.2,
		LEARNING_RATE = 0.3,
		HIDDEN1 = 'a',
		HIDDEN2 = None,
		NUMR_NORM = False,
		ATTR_NORM = True,
		):

		self.trained = False
		self.WEKA_PATH = getWekaPath()
		self.WEKA_CLASS = 'weka.classifiers.functions.MultilayerPerceptron'
		self.WEKA_OPTION = ' -L %.2f -M %.2f -N %d -x %d -V 0 -S 0 -E 20 -H %s%s -B -R -v %s%s'%(
				LEARNING_RATE,MOMENTUM,EPOCH,N_FOLD,HIDDEN1,
				',%s'%HIDDEN2 if HIDDEN2 else '',
				'-C' if not NUMR_NORM else '',
				' -I' if not ATTR_NORM else '',)

		self.hidden_size = []
		for HIDDEN in (HIDDEN1,HIDDEN2):
			if HIDDEN == 'a' :
				self.hidden_size.append(lambda x,y : (x+y)/2)
			elif HIDDEN == 'i' :
				self.hidden_size.append(lambda x,y : x)
			elif HIDDEN == 'o' :
				self.hidden_size.append(lambda x,y : y)
			elif HIDDEN == 't' :
				self.hidden_size.append(lambda x,y : x+y)
			elif type(HIDDEN) is int:
				self.hidden_size.append(lambda x,y : HIDDEN)

	def runWEKA(self,arfffile,statOnly=False):
		start = datetime.datetime.now()
		WEKA_CMD = " ".join(["java",self.WEKA_PATH,self.WEKA_CLASS,self.WEKA_OPTION,"-t",getPath_arff(arfffile)])
		if not statOnly : print WEKA_CMD
		run = subprocess.Popen(WEKA_CMD, stdout=subprocess.PIPE, shell=True)
		result = run.communicate()[0]

		if statOnly : return getStat_WEKA(result,True,False)
		
		stat = getStat_WEKA(result)
		stat['TIME'] = datetime.datetime.now() - start
		print "FINISH WEKA take time %s and Accuracy"%(stat['TIME'],stat['ACC'])
		# print result
		return result,stat

	def train(self,arfffile):
		afile = open(getPath_arff(arfffile),'r')
		self.minarray = readarray(afile.readline(),float)
		self.maxarray = readarray(afile.readline(),float)
		input_size,output_size = readarray(afile.readline(),int)
		afile.close()

		self.layerconfig = [input_size] + [fn(input_size,output_size) for fn in self.hidden_size] + [output_size]
		weights = [map(lambda x:[WekaTrainer.W0,1],range(input_size))] +\
			[map(lambda x:[],range(x)) for x in self.layerconfig[1:]]

		print "START WEKA"
		result,stat = self.runWEKA(arfffile)
		
		lines = result.splitlines()
		node_idx = 0
		getting_attr = False
		node_sorted = weights[-1] + weights[-2] + (weights[-3] if len(weights) == 4 else [])
		for line in lines:
			if getting_attr :
				if re.match(r'\s{2,}', line):
					try :
						res = re.search(r'[\w\-\.]+$', line)
						res = float(res.group(0))
						node_sorted[node_idx].append(res)
					except :
						pass
				else :
					node_idx += 1

			if re.match(r'Sigmoid Node',line ):
				getting_attr = True

			if node_idx == len(node_sorted) :
				break

		self.wekastat = stat
		self.weights = weights
		self.trained = True

	def loadTrained(self,filename):
		afile = open(getPath_train(filename),'r')
		self.minarray = readarray(afile.readline(),float)
		self.maxarray = readarray(afile.readline(),float)
		self.layerconfig = readarray(afile.readline(),int)
		weights = []
		for layer in self.layerconfig:
			weight = []
			for i in range(layer):
				weight.append(readarray(afile.readline(),float))
			weights.append(weight)

		self.weights = weights
		self.trained = True
		afile.close()

	def saveTrained(self,filename):
		if not self.trained :
			raise Exception("This trainer is not trained.")
		afile = open(getPath_train(filename),'w')
		afile.write("MIN %s\n"%self.minarray)
		afile.write("MAX %s\n"%self.maxarray)
		afile.write("CFG %s\n"%self.getLayerConfig())

		if len(self.weights) == 3 : layer_names = ['I','H','O']
		if len(self.weights) == 4 : layer_names = ['I','Hi','Ho','O']
		for layer,name in zip(self.weights,layer_names):
			count = 1
			for weight in layer:
				afile.write("%s%d %s\n"%(name,count,weight))
				count += 1
		afile.write("== Stat form WEKA ==\n")
		for key in sorted(self.wekastat):
			afile.write("%s\t%s\n"%(key, self.wekastat[key]))

		afile.close()

	def getLayerConfig(self):
		if not self.trained :
			raise Exception("This trainer is not trained.")
		return self.layerconfig

	def buildNetwork(self):
		from emg_ann import Network
		if not self.trained :
			raise Exception("This trainer is not trained.")

		if len(self.layerconfig) != 3 :
			raise Exception("The network is not support %d layers."%(len(self.layerconfig)))

		network = Network(*self.layerconfig)
		network.setMinMax(self.minarray, self.maxarray)
		network.setWeight(self.weights)

		return network


if __name__ == '__main__':

	trainer = WekaTrainer()
	filename = "150305"

	# fd_store(filename)
	# storepick_arff(0, filename)
	# trainer.train(filename)
	# trainer.saveTrained(filename)
	# trainer.loadTrained(filename)
	# network = trainer.buildNetwork()
	# supervised = arffToData(filename)

	# count = 0
	# for i in supervised:
	# 	after_input,after_hidden,after_output,result = network.activate(i[0],verbose=True)
	# 	# print after_input,result+1,i[1]

	# 	if result + 1 == i[1] : 
	# 		count += 1

	# print '%0.01f%%'%(count*100.0/len(supervised))