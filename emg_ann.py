from emg_arff import getPath_arff,getPath_train,fd_store,storepick_arff
import numpy as np
import re, os
import datetime

add = lambda x,y=0 : x+y
mul = lambda x,y=1 : x*y

class Node(object):

	X0 = 1 # default bias value

	def __init__(self,input_size,actfunc,init_weight=0):
		super(Node, self).__init__()

		self.weights = [init_weight]*(input_size+1)
		self.actfunc = actfunc

	def activate(self,data):
		if type(data) not in (list,tuple):
			data = [data]
		if len(data) != len(self.weights)-1:
			raise Exception("node_input_size and data_size missmatch")

		sum_result = reduce(add,map(lambda x:mul(*x),zip([Node.X0]+data,self.weights)))
		return self.actfunc(sum_result)

	def setWeight(self,weights):
		if type(weights) not in (list,tuple):
			weights = [weights]
		if len(weights) != len(self.weights):
			raise Exception("node_input_size and weight_size missmatch")

		self.weights = weights
		return

normalizefn = lambda r,b : (lambda x: float(x - b)/r) if r != 0 else (lambda x: float(x - b))
linearfn = lambda x: x
sigmoidfn = lambda x: 1/(1+np.exp(-x))

class Network(object):

	def __init__(self,input_size,hidden_size,output_size,
			input_function = linearfn,
			hidden_function = sigmoidfn,
			output_function = sigmoidfn,):
		super(Network, self).__init__()

		self.inputNodes = map(lambda x: Node(1,input_function), range(input_size))
		self.hiddenNodes = 	map(lambda x: Node(input_size,hidden_function), range(hidden_size))
		self.outputNodes = 	map(lambda x: Node(hidden_size,output_function), range(output_size))

	def convertToMotion(self,alist):
		idx = max(range(len(alist)), key=lambda i: alist[i])
		return idx

	def activate(self,data,verbose=False):
		if len(data) != len(self.inputNodes):
			raise Exception("network_input_size and data_size missmatch")

		# after_input = data
		after_input = map(lambda x: x[0].activate(x[1]), zip(self.inputNodes,data))
		after_hidden = map(lambda x: x.activate(after_input), self.hiddenNodes)
		after_output = map(lambda x: x.activate(after_hidden), self.outputNodes)

		if verbose :
			return after_input,after_hidden,after_output,self.convertToMotion(after_output)
		else :
			return self.convertToMotion(after_output)

	def setWeight(self,weights):
		map(lambda x: x[0].setWeight(x[1]), zip(self.inputNodes,weights[0]))
		map(lambda x: x[0].setWeight(x[1]), zip(self.hiddenNodes,weights[1]))
		map(lambda x: x[0].setWeight(x[1]), zip(self.outputNodes,weights[2]))

	def setMinMax(self,minarray,maxarray):
		if len(minarray) != len(self.inputNodes) or len(maxarray) != len(self.inputNodes):
			raise Exception("network_input_size and min_max_size missmatch")

		rangearray = map(lambda x: float(x[0] - x[1])/2 , zip(maxarray,minarray))
		basearray = map(lambda x: float(x[0] + x[1])/2 , zip(maxarray,minarray))
		self.inputNodes = map(lambda x: Node(1,normalizefn(*x)), zip(rangearray, basearray))

readarray = lambda x,fn : map(fn,re.search(r'\[([^\]]+)]', x).group(1).split(','))

class WekaTrainer(object):

	W0 = 0 # initial bias weight

	def __init__(self,
		LEARNING_RATE = 0.3,
		MOMENTUM = 0.2,
		EPOCH = 500,
		N_FOLD = 10,
		NUMR_NORM = False,
		ATTR_NORM = True,
		HIDDEN1 = 'a',
		HIDDEN2 = None,
		):

		self.trained = False
		self.WEKA_PATH = '-classpath "C:\Program Files\Weka-3-6\weka.jar"'
		self.WEKA_CLASS = 'weka.classifiers.functions.MultilayerPerceptron'
		self.WEKA_OPTION = ' -L %.2f -M %.2f -N %d -x %d -V 0 -S 0 -E 20 -H %s%s -B -v %s %s'%(
				LEARNING_RATE,MOMENTUM,EPOCH,N_FOLD,HIDDEN1,
				',%s'%HIDDEN2 if HIDDEN2 else '',
				'-C' if not NUMR_NORM else '',
				'-I' if not ATTR_NORM else '',)

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


	def train(self,arfffile):
		afile = open(getPath_arff(0,arfffile),'r')
		self.minarray = readarray(afile.readline(),float)
		self.maxarray = readarray(afile.readline(),float)
		input_size,output_size = readarray(afile.readline(),int)
		afile.close()

		self.layerconfig = [input_size] + [fn(input_size,output_size) for fn in self.hidden_size] + [output_size]

		weights = [map(lambda x:[WekaTrainer.W0,1],range(input_size))] +\
			[map(lambda x:[],range(x)) for x in self.layerconfig[1:]]

		print "START WEKA"
		start = datetime.datetime.now()
		WEKA_CMD = " ".join(["java",self.WEKA_PATH,self.WEKA_CLASS,self.WEKA_OPTION,"-t",getPath_arff(0,arfffile)])
		print WEKA_CMD
		result = os.popen(WEKA_CMD).read()
		lines = result.splitlines()
		# print result
		accu = re.search(r'Correctly Classified Instances\s+(\d+)\s+([\d.]+).*', result)
		print "FINISH WEKA with Accuracy %s%%"%(accu.group(2))
		print "FINISH WEKA take time %s"%(datetime.datetime.now() - start)

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

		afile.close()

	def getLayerConfig(self):
		if not self.trained :
			raise Exception("This trainer is not trained.")
		return self.layerconfig

	@staticmethod
	def calcMinMax(sample):
		sample = zip(*sample)
		if len(sample) != input_size :
			raise Exception("input_size and sample_size missmatch")
		minarray = map(np.min, sample)
		maxarray = map(np.max, sample)
		return minarray,maxarray

if __name__ == '__main__':
	supervised = fd_store(0, "150206")
	storepick_arff(5000, 0, "150206")
	
	trainer = WekaTrainer()
	trainer.train("150206")
	trainer.saveTrained("150206")
	# trainer.loadTrained("150206")