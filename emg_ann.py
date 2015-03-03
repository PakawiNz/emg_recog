from emg_arff import getPath_arff,getPath_train,fd_store,storepick_arff
from emg_autoweka import getWekaPath,getStat_WEKA
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
		start = datetime.datetime.now()
		WEKA_CMD = " ".join(["java",self.WEKA_PATH,self.WEKA_CLASS,self.WEKA_OPTION,"-t",getPath_arff(arfffile)])
		print WEKA_CMD
		result = os.popen(WEKA_CMD).read()
		lines = result.splitlines()
		# print result
		
		self.wekastat = getStat_WEKA(result)
		self.wekastat['TIME'] = datetime.datetime.now() - start
		print "FINISH WEKA take time %s"%(self.wekastat['TIME'])
		print "FINISH WEKA with Accuracy : %s%%"%(self.wekastat['ACC'])
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
		afile.write("== Stat form WEKA ==\n")
		for key in sorted(self.wekastat):
			afile.write("%s\t%s\n"%(key, self.wekastat[key]))

		afile.close()

	def getLayerConfig(self):
		if not self.trained :
			raise Exception("This trainer is not trained.")
		return self.layerconfig

	def buildNetwork(self):
		if not self.trained :
			raise Exception("This trainer is not trained.")

		if len(self.layerconfig) != 3 :
			raise Exception("The network is not support %d layers."%(len(self.layerconfig)))

		network = Network(*self.layerconfig)
		network.setWeight(self.weights)
		network.setMinMax(self.minarray, self.maxarray)

		return network

	@staticmethod
	def calcMinMax(sample):
		sample = zip(*sample)
		if len(sample) != input_size :
			raise Exception("input_size and sample_size missmatch")
		minarray = map(np.min, sample)
		maxarray = map(np.max, sample)
		return minarray,maxarray

if __name__ == '__main__':

	trainer = WekaTrainer()
	# filename = "150206"
	filename = "data5000"

	# # supervised = fd_store( filename)
	# storepick_arff(5000, filename)
	trainer.train(filename)
	trainer.saveTrained(filename)
	# trainer.loadTrained(filename)
	# network = trainer.buildNetwork()

	# supervised = [([21.118532,62.999746,12.684460,10.519438,10.387631,11.685056,62.565823,19.993782],5),
	# 	([19.310054,64.152326,13.212357,8.860584,8.858590,11.498716,64.569127,18.682508],5),
	# 	([16.992022,64.612842,9.747459,7.517548,7.257055,8.316456,63.423046,17.723756],5),
	# 	([18.766140,63.760157,14.643641,11.581990,11.067794,14.170338,63.207139,18.051383],5),
	# 	([24.611953,65.072881,9.259757,7.909290,7.459348,9.559294,64.841156,21.564185],5),
	# 	([18.568221,66.322171,10.374216,11.080600,11.160580,9.379371,66.122006,16.911144],5),
	# 	([21.881888,65.216293,13.490270,10.054914,9.735708,12.226527,65.474979,21.068340],5),
	# 	([23.189213,63.061800,11.986820,10.701041,10.135923,11.014461,64.207857,21.266626],5),
	# 	([19.578129,64.249013,14.982886,10.110107,9.541771,14.604983,65.091028,16.698812],5),
	# 	([25.803094,65.136486,9.737080,9.920804,9.835623,9.691144,64.626364,22.955705],5),
	# 	([21.060543,61.647437,8.773530,10.652985,11.005020,7.922391,61.748437,18.933209],5),]

	# # # trainner.train(arfffile)
	# # # recog = Network(8, 6, 5, normalize=test)
	# # # recog.setWeight(trainner.weights)

	# count = 0
	# for i in supervised:
	# 	after_input,after_hidden,after_output,result = network.activate(i[0],verbose=True)

	# 	if result + 1 == i[1] : 
	# 		count += 1

	# print '%0.01f%%'%(count*100.0/len(supervised))