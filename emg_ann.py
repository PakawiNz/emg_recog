import numpy as np
import re, os
import datetime

def convertToMotion(alist):
	idx = max(range(len(alist)), key=lambda i: alist[i])
	return idx

def add(x,y=0):
	return x + y

def mul(x,y=1):
	return x * y

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
			output_function = sigmoidfn,
			normalize = False):
		super(Network, self).__init__()

		if normalize :
			sample = zip(*normalize)
			if len(sample) != input_size :
				raise Exception("input_size and sample_size missmatch")
			minarray = map(np.min, sample)
			maxarray = map(np.max, sample)
			rangearray = map(lambda x: float(x[0] - x[1])/2 , zip(maxarray,minarray))
			basearray = map(lambda x: float(x[0] + x[1])/2 , zip(maxarray,minarray))

			self.inputNodes = map(lambda x: Node(1,normalizefn(*x)), zip(rangearray, basearray))
		else :
			self.inputNodes = map(lambda x: Node(1,input_function), range(input_size))

		self.hiddenNodes = 	map(lambda x: Node(input_size,hidden_function), range(hidden_size))
		self.outputNodes = 	map(lambda x: Node(hidden_size,output_function), range(output_size))

	def activate(self,data,verbose=False):
		if len(data) != len(self.inputNodes):
			raise Exception("network_input_size and data_size missmatch")

		# after_input = data
		after_input = map(lambda x: x[0].activate(x[1]), zip(self.inputNodes,data))
		after_hidden = map(lambda x: x.activate(after_input), self.hiddenNodes)
		after_output = map(lambda x: x.activate(after_hidden), self.outputNodes)

		if verbose :
			return after_input,after_hidden,after_output
		else :
			return after_output

	def setWeight(self,weights):
		map(lambda x: x[0].setWeight(x[1]), zip(self.inputNodes,weights[0]))
		map(lambda x: x[0].setWeight(x[1]), zip(self.hiddenNodes,weights[1]))
		map(lambda x: x[0].setWeight(x[1]), zip(self.outputNodes,weights[2]))

class WekaTrainer(object):

	W0 = 0 # initial bias weight

	def __init__(self,
		LEARNING_RATE = 0.3,
		MOMENTUM = 0.2,
		EPOCH = 500,
		N_FOLD = 10,
		NUMR_NORM = False,
		ATTR_NORM = True,
		):

		self.WEKA_PATH = '-classpath "C:\Program Files\Weka-3-6\weka.jar"'
		self.WEKA_CLASS = 'weka.classifiers.functions.MultilayerPerceptron'
		self.WEKA_OPTION = ' -L %.2f -M %.2f -N %d -x %d -V 0 -S 0 -E 20 -H a -B -v %s %s'%(
				LEARNING_RATE,MOMENTUM,EPOCH,N_FOLD,
				'-C' if not NUMR_NORM else '',
				'-I' if not ATTR_NORM else '')

	def train(self,filename):
		print "START WEKA"
		start = datetime.datetime.now()
		WEKA_CMD = " ".join(["java",self.WEKA_PATH,self.WEKA_CLASS,self.WEKA_OPTION,"-t",filename])
		print WEKA_CMD
		result = os.popen(WEKA_CMD).read()
		lines = result.splitlines()
		# print result
		accu = re.search(r'Correctly Classified Instances\s+(\d+)\s+([\d.]+).*', result)
		print "FINISH WEKA with Accuracy %s%% take %s"%(accu.group(2))
		print "FINISH WEKA take time %s"%(datetime.datetime.now() - start)

		weights = [
			map(lambda x:[WekaTrainer.W0,1],range(8)),
			map(lambda x:[],range(6)),
			map(lambda x:[],range(5)),
		]
		node_sorted = weights[2] + weights[1]
		node_idx = 0
		getting_attr = False
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

		return weights


strFloatList = lambda z : "[%s]"%", ".join(map(lambda x :"%.03f"%x,z))

if __name__ == '__main__':
	# import itertools
	# test = list(itertools.product(range(1,6),range(6),range(5),range(4),range(3),range(2),range(1),range(1)))
	# arfffile = 'result/stupid.arff'
	# supervised = map(lambda x: (x,x[0]), test)

	from emg_arff import rawtoarff,rawpick
	supervised = rawtoarff(0, "150206")
	arfffile = rawpick(5000, 0, "150206")
	test = zip(*supervised)[0]

	trainner = WekaTrainer()
	weights = trainner.train(arfffile)
	recog = Network(8, 6, 5, normalize=test)
	recog.setWeight(weights)

	count = 0
	for i in supervised:
		after_input,after_hidden,after_output = recog.activate(map(lambda x: x ,i[0]),verbose=True)
		# print strFloatList(after_input),strFloatList(after_hidden),strFloatList(after_output),
		# print strFloatList(after_output),

		result = after_output
		# print convertToMotion(result) + 1, i[1]
		if convertToMotion(result) + 1 == i[1] : 
			count += 1

	print '%0.01f%%'%(count*100.0/len(supervised))
