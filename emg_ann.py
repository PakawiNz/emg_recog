import numpy as np

LEARNING_RATE = 0.3
MOMENTUM = 0.2
EPOCH = 500
N_FOLD = 10

WEKA_PATH = '-classpath "C:\Program Files\Weka-3-6\weka.jar"'
WEKA_CLASS = 'weka.classifiers.functions.MultilayerPerceptron'
WEKA_OPTION = ' -L %.2f -M %.2f -N %d -V 0 -S 0 -E 20 -H a -B -C -v -I -x %d'%(
		LEARNING_RATE,MOMENTUM,EPOCH,N_FOLD)

W0 = 0
X0 = 1

def readWEKA(filename):
	import re
	import os
	import datetime
	
	print "START WEKA"
	start = datetime.datetime.now()
	WEKA_CMD = " ".join(["java",WEKA_PATH,WEKA_CLASS,WEKA_OPTION,"-t",filename])
	print WEKA_CMD
	result = os.popen(WEKA_CMD).read()
	lines = result.splitlines()
	accu = re.search(r'Correctly Classified Instances\s+(\d+)\s+([\d.]+).*', result)
	print "FINISH WEKA with Accuracy %s%% take %s"%(accu.group(2),datetime.datetime.now() - start)

	weights = [
		map(lambda x:[W0,1],range(8)),
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

OUTPUTSIZE = 5
def convertToActivation(x):
	result = [0]*OUTPUTSIZE
	result[int(x)-1] = 1
	return result

def convertToMotion(alist):
	idx = max(range(len(alist)), key=lambda i: alist[i])
	return idx

def add(x,y=0):
	return x + y

def mul(x,y=1):
	return x * y

class Node(object):
	def __init__(self,input_size,actfunc,init_weight=0):
		super(Node, self).__init__()

		self.weights = [init_weight]*(input_size+1)
		self.actfunc = actfunc

	def activate(self,data):
		if type(data) not in (list,tuple):
			data = [data]
		if len(data) != len(self.weights)-1:
			raise Exception("node_input_size and data_size missmatch")

		sum_result = reduce(add,map(lambda x:mul(*x),zip([X0]+data,self.weights)))
		return self.actfunc(sum_result)

	def setWeight(self,weights):
		if type(weights) not in (list,tuple):
			weights = [weights]
		if len(weights) != len(self.weights):
			raise Exception("node_input_size and weight_size missmatch")

		self.weights = weights
		return

linearfn = lambda x: x
sigmoidfn = lambda x: 1/(1+np.exp(-x))

class Network(object):

	def __init__(self,input_size,hidden_size,output_size):
		super(Network, self).__init__()

		input_function = linearfn
		hidden_function = sigmoidfn
		output_function = sigmoidfn

		self.inputNodes = 	map(lambda x: Node(1,input_function), range(input_size))
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

	recognize = activate

# print WEKA_CMD

if __name__ == '__main__':
	import itertools

	weights = readWEKA("result/stupid.arff")
	test = list(itertools.product(range(1,6),range(6),range(5),range(4),range(3),range(2),range(1),range(1)))
	test = map(lambda x: (x,x[0]), test)

	recog = Network(8, 6, 5)
	recog.setWeight(weights)

	count = 0
	for i in test:
		result = recog.recognize(i[0])
		# print result, convertToMotion(result) + 1, i[1]
		if convertToMotion(result) + 1 == i[1] : 
			count += 1

	# print "\n".join(map(str ,recog.recognize(test[0][0],True)))
	print count*100.0/len(test)