import numpy as np
import itertools

#Exp0 correct command /
# N_FOLD 	= [10]
# EPOCH 	= [50]
# MOMENTUM 	= [0.2]
# LEARNING_RATE = [0.3]
# HIDDEN0 	= [8]
# HIDDEN1 	= [0]

#Exp1 vary HIDDEN0, HIDDEN1 >>> test 1 or 2 Layer
N_FOLD 	= [10]
EPOCH 	= [500] 
MOMENTUM 	= [0.1]
LEARNING_RATE = [0.1]
HIDDEN0 	= [9,8,7,6,5,4,3,2,1]
# HIDDEN1 	= [7,6,5]

#Exp2 vary learning rate & momentum & hidden0
# N_FOLD 	= [10]
# EPOCH 	= [500] 
# MOMENTUM 	= [0.05,0.1,0.2,0.3,0.4,0.5]
# LEARNING_RATE = [0.05,0.1,0.2,0.3,0.4,0.5]
# HIDDEN0 	= []<<<<<<<<<<<<<<<<form Exp1
# HIDDEN1 	= []<<<<<<<<<<<<<<<<form Exp1

#Ex3 vary EPOCH
# N_FOLD 	= [10,20,50,100] 
# EPOCH 	= [500,1000,2000,5000,800,10000]
# MOMENTUM 	= []<<<<<<<<<<<<<<<<form Exp2
# LEARNING_RATE = []<<<<<<<<<<<<<<<<form Exp2
# HIDDEN0 	= []
# HIDDEN1 	= []

AUTOMATION0 = [N_FOLD,EPOCH,MOMENTUM,LEARNING_RATE,HIDDEN0,HIDDEN1]
AUTOMATION1 = [N_FOLD,EPOCH,MOMENTUM,LEARNING_RATE,HIDDEN0]

WEKA_PATH = '-classpath "/Users/Gift/Downloads/meka-1.7.5/lib/weka.jar"' #for mac
# WEKA_PATH = '-classpath "C:\Program Files\Weka-3-6\weka.jar"'#for windows
WEKA_CLASS = 'weka.classifiers.functions.MultilayerPerceptron'

W0 = 0
X0 = 1

def autoWEKA(exp,filename):
	import re
	import os
	import datetime

	fn = filename.split('.')[0].split('/')[1]	
	afile = open('3stat/'+exp+'/'+fn+'-'+exp+'.csv','a+')
	count = 0

	cartesian0 = list(itertools.product(*AUTOMATION0)) #Epx1
	cartesian1 = list(itertools.product(*AUTOMATION1))
	print "weka variation = %d"%len(cartesian)

	print "START WEKA"

	# for var in cartesian0 :
	for var in cartesian1 :
		# WEKA_OPTION = ' -L %.2f -M %.2f -N %d -V 0 -S 0 -E 20 -H %d,%d -B -C -R -v -x %d'%(var[3],var[2],var[1],var[4],var[5],var[0])
		WEKA_OPTION = ' -L %.2f -M %.2f -N %d -V 0 -S 0 -E 20 -H %d -B -C -R -v -x %d'%(var[3],var[2],var[1],var[4],var[0])

		WEKA_CMD = " ".join(["java",WEKA_PATH,WEKA_CLASS,WEKA_OPTION,"-t",filename])
		count += 1
		print count
		# print WEKA_CMD
		start = datetime.datetime.now()
		result = os.popen(WEKA_CMD).read()
		# print result
		lines = result.splitlines()

		ACCU = re.search(r'Correctly Classified Instances\s+(\d+)\s+([\d.]+).*', result)
		EPE	 = 0
		MAE	 = re.search(r'Mean absolute error\s+([\d.]+).*', result)
		RMSE = re.search(r'Root mean squared error\s+([\d.]+).*', result)
		RAE	 = re.search(r'Relative absolute error\s+([\d.]+).*', result)
		RAE	 = re.search(r'Relative absolute error\s+([\d.]+).*', result)
		RRSE = re.search(r'Root relative squared error\s+([\d.]+).*', result)

		table = re.search(r'=== Detailed Accuracy By Class ===\s+([\w\t -]+)\s+([\d.\s]+).*', result)
		lines = re.split('\s+',table.group(2))
		data = [[],[],[],[],[],[],[],[],[]]
		for i,word in enumerate(lines):
			data[i%9].append(word)

		# csvdata = ",".join(map(str,list(var)+[datetime.datetime.now() - start]+[ACCU.group(2)]+[]+[MAE.group(1)]+[RMSE.group(1)]+[RAE.group(1)]+[RRSE.group(1)]+[data[2]]+[data[3]]+[data[4]]+[WEKA_OPTION]))
		csvdata = ",".join(map(str,list(var)+[]+[datetime.datetime.now() - start]+[ACCU.group(2)]+[]+[MAE.group(1)]+[RMSE.group(1)]+[RAE.group(1)]+[RRSE.group(1)]+[data[2]]+[data[3]]+[data[4]]+[WEKA_OPTION]))
		afile.write(csvdata+'\n')
	
	afile.close()
	return 1

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

	def __init__(self,input_size,hidden_size,output_size,
			input_function = linearfn,
			hidden_function = sigmoidfn,
			output_function = sigmoidfn):
		super(Network, self).__init__()

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

if __name__ == '__main__':
	import itertools

	weights = readWEKA("result/recog 150216.arff")
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



