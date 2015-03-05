import numpy as np

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

		oldweight = map(lambda x: x.weights, self.inputNodes)
		rangearray = map(lambda x: float(x[0] - x[1])/2 , zip(maxarray,minarray))
		basearray = map(lambda x: float(x[0] + x[1])/2 , zip(maxarray,minarray))
		self.inputNodes = map(lambda x: Node(1,normalizefn(*x)), zip(rangearray, basearray))
		map(lambda x: x[0].setWeight(x[1]), zip(self.inputNodes,oldweight))
