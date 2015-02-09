from pybrain import structure as STRUCT
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer,RPropMinusTrainer
import time

OUTPUTSIZE = 6

class Recognition(object):
	"""docstring for Recognition"""
	def __init__(self,freq_domain):
		super(Recognition, self).__init__()
		self._ds = SupervisedDataSet(freq_domain, OUTPUTSIZE)
		self._net = buildNetwork(freq_domain, freq_domain*2, OUTPUTSIZE)
		self._trainer = BackpropTrainer(self._net, self._ds)

	def addSample(self,features,activity):
		self._ds.addSample(features,Recognition.convertToActivation(activity))

	def validate(self):
		return 100 - float(reduce(lambda x,y : x+y ,map(self.__errorCount, range(self._ds.getLength())))) / self._ds.getLength() * 100

	def training(self,epochs,update=None):
		results = []
		for i in range(epochs):
			result = self._trainer.train()
			results.append(result)
			if update :
				update(float(i*100)/epochs,result)
		return results

	def recognize(self,features):
		return self._net.activate(features)

	def __errorCount(self,idx):
		sample = self._ds.getSample(idx)
		truth = Recognition.convertToMotion(sample[1])
		recog = Recognition.convertToMotion(self.recognize(sample[0]))
		if truth == recog : return 0
		else : return 1

	@staticmethod
	def buildTrianingSet(features):
		dummy = Recognition(len(features[0][0]))
		map(lambda x : dummy.addSample(*x),features)
		return dummy._ds

	@staticmethod
	def convertToActivation(x):
		result = [0]*OUTPUTSIZE
		result[int(x)] = 1
		return result

	@staticmethod
	def convertToMotion(alist):
		idx = max(range(len(alist)), key=lambda i: alist[i])
		return idx

class CustomRecognition(Recognition):
	"""docstring for CustomRecognition"""
	def __init__(self,freq_domain,InputLayer,InputConnecton,OutputLayer,OutputConnection,HiddenLayer,HiddenSize,NetworkType,TrainerType,reusedDataSet=None):
		super(CustomRecognition, self).__init__(freq_domain)

		inputLayer = InputLayer(freq_domain)
		hiddenLayer = HiddenLayer(freq_domain*2)
		outputLayer = OutputLayer(OUTPUTSIZE)

		in_to_hi = InputConnecton(inputLayer, hiddenLayer)
		hi_to_out = OutputConnection(hiddenLayer, outputLayer)

		net = NetworkType()
		net.addInputModule(inputLayer)
		net.addModule(hiddenLayer)
		net.addOutputModule(outputLayer)
		net.addConnection(in_to_hi)
		net.addConnection(hi_to_out)
		net.sortModules()

		if reusedDataSet :
			self._ds = reusedDataSet
		trainer = TrainerType(net, self._ds)

		self._net = net
		self._trainer = trainer

test = [([1,2,3],1),
		([2,2,3],2),
		([3,2,3],3),
		([4,2,3],4),
		([5,2,3],5),]
		
if __name__ == '__main__':
	for i in range(10):
		network,inlayer,hidlayer,outlayer = (STRUCT.FeedForwardNetwork,STRUCT.LinearLayer,STRUCT.MultiplicationLayer,STRUCT.SoftmaxLayer)
		recog = CustomRecognition(3,inlayer, STRUCT.FullConnection, outlayer, STRUCT.FullConnection, hidlayer, 0, network, BackpropTrainer)
		map(lambda x : recog.addSample(*x),test)
		err = recog.training(10)
		acc = recog.validate()
		res = recog.recognize(test[0][0])
		rec = Recognition.convertToMotion(res)

		text = "\t".join(['>'*10,"%.3f"%err,"%.3f"%acc,"%d"%rec,] + map(lambda x: "%.3f"%x,res))
		print text
