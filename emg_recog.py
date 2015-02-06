# from emg_train import MotionModel
# import numpy as np

# motion = MotionModel(MotionModel.FILE_FLEX)
# motion = motion.load()

# def z_calc(x,mean,stdev) :
# 	result = -np.square(x-mean) / (2.0 * np.square(stdev))
# 	result = np.exp(result) #/ (stdev * np.sqrt(2 * np.pi))
# 	return result

# def recognize(fft_result) :
# 	FREQ_DOMAIN = len(fft_result)
# 	result = 0
# 	# use normal distribution equation
# 	for i in range(FREQ_DOMAIN) :
# 		z_score = z_calc(fft_result[i], motion.average[i], motion.stdev[i])
# 		result += z_score
# 		# print z_score

# 	result /= FREQ_DOMAIN
# 	result = 0 if result < 0.1 else result

# 	return result

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer,RPropMinusTrainer
import time

class Recognition(object):
	"""docstring for Recognition"""
	def __init__(self,freq_domain):
		super(Recognition, self).__init__()
		self._ds = SupervisedDataSet(freq_domain, 1)
		self._net = buildNetwork(freq_domain, freq_domain*3/4, freq_domain*2/4, freq_domain*1/4, 1)
		self._trainer = BackpropTrainer(self._net, self._ds)

	def addSample(self,features,activity):
		self._ds.addSample(features,activity)

	def training(self,epochs,update=None):
		for i in range(epochs):
			self._trainer.train()
			if update :
				update(float(i*100)/epochs)

	def recognize(self,features):
		print self._net.activate(features)
