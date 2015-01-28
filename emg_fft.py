import numpy as np
import threading

CALC_SIZE = 128
SLIDING_SIZE = 16
BUFFER_SIZE = 4

FFT_NORM_CORRECTION = 0.54
DATA_SIZE = 1024
FREQ_DOMAIN = 16
FREQ_GROUPING = map(lambda a: (a*CALC_SIZE/FREQ_DOMAIN,(a+1)*CALC_SIZE/FREQ_DOMAIN),range(FREQ_DOMAIN))

class FeatureExtractor(object) :
	
	def __init__(self) :
		self.inBuffer = [0]*CALC_SIZE*BUFFER_SIZE
		self.header = 0
		self.counter = 0

	def gather(self,data) :
		adj_data = (data-DATA_SIZE/2.0)/DATA_SIZE
		target = self.header + self.counter

		self.inBuffer[target] = adj_data
		if target >= CALC_SIZE * (BUFFER_SIZE-1) :
			self.inBuffer[target - CALC_SIZE * (BUFFER_SIZE-1)] = adj_data

		self.counter += 1

		if self.counter == CALC_SIZE :
			self.counter -= SLIDING_SIZE
			readyCalc = True
		else :
			readyCalc = False

		if readyCalc :
			result = self.calc(self.inBuffer,self.header)

			self.header += SLIDING_SIZE
			if self.header + CALC_SIZE == CALC_SIZE * BUFFER_SIZE :
				self.header = 0

			return result

	def calc(self,raw_data,header) :
		out = np.fft.fft(raw_data[header:header+CALC_SIZE])

		result = map(FeatureExtractor.norm ,out)

		result = map(lambda a: reduce(lambda x,y: x+y,result[a[0]:a[1]])*100 , FREQ_GROUPING)

		return result

	@staticmethod
	def norm(a):
		return np.sqrt((np.square(a.real) + np.square(a.imag))/(CALC_SIZE*FFT_NORM_CORRECTION))
