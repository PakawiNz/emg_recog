import numpy as np
import threading
import math

UI_RANGE = 500
DATA_RANGE = 1024
BUFFER_SIZE = 4
FFT_NORM_CORRECTION = 0.54

class FeatureExtractor(object) :

	DEFAULT_CALC_SIZE = 128
	DEFAULT_SLIDING_SIZE = 16
	DEFAULT_FREQ_DOMAIN = 16
	DEFAULT_TREND_CHUNK = 20

	OUTPUT_RAW = 0
	OUTPUT_DIFF = 1
	OUTPUT_TREND = 2
	
	def __init__(self,
		OUTPUT_TYPE = OUTPUT_RAW,
		CALC_SIZE = DEFAULT_CALC_SIZE,
		SLIDING_SIZE = DEFAULT_SLIDING_SIZE,
		FREQ_DOMAIN = DEFAULT_FREQ_DOMAIN,
		TREND_CHUNK = DEFAULT_TREND_CHUNK) :

		self.OUTPUT_TYPE = OUTPUT_TYPE
		self.CALC_SIZE = CALC_SIZE
		self.SLIDING_SIZE = SLIDING_SIZE
		self.FREQ_DOMAIN = FREQ_DOMAIN
		self.TREND_CHUNK = TREND_CHUNK

		self.TREND_GROUPING = range(self.TREND_CHUNK)
		self.FREQ_GROUPING = map(lambda a: 
			(a*self.CALC_SIZE/self.FREQ_DOMAIN,(a+1)*self.CALC_SIZE/self.FREQ_DOMAIN),
			range(self.FREQ_DOMAIN))

		self.clearBuffer()

	def clearBuffer(self) :
		self.inBuffer = [0]*self.CALC_SIZE*BUFFER_SIZE
		self.header = 0
		self.counter = 0
		self.TAIL = self.CALC_SIZE * (BUFFER_SIZE-1)
		self.TAIL += (self.SLIDING_SIZE - self.TAIL % self.SLIDING_SIZE)

		if self.OUTPUT_TYPE == FT.OUTPUT_TREND :
			self.tempResult = []
		elif self.OUTPUT_TYPE == FT.OUTPUT_DIFF :
			self.tempResult = None

	def gather(self,data) :
		adj_data = (data-DATA_RANGE/2.0)/DATA_RANGE
		target = self.header + self.counter
		# print self.header, self.counter, target, target - self.TAIL, self.TAIL

		self.inBuffer[target] = adj_data
		if target >= self.TAIL :
			self.inBuffer[target - self.TAIL] = adj_data

		self.counter += 1

		if self.counter == self.CALC_SIZE :
			self.counter -= self.SLIDING_SIZE
			readyCalc = True
		else :
			readyCalc = False

		result = None
		if readyCalc :
			# print "CALC %s"%self.header
			result = self.calc(self.inBuffer,self.header)
			self.header += self.SLIDING_SIZE

		if target == len(self.inBuffer) - 1 :
			self.header = 0

		return result

	def calc(self,raw_data,header) :
		out = np.fft.fft(raw_data[header:header+self.CALC_SIZE])

		result = map(self.norm ,out)
		result = map(lambda a: reduce(lambda x,y: x+y,result[a[0]:a[1]]) , self.FREQ_GROUPING)
		result = map(lambda a: a * 100						# SCALING
			* self.FREQ_DOMAIN 	/ FT.DEFAULT_FREQ_DOMAIN 	# STRAIGHT ADJUST
			/ self.CALC_SIZE 	* FT.DEFAULT_CALC_SIZE 		# REVERSED ADJUST
			,result)

		if self.OUTPUT_TYPE == FT.OUTPUT_RAW :
			output = result

		elif self.OUTPUT_TYPE == FT.OUTPUT_DIFF :
			output = None
			if self.tempResult :
				output = map(lambda x: x[0]-x[1],zip(result,self.tempResult))
			self.tempResult = result

		elif self.OUTPUT_TYPE == FT.OUTPUT_TREND :
			output = None
			if len(self.tempResult) == self.TREND_CHUNK :
				output = map(self.linefit,zip(*self.tempResult))
				self.tempResult.pop()
			self.tempResult.append(result)

		return output

	def norm(self,a):
		return np.sqrt((np.square(a.real) + np.square(a.imag))/(self.CALC_SIZE*FFT_NORM_CORRECTION))

	def linefit(self,data):
		slope = np.polyfit(data,self.TREND_GROUPING,1)[0]
		return np.sign(slope) * np.log10(np.abs(slope)) * UI_RANGE/5 + UI_RANGE/2

FT = FeatureExtractor