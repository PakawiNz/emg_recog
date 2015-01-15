import numpy as np
import threading

DATA_SIZE = 1024
CALC_SIZE = 128
BUFFER_SIZE = 4
FFT_NORM_CORRECTION = 0.54
FREQ_DOMAIN = 16
FREQ_GROUPING = map(lambda a: (a*CALC_SIZE/FREQ_DOMAIN,(a+1)*CALC_SIZE/FREQ_DOMAIN),range(FREQ_DOMAIN))

class FeatureExtractor(object) :
	
	def __init__(self,recognize,train_only=False,debug=True) :
		self.inBuffer = map(lambda a : [0]*CALC_SIZE, range(BUFFER_SIZE))
		self.counter = 0
		self.atBuffer = 0
		self.storage = []
		
		self.recognize = recognize
		self.train_only = train_only
		self.debug = debug

	def gather(self,data) :
		self.inBuffer[self.atBuffer][self.counter] = (data-DATA_SIZE/2.0)/DATA_SIZE
		self.counter += 1

		if self.counter == CALC_SIZE :
			raw_data = self.inBuffer[self.atBuffer]
			self.atBuffer = (self.atBuffer + 1) % BUFFER_SIZE
			self.counter = 0

			t = threading.Thread(target=self.calc, args=(raw_data,))
			# t.daemon = True
			t.start()

	def calc(self,raw_data,) :
		out = np.fft.fft(raw_data)

		result = map(FeatureExtractor.norm ,out)

		result = map(lambda a: reduce(lambda x,y: x+y,result[a[0]:a[1]])*100 , FREQ_GROUPING)

		self.storage.append(result)

		if self.train_only :
			if self.debug : printMagnitude(result)
			return None
		else :
			motion = self.recognize(result)
			if self.debug : printMagnitude(result + [motion])
			return self.recognize(result)

	@staticmethod
	def norm(a):
		return np.sqrt((np.square(a.real) + np.square(a.imag))/(CALC_SIZE*FFT_NORM_CORRECTION))

def printMagnitude(result) :
	result = map(lambda a: "%.03f"%(a) ,result)

	for data in result :
		print data + ",",

	print ""