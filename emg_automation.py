from emg_fft import FeatureExtractor
from emg_recog import CustomRecognition

from pybrain import structure as STRUCT
from pybrain.supervised import BackpropTrainer, RPropMinusTrainer
import itertools,re
import time
import traceback
import numpy as np

from PyQt4 import QtGui,QtCore
import sys,threading

CALC_SIZE = [128,256,512,1024]
SLIDING_SIZE = [4,8,12,16,20]
FREQ_DOMAIN = [8,16,64]

OUTPUT_TYPE1 = [0,1]
TREND_CHUNK1 = [0]

OUTPUT_TYPE2 = [2]
TREND_CHUNK2 = [3,5,7,9,11,13,15,17,19,21]

NN_LAYER = map(STRUCT.__getattribute__,filter(lambda x: x.endswith('Layer'),dir(STRUCT)))
NN_CONNECTION = map(STRUCT.__getattribute__,filter(lambda x: x.endswith('Connection'),dir(STRUCT)))
NN_NETWORK = map(STRUCT.__getattribute__,filter(lambda x: x.endswith('Network'),dir(STRUCT)))
NN_TRAINER =[BackpropTrainer,RPropMinusTrainer]
NN_HIDDEN = [8,12,16,64,128]

AUTOMATION1 = [OUTPUT_TYPE1,CALC_SIZE,SLIDING_SIZE,FREQ_DOMAIN,TREND_CHUNK1]
AUTOMATION2 = [OUTPUT_TYPE2,CALC_SIZE,SLIDING_SIZE,FREQ_DOMAIN,TREND_CHUNK2]

# AUTOMATION0 = [NN_LAYER,NN_CONNECTION,NN_LAYER,NN_CONNECTION,NN_LAYER,NN_HIDDEN,NN_NETWORK,NN_TRAINER]
AUTOMATION0 = [	[STRUCT.LinearLayer],NN_CONNECTION,
				[STRUCT.LinearLayer],NN_CONNECTION,
				[STRUCT.SigmoidLayer],[0],
				NN_NETWORK,NN_TRAINER]

current_milli_time = lambda: int(round(time.time() * 1000))

def tuplation(x):
	try :
		if ',' in x :
			return tuple(map(float,x.split(',')))
		else :
			return float(x)
	except :
		return None

def feature_extr(config,dataset,profile=True):
	extr = FeatureExtractor(*config)

	calctime = [0,0]
	features = []
	supervised = 0

	for data in dataset :
		if data == None : continue
		if type(data) == tuple :
			data,supervised = data[0],data[1]
			extr.clearBuffer()

		begin_calc = current_milli_time()
		result = extr.gather(data)
		if result and supervised:
			features.append((result,supervised))
			calctime[0] += current_milli_time() - begin_calc
			calctime[1] += 1

	if profile :
		time_fillin = float(extr.CALC_SIZE)/256
		time_calc = float(calctime[0])/calctime[1]
		return features,time_fillin,time_calc
	else :
		return features

class Automation(QtCore.QObject) :

	exitSignal = QtCore.pyqtSignal()

	def __init__(self):
		super(Automation, self).__init__()
		self.terminate = False

	def get_rawdata(self, filename) :
		lines = open(filename,'r').readlines()
		result = []
		for line in lines :
			line = re.split(r'\s+',line)
			result += map(tuplation,line)
		return result

	def compare_fft(self) :
		elementANN = (
			STRUCT.LinearLayer,STRUCT.FullConnection,
			STRUCT.SoftmaxLayer,STRUCT.FullConnection,
			STRUCT.SigmoidLayer,STRUCT.FullConnection,
			STRUCT.FeedForwardNetwork,BackpropTrainer)

		rawdata = self.get_rawdata('result/recog 150206.txt')

		cartesian = list(itertools.product(*AUTOMATION1))
		print "fft variation = %d"%len(cartesian)

		count = 0
		afile = open('result/compare_fft.txt','a+')
		for elementFFT in cartesian[count:] :
			text = "\t".join([str(count),'>'*10] + map(lambda x: "%d"%x,elementFFT))
			print text
			count += 1
			try:
				if self.terminate :
					break

				features,t_fill,t_calc = feature_extr(elementFFT,rawdata)

				recog = CustomRecognition(elementFFT[3],*elementANN)
				map(lambda x : recog.addSample(*x),features)

				err = recog.training(20)
				acc = recog.validate()

				try :
					text = "\t".join(
						['>'*10] + map(lambda x: "%d"%x,elementFFT) + 
						["%.3f"%t_fill,"%.3f"%t_calc,"%.3f"%acc] + map(lambda x: "%.3f"%x,err))
					afile.write(text + "\n")
					print ">> SUCCESS"
				except :
					print traceback.print_exc()
					break
			except :
				continue

		print "FINISHED"

		afile.close()
		exit()


	def compare_ann(self,feature=None):
		elementFFT = [0,128,4,8,0]
		rawdata = self.get_rawdata('result/recog 150206.txt')
		if not feature :
			features = feature_extr(elementFFT, rawdata, profile=False)

		from temp2 import doublelayer_top_ten as elements
		cartesian = elements*100
		# cartesian = list(itertools.product([STRUCT.FeedForwardNetwork],NN_LAYER,NN_LAYER,NN_LAYER,NN_LAYER))
		print "ann variation = %d"%len(cartesian)

		ds = CustomRecognition.buildTrianingSet(features)

		count = 0
		afile = open('result/compare_ann.txt','a+')
		for network,inlayer,hidlayer1,hidlayer2,outlayer in cartesian[count:] :
			text = "\t".join([str(count),'>'*10,network.__name__,inlayer.__name__,hidlayer1.__name__,hidlayer2.__name__,outlayer.__name__])
			print text
			count += 1
			try :
				if self.terminate :
					break

				elementANN = (
					inlayer, STRUCT.FullConnection, 
					outlayer, STRUCT.FullConnection, 
					[hidlayer1,hidlayer2], STRUCT.FullConnection, 
					network, BackpropTrainer)

				recog = CustomRecognition(len(features[0][0]),*elementANN,reusedDataSet=ds)
				err = recog.training(5)
				acc = recog.validate()
				res = recog.recognize(features[0][0])
				rec = CustomRecognition.convertToMotion(res)

				# recog = CustomRecognition(len(features[0][0]),*elementANN,reusedDataSet=ds)
				# err2 = recog._trainer.trainUntilConvergence()	# NEVER END FUNCTION (Waitng More than 7 Hours)
				# acc2 = recog.validate()
				# res2 = recog.recognize(features[0][0])
				# rec2 = CustomRecognition.convertToMotion(res)

				try :
					text = "\t".join(
						['FXEP',network.__name__,inlayer.__name__,hidlayer1.__name__,hidlayer2.__name__,outlayer.__name__] + 
						["%.3f"%acc,"%d"%rec,] + map(lambda x: "%.3f"%x,res) + map(lambda x: "%.3f"%x,err))
					afile.write(text + "\n")
					# text = "\t".join(
					# 	['CVGN',network.__name__,inlayer.__name__,hidlayer1.__name__,hidlayer2.__name__,outlayer.__name__] + 
					# 	["%.3f"%acc2,"%d"%rec2,] + map(lambda x: "%.3f"%x,res2) + map(lambda x: "%.3f"%x,err2))
					# afile.write(text + "\n")
					print ">> SUCCESS"
				except :
					print traceback.print_exc()
					break
			except :
 				continue

		print "FINISHED"

		afile.close()
		self.exitSignal.emit()
		exit()

class MainWindow(QtGui.QMainWindow):
	def __init__(self,work):
		QtGui.QMainWindow.__init__(self)
		self.setWindowTitle("AUTOMATION")
		work.exitSignal.connect(sys.exit)

if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)
	work = Automation()
	gui = MainWindow(work)
	# t = threading.Thread(target=work.compare_fft)
	t = threading.Thread(target=work.compare_ann)

	t.start()
	gui.show()
	app.exec_()
	work.terminate = True
	sys.exit()