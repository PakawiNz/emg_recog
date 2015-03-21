from emg_fft import FeatureExtractor
from emg_serial import SerialManager
from emg_weka import WekaTrainer
from emg_utils import current_milli_time
from main_ui import MainWindow,action_name

import sys
import time

from PyQt4 import QtGui, QtCore

class WorkingThread(QtCore.QObject):
	
	updateAct = QtCore.pyqtSignal(int)
	updateTime = QtCore.pyqtSignal(int)
	updateRaw = QtCore.pyqtSignal(int)
	updateFFT = QtCore.pyqtSignal(list)

	def __init__(self):
		super(WorkingThread, self).__init__()
		self.paused = False
		self.message = False
		self._config = {
			'OUTPUT_TYPE' : 0,
			'CALC_SIZE' : 128,
			'SLIDING_SIZE' : 4,
			'FREQ_DOMAIN' : 8,
			'TREND_CHUNK' : 0,
		}
		self.trainfile = None
		self.confusion = [[0]*6 for i in range(6)]


	def selectFile(self,filename):
		self.trainfile = filename

	def config(self,key,value):
		self._config[key] = int(value)

	def start(self):
		self.activity = None
		self.terminate = False

		ser = SerialManager()
		extr = FeatureExtractor(**self._config)

		if self.trainfile == None:
			raise Exception('no trained data is selected')

		trainer = WekaTrainer()
		trainer.loadTrained(self.trainfile)
		self.network = trainer.buildNetwork()
		print self.trainfile

		infinite = True
		while infinite :
			## -------- TERMINATE ---------------------------------------------------------------------
			if self.terminate :
				infinite = False
				break
			## -------- PAUSE ---------------------------------------------------------------------
			if self.paused :
				time.sleep(0.003)
				continue
			## -------- RECIEVE ---------------------------------------------------------------------
			data = ser.recieve().ch1
			self.updateRaw.emit(data)
			## -------- CALCULATE ---------------------------------------------------------------------
			calctime = current_milli_time()
			result = extr.gather(data)
			if result : 
				self.updateTime.emit(current_milli_time() - calctime)
				self.updateFFT.emit(result)
				action = self.network.activate(result) + 1
				self.updateAct.emit(action)
				if self.activity :
					self.confusion[self.activity[1]][action] += 1

			## -----------------------------------------------------------------------------

		ser.close()
		
		matrixStr = "\nConfusion Matrix ::\n----\t"
		matrixStr += "\t".join(map(str,action_name[1:])) + "\n"

		for arow,action in zip(self.confusion[1:],action_name[1:]) :
			matrixStr += "%s\t"%(action)
			matrixStr += "\t\t".join(map(str,arow[1:])) + "\n"

		print matrixStr

	def train(self,sec=1):
		return

	def play(self,sec=1):
		self.start()

if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)
	work = WorkingThread()
	MainWindow(work).show()
	app.exec_()
	work.terminate = True
	sys.exit()