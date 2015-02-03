from emg_serial import SerialManager
from emg_fft import FeatureExtractor
from emg_recog import Recognition
from main_ui import MainWindow

import sys
import time
import datetime
import numpy as np

from PyQt4 import QtGui, QtCore

current_milli_time = lambda: int(round(time.time() * 1000))

class WorkingThread(QtCore.QObject):
	
	updateAct = QtCore.pyqtSignal(str)
	updateTime = QtCore.pyqtSignal(str)
	updateRaw = QtCore.pyqtSignal(int)
	updateFFT = QtCore.pyqtSignal(list)

	def __init__(self):
		super(WorkingThread, self).__init__()
		self.paused = False
		self.message = False
		self._recog = None
		self._config = {}

	def config(self,key,value):
		self._config[key] = int(value)

	def start(self,sec,train,debug):
		self.activity = None
		self.terminate = False

		ser = SerialManager()
		extr = FeatureExtractor(**self._config)
		begin = datetime.datetime.now()

		if train :
			mem = open(begin.strftime("recog %y%m%d-%H%M.txt"),'w')
			if not self._recog :
				self._recog = Recognition(extr.FREQ_DOMAIN)

		r = 0
		infinite = (sec <= 0)
		while infinite or r < sec :
			for i in range(256) :
				## -------- TERMINATE ---------------------------------------------------------------------
				if self.terminate :
					infinite = False
					sec = 0
					break
				## -------- PAUSE ---------------------------------------------------------------------
				if self.paused :
					time.sleep(0.003)
					continue
				## -------- RECIEVE ---------------------------------------------------------------------
				# time.sleep(0.003)
				# data = np.random.uniform(0,1024)
				data = ser.recieve().ch1
				self.updateRaw.emit(data)
				## -------- CALCULATE ---------------------------------------------------------------------
				calctime = current_milli_time()
				result = extr.gather(data)
				if result : 
					self.updateTime.emit("%d"%(current_milli_time() - calctime))
					self.updateFFT.emit(result)
					if debug : printMagnitude(result)

					if train :
						# mem.write("%s\n"%(", ".join(map(lambda x: "%.05f"%x ,result))))
						if self.activity :
							self.updateAct.emit(self.activity[0])
							if self.activity[1]:
								self._recog.addSample(result, self.activity[1])
					elif self._recog :
						self._recog.recognize(result)
				## -----------------------------------------------------------------------------
			r += 1

		interval = datetime.datetime.now() - begin
		print interval
		
		ser.close()
		
		if train :
			mem.close()
			self._recog.training(500,lambda x: self.updateAct.emit("TRAINING %.02f"%(x)))
			self.updateAct.emit("FINISHED")


	def train(self,sec=1):
		self.start(sec=sec,train=True,debug=False)

	def play(self,sec=1):
		self.start(sec=sec,train=False,debug=False)


def printMagnitude(result) :
	result = map(lambda a: "%.03f"%(a) ,result)

	print ",".join(result)

if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)
	work = WorkingThread()
	MainWindow(work).show()
	app.exec_()
	work.terminate = True
	sys.exit()
