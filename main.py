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
	
	updateTime = QtCore.pyqtSignal(int)
	updateRaw = QtCore.pyqtSignal(int)
	updateFFT = QtCore.pyqtSignal(list)

	def __init__(self):
		super(WorkingThread, self).__init__()
		self.paused = False
		self.message = False
		self._config = {}

	def config(self,key,value):
		self._config[key] = int(value)

	def start(self,sec,train,debug):

		# ser = SerialManager()
		extr = FeatureExtractor(**self._config)
		begin = datetime.datetime.now()

		if trian :
			mem = open(begin.strftime("recog %y%m%d-%H%M.txt"),'w')

		if sec <= 0 :
			infinite = True
		else :
			infinite = False

		r = 0
		while infinite or r < sec :
			for i in range(256) :

				time.sleep(0.003)
				# data = ser.recieve().ch1
				calctime = current_milli_time()
				data = np.random.uniform(0,1024)
				self.updateRaw.emit(data)

				result = extr.gather(data)

				if result : 
					self.updateTime.emit(current_milli_time() - calctime)
					self.updateFFT.emit(result)

					if trian :
						mem.write("%s\n"%(", ".join(map(lambda x: "%.05f"%x ,result))))
					if debug :
						printMagnitude(result)

				if self.paused :
					infinite = False
					sec = 0
					break

			# print "NOOB"
			r += 1

		# interval = datetime.datetime.now() - begin
		# print interval
		
		if trian :
			mem.close()

		# ser.close()

	def trian(self,sec=1):
		self.start(sec=sec,train=True,debug=False)

	def play(self,sec=1):
		self.start(sec=sec,train=False,debug=False)


def printMagnitude(result) :
	result = map(lambda a: "%.03f"%(a) ,result)

	print ",".join(result)

if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)
	main = MainWindow(WorkingThread)
	main.show()
	sys.exit(app.exec_())
