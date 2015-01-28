from emg_serial import SerialManager
from emg_fft import FeatureExtractor
from emg_train import MotionModel
from emg_recog import recognize

import sys
import time
import threading
import datetime
import numpy as np

from PyQt4 import QtGui, QtCore
from PyQt4 import Qwt5 as Qwt
import pyqtgraph as pg
from customplot import SlidingPlot

class WorkingThread(QtCore.QObject):
	
	updateRaw = QtCore.pyqtSignal(int)
	updateFFT = QtCore.pyqtSignal(list)

	def __init__(self):
		super(WorkingThread, self).__init__()
		self.paused = False
		self.message = False

	def start(self,sec,train,debug):

		ser = SerialManager()
		extr = FeatureExtractor()

		begin = datetime.datetime.now()

		if self.trian :
			mem = open(begin.strftime("recog %y%m%d-%H%M.txt"),'w')

		if sec <= 0 :
			infinite = True
		else :
			infinite = False

		r = 0
		while infinite or r < sec :
			if self.paused :
				break
			for i in range(256) :
				pack = ser.recieve()
				result = extr.gather(pack.ch1)

				self.updateRaw.emit(pack.ch1)

				if result : 
					self.updateFFT.emit(result)
					if self.trian :
						mem.write("%s\n"%(", ".join(map(lambda x: "%.05f"%x ,result))))
					if debug :
						printMagnitude(result)

			r += 1

		interval = datetime.datetime.now() - begin
		print interval
		
		if self.trian :
			mem.close()

		ser.close()

	def trian(self,sec=1):
		self.start(sec=sec,train=True,debug=False)

	def play(self,sec=1):
		self.start(sec=sec,train=False,debug=False)

class MainWindow(QtGui.QMainWindow):
	"""docstring for MainWindow"""
	def __init__(self):
		QtGui.QMainWindow.__init__(self)
		self.slave = WorkingThread()
		
		layout = QtGui.QGridLayout()
		
		w = QtGui.QWidget()
		w.resize(1200,500)
		w.setLayout(layout)
		self.setCentralWidget(w)
		
		widget = QtGui.QPushButton('Train')
		widget.clicked.connect(self.btnTrainFN)
		self.btnTrain = widget

		widget = QtGui.QPushButton('Play')
		widget.clicked.connect(self.btnPlayFN)
		self.btnPlay = widget

		widget = QtGui.QPushButton('Pause')
		widget.setDisabled(True)
		widget.clicked.connect(self.btnPauseFN)
		self.btnPause = widget

		widget = SlidingPlot(lockAspect=True, enableMouse=False, enableMenu=False)
		self.slave.updateRaw.connect(self.updateRaw)
		self.plotArea = widget

		widget = pg.PlotWidget(lockAspect=True, enableMouse=False, enableMenu=False)

		widget.setYRange(0,500)
		self.slave.updateFFT.connect(self.updateFFT)
		self.fftArea = widget

		layout.addWidget(self.btnTrain	, 0, 0)   # button goes in upper-left
		layout.addWidget(self.btnPlay	, 0, 1)   # text edit goes in middle-left
		layout.addWidget(self.btnPause	, 0, 2)  # list widget goes in bottom-left
		layout.addWidget(self.plotArea	, 1, 0, 1, 3)  # plot goes on right side, spanning 3 rows
		layout.addWidget(self.fftArea	, 2, 0, 1, 3)  # plot goes on right side, spanning 3 rows

		self.resize(1200,500)
		self.setWindowTitle("EMG RECOGNITION SYSTEM")

	@QtCore.pyqtSlot()
	def btnTrainFN(self):
		t = threading.Thread(target=self.btnTrain_slave)
		t.start()

	@QtCore.pyqtSlot()
	def btnPlayFN(self):
		t = threading.Thread(target=self.btnPlay_slave)
		t.start()

	@QtCore.pyqtSlot()
	def btnPauseFN(self):
		self.slave.paused = True

	@QtCore.pyqtSlot(int)
	def updateRaw(self,data_new):
		self.plotArea.update(data_new)

	@QtCore.pyqtSlot(list)
	def updateFFT(self,fft_result):
		specItem = self.fftArea.getPlotItem()
		specItem.plot(fft_result,clear=True, symbolBrush=(255,0,0))

	def btnTrain_slave(self):
		self.slave.paused = False
		self.btnTrain.setDisabled(True)
		self.btnPause.setDisabled(False)
		self.slave.trian(0)
		self.btnTrain.setDisabled(False)
		self.btnPause.setDisabled(True)

	def btnPlay_slave(self):
		self.slave.paused = False
		self.btnPlay.setDisabled(True)
		self.btnPause.setDisabled(False)
		self.slave.play(0)
		self.btnPlay.setDisabled(False)
		self.btnPause.setDisabled(True)

if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)
	main = MainWindow()
	main.show()
	sys.exit(app.exec_())

def printMagnitude(result) :
	result = map(lambda a: "%.03f"%(a) ,result)

	for data in result :
		print data + ",",

	print ""