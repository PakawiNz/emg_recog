from emg_serial import SerialManager
from emg_arff import fd_store,storepick_arff
from emg_utils import getPath_raw
from emg_weka import WekaTrainer
from config import FEATURE_CONFIG,TRAINER_CONFIG

import sys
import datetime
import threading

from PyQt4 import QtGui, QtCore
from customplot import SlidingPlot

class WorkingThread(QtCore.QObject):
	
	updateAct = QtCore.pyqtSignal(str)
	updateTime = QtCore.pyqtSignal(str)
	updateRaw = QtCore.pyqtSignal(int)
	updateFFT = QtCore.pyqtSignal(list)

	def __init__(self):
		super(WorkingThread, self).__init__()
		self.filename = datetime.datetime.now().strftime("%y%m%d")

	def datastore(self):
		ser = SerialManager()
		self.terminate = False
		self.activity = None
		lastActivity = 0

		count = 0
		mem = open(getPath_raw(self.filename),'a+')
		while not self.terminate :
			data = ser.recieve().ch1
			self.updateRaw.emit(data)

			activity = self.activity
			if activity :
				count += 1
				self.updateTime.emit("%d"%(count))
				if lastActivity != activity[1] :
					lastActivity = activity[1]
					mem.write("%.03f,%s "%(data,lastActivity))
				else :
					mem.write("%.03f "%(data))

		try:
			ser.close()
			mem.close()
		except e:
			pass
			
		self.updateTime.emit("0")

	def train(self,sec=1):
		self.datastore()

	def convert(self):
		print "START CONVERSION"
		fd_store(self.filename,fftconfig=FEATURE_CONFIG)
		storepick_arff(0, self.filename)
		
		print "START TRAINING"
		trainer = WekaTrainer(*TRAINER_CONFIG)
		trainer.train(self.filename)
		trainer.saveTrained(self.filename)


class MainWindow(QtGui.QMainWindow):
	"""docstring for MainWindow"""
	def __init__(self,slave):
		QtGui.QMainWindow.__init__(self)
		self.slave = slave
		
		layout = QtGui.QGridLayout()
		
		w = QtGui.QWidget()
		w.resize(1000,200)
		w.setLayout(layout)
		w.keyPressEvent = self.keyPressEvent
		self.setCentralWidget(w)
		
		widget = QtGui.QPushButton('Record')
		widget.clicked.connect(self.btnTrainFN)
		self.btnTrain = widget

		widget = QtGui.QPushButton('Convert')
		widget.clicked.connect(self.btnConvertFN)
		self.btnConvert = widget

		widget = QtGui.QPushButton('Terminate')
		widget.setDisabled(True)
		widget.clicked.connect(self.btnTerminateFN)
		self.btnTerminate = widget

		self.calcTime = QtGui.QLabel('0')
		self.slave.updateTime.connect(lambda x : self.calcTime.setText("  >> %s"%(x)))

		self.actLabel = QtGui.QLabel('None')
		self.slave.updateAct.connect(lambda x : self.actLabel.setText("  %s"%(x)))

		widget = SlidingPlot(lockAspect=True, enableMouse=False, enableMenu=False)
		self.slave.updateRaw.connect(self.updateRaw)
		self.plotArea = widget

		layout.addWidget(self.btnTrain							,1,0)
		layout.addWidget(self.btnConvert						,2,0)
		layout.addWidget(self.btnTerminate						,3,0)
		layout.addWidget(QtGui.QLabel('Data Count')				,4,0)
		layout.addWidget(self.calcTime							,5,0)
		layout.addWidget(QtGui.QLabel('Current Activity')		,6,0)
		layout.addWidget(self.actLabel							,7,0)
		layout.addWidget(self.plotArea							,0,1,10,1)

		self.resize(1000,200)
		self.setWindowTitle("EMG RECOGNITION SYSTEM")

	@QtCore.pyqtSlot()
	def btnTrainFN(self):
		t = threading.Thread(target=self.btnTrain_slave)
		t.start()

	@QtCore.pyqtSlot()
	def btnConvertFN(self):
		t = threading.Thread(target=self.slave.convert)
		t.start()

	@QtCore.pyqtSlot()
	def btnTerminateFN(self):
		self.slave.terminate = True

	@QtCore.pyqtSlot(int)
	def updateRaw(self,data_new):
		self.plotArea.update(data_new)

	def btnTrain_slave(self):
		self.btnTrain.setDisabled(True)
		self.btnConvert.setDisabled(True)
		self.btnTerminate.setDisabled(False)
		self.slave.train(0)
		self.btnTrain.setDisabled(False)
		self.btnConvert.setDisabled(False)
		self.btnTerminate.setDisabled(True)

	def keyPressEvent(self, event):
	    if type(event) == QtGui.QKeyEvent and event.key() == QtCore.Qt.Key_A : 
	    	self.slave.activity = ('CIR_L',1)
	    elif type(event) == QtGui.QKeyEvent and event.key() == QtCore.Qt.Key_S:
	    	self.slave.activity = ('FLEX',2)
	    elif type(event) == QtGui.QKeyEvent and event.key() == QtCore.Qt.Key_D:
	    	self.slave.activity = ('CIR_R',3)
	    elif type(event) == QtGui.QKeyEvent and event.key() == QtCore.Qt.Key_W:
	    	self.slave.activity = ('EXTD',4)
	    elif type(event) == QtGui.QKeyEvent and event.key() == QtCore.Qt.Key_Q:
	    	self.slave.activity = ('REST',5)
	    elif type(event) == QtGui.QKeyEvent and event.key() == QtCore.Qt.Key_E:
	    	self.slave.activity = None

if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)
	work = WorkingThread()
	# work.filename = "150320"
	work.filename = "champ_CORE_02"
	MainWindow(work).show()
	app.exec_()
	work.terminate = True
	sys.exit()