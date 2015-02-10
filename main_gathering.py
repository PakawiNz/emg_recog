from emg_serial import SerialManager

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

	def datastore(self):
		ser = SerialManager()
		self.terminate = False
		self.activity = None
		lastActivity = 0

		count = 0
		mem = open(datetime.datetime.now().strftime("recog %y%m%d.txt"),'a+')
		while not self.terminate :
			data = ser.recieve().ch1
			self.updateRaw.emit(data)

			if self.activity :
				count += 1
				self.updateTime.emit("%d"%(count))
				if lastActivity != self.activity[1] :
					lastActivity = self.activity[1]
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

	def play(self,sec=1):
		return


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
		layout.addWidget(self.btnTerminate						,2,0)
		layout.addWidget(QtGui.QLabel('Data Count')				,3,0)
		layout.addWidget(self.calcTime							,4,0)
		layout.addWidget(QtGui.QLabel('Current Activity')		,5,0)
		layout.addWidget(self.actLabel							,6,0)
		layout.addWidget(self.plotArea							,0,1,10,1)

		self.resize(1000,200)
		self.setWindowTitle("EMG RECOGNITION SYSTEM")

	@QtCore.pyqtSlot()
	def btnTrainFN(self):
		t = threading.Thread(target=self.btnTrain_slave)
		t.start()

	@QtCore.pyqtSlot()
	def btnTerminateFN(self):
		self.slave.terminate = True

	@QtCore.pyqtSlot(int)
	def updateRaw(self,data_new):
		self.plotArea.update(data_new)

	def btnTrain_slave(self):
		self.btnPlay.setDisabled(True)
		self.btnTrain.setDisabled(True)
		self.btnTerminate.setDisabled(False)
		self.slave.train(0)
		self.btnPlay.setDisabled(False)
		self.btnTrain.setDisabled(False)
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
	MainWindow(work).show()
	app.exec_()
	work.terminate = True
	sys.exit()