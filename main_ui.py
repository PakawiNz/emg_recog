from emg_fft import FeatureExtractor
import threading

from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
from customplot import SlidingPlot

class MainWindow(QtGui.QMainWindow):
	"""docstring for MainWindow"""
	def __init__(self,slave):
		QtGui.QMainWindow.__init__(self)
		self.slave = slave
		
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

		widget = QtGui.QPushButton('Terminate')
		widget.setDisabled(True)
		widget.clicked.connect(self.btnTerminateFN)
		self.btnTerminate = widget

		widget = QtGui.QCheckBox('Pause')
		widget.stateChanged.connect(self.btnPauseFN)
		self.btnPause = widget

		widget = SlidingPlot(lockAspect=True, enableMouse=False, enableMenu=False)
		self.slave.updateRaw.connect(self.updateRaw)
		self.plotArea = widget

		widget = pg.PlotWidget(lockAspect=True, enableMouse=False, enableMenu=False)
		widget.setYRange(0,500)
		widget.keyPressEvent = self.keyPressEvent
		self.slave.updateFFT.connect(self.updateFFT)
		self.fftArea = widget

		layout.setColumnStretch(0, 1)
		layout.setColumnStretch(1, 3)
		layout.setColumnStretch(2, 3)
		layout.setColumnStretch(3, 3)
		layout.setColumnStretch(4, 3)

		layout.addWidget(self.configZone()	, 0, 0, 3, 1)	# left zone
		layout.addWidget(self.btnTrain		, 0, 1)
		layout.addWidget(self.btnPlay		, 0, 2)
		layout.addWidget(self.btnTerminate	, 0, 3)
		layout.addWidget(self.btnPause		, 0, 4)
		layout.addWidget(self.plotArea		, 1, 1, 1, 4)  # plot goes on right side, spanning 3 rows
		layout.addWidget(self.fftArea		, 2, 1, 1, 4)  # plot goes on right side, spanning 3 rows

		self.resize(1200,500)
		self.setWindowTitle("EMG RECOGNITION SYSTEM")

	def configZone(self):
		layout = QtGui.QGridLayout()
		w = QtGui.QWidget()
		w.setLayout(layout)
		w.resize(100,300)

		widget = QtGui.QSpinBox()
		widget.setMaximum(25600)
		widget.setLocale(QtCore.QLocale(QtCore.QLocale.English,QtCore.QLocale.UnitedStates))
		widget.setValue(FeatureExtractor.DEFAULT_CALC_SIZE)
		widget.valueChanged.connect(lambda x : self.slave.config('CALC_SIZE',x))
		self.calcSize = widget
		
		widget = QtGui.QSpinBox()
		widget.setLocale(QtCore.QLocale(QtCore.QLocale.English,QtCore.QLocale.UnitedStates))
		widget.setValue(FeatureExtractor.DEFAULT_SLIDING_SIZE)
		widget.valueChanged.connect(lambda x : self.slave.config('SLIDING_SIZE',x))
		self.slideSize = widget

		widget = QtGui.QSpinBox()
		widget.setLocale(QtCore.QLocale(QtCore.QLocale.English,QtCore.QLocale.UnitedStates))
		widget.setValue(FeatureExtractor.DEFAULT_FREQ_DOMAIN)
		widget.valueChanged.connect(lambda x : self.slave.config('FREQ_DOMAIN',x))
		self.freqSize = widget

		widget = QtGui.QSpinBox()
		widget.setLocale(QtCore.QLocale(QtCore.QLocale.English,QtCore.QLocale.UnitedStates))
		widget.setValue(FeatureExtractor.DEFAULT_TREND_CHUNK)
		widget.valueChanged.connect(lambda x : self.slave.config('TREND_CHUNK',x))
		self.trendSize = widget

		widget = QtGui.QComboBox()
		widget.setLocale(QtCore.QLocale(QtCore.QLocale.English,QtCore.QLocale.UnitedStates))
		widget.addItem("RAW")
		widget.addItem("DIFF")
		widget.addItem("TREND")
		widget.currentIndexChanged.connect(lambda x : self.slave.config('OUTPUT_TYPE',x))
		self.outType = widget

		self.calcTime = QtGui.QLabel('0')
		self.slave.updateTime.connect(lambda x : self.calcTime.setText("  >> %s"%(x)))

		self.actLabel = QtGui.QLabel('None')
		self.slave.updateAct.connect(lambda x : self.actLabel.setText("  %s"%(x)))

		layout.setRowStretch(0, 100)
		layout.addWidget(QtGui.QLabel('') 							, 0,0)
		layout.addWidget(QtGui.QLabel('Calculation Chunk Size') 	, 1,0)
		layout.addWidget(self.calcSize								, 2,0)
		layout.addWidget(QtGui.QLabel('Sliding Step') 				, 3,0)
		layout.addWidget(self.slideSize								, 4,0)
		layout.addWidget(QtGui.QLabel('Frequency Domain') 			, 5,0)
		layout.addWidget(self.freqSize								, 6,0)
		layout.addWidget(QtGui.QLabel('Trend Chunk Size') 			, 7,0)
		layout.addWidget(self.trendSize								, 8,0)
		layout.addWidget(QtGui.QLabel('Output Type') 				, 9,0)
		layout.addWidget(self.outType								, 10,0)
		layout.addWidget(QtGui.QLabel('Calculation Time')			, 11,0)
		layout.addWidget(self.calcTime								, 12,0)
		layout.addWidget(QtGui.QLabel('Current Activity')			, 13,0)
		layout.addWidget(self.actLabel								, 14,0)

		return w

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
		self.slave.paused = self.btnPause.isChecked()

	@QtCore.pyqtSlot()
	def btnTerminateFN(self):
		self.slave.terminate = True

	@QtCore.pyqtSlot(int)
	def updateRaw(self,data_new):
		self.plotArea.update(data_new)

	@QtCore.pyqtSlot(list)
	def updateFFT(self,fft_result):
		specItem = self.fftArea.getPlotItem()
		specItem.plot(fft_result,clear=True, symbolBrush=(255,0,0))

	def btnTrain_slave(self):
		self.btnPlay.setDisabled(True)
		self.btnTrain.setDisabled(True)
		self.btnTerminate.setDisabled(False)
		self.slave.train(0)
		self.btnPlay.setDisabled(False)
		self.btnTrain.setDisabled(False)
		self.btnTerminate.setDisabled(True)

	def btnPlay_slave(self):
		self.btnPlay.setDisabled(True)
		self.btnTrain.setDisabled(True)
		self.btnTerminate.setDisabled(False)
		self.slave.play(0)
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