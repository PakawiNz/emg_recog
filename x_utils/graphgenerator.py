from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

from graphplotter import readCSV

white = (255,255,255)
red = (255,0,0)
yellow = (255,255,0)
green = (0,255,0)
cyan = (0,255,255)
blue = (0,0,255)
purple = (0,255,255)

# colorset = [white,red,yellow,green,cyan,blue,purple]
colorset = [white,white,white,white,white,white,white]
# colorset = [white,red,red,green,green,blue,blue]

epoch 			= [500]
momentum 		= [0.05,0.1,0.2,0.4,0.8,1.6,3.2]
learning_rate 	= [0.05,0.1,0.2,0.4,0.8,1.6,3.2]
hidden0 		= [4,5,6,7,8,9,10,12,14,16,20,24,30,36]
hidden1 		= [None,5,7,9,12,15,18]

def getPlotdata(): # return list of (xset,yset,color)
	testResult = readCSV(cutoff=False)
	plotData = []

	# for hidd0,color in zip(hidden0,colorset) :
	# 	filtered = filter(lambda x : x.hidd0==hidd0,testResult)
	# 	hidd1 = [x.hidd1 for x in filtered]
	# 	accu = [x.accu for x in filtered]
	# 	plotData.append((hidd1,accu,color))

	# filtered = filter(lambda x : x.hidd0==4,testResult)
	# hidd1 = [x.hidd1 for x in filtered]
	# accu = [x.accu for x in filtered]
	# desc = [x.getParam() for x in filtered]
	# plotData.append((hidd1,accu,red,desc))

	# filtered = filter(lambda x : x.hidd0==8,testResult)
	# hidd1 = [x.hidd1 for x in filtered]
	# accu = [x.accu for x in filtered]
	# desc = [x.getParam() for x in filtered]
	# plotData.append((hidd1,accu,green,desc))

	# filtered = filter(lambda x : x.hidd0==12,testResult)
	# hidd1 = [x.hidd1 for x in filtered]
	# accu = [x.accu for x in filtered]
	# desc = [x.getParam() for x in filtered]
	# plotData.append((hidd1,accu,blue,desc))

	filtered = testResult
	# filtered = filter(lambda x : x.hidd1 == 0,testResult)
	# filtered = filter(lambda x : x.learn == 0.05,filtered)
	# filtered = filter(lambda x : x.momtm == 0.05,filtered)
	# hidd0 = [x.hidd0 for x in filtered]
	xset = [x.hidd1 for x in filtered]
	yset = [x.accu for x in filtered]
	desc = [x.getParam() for x in filtered]
	plotData.append((xset,yset,cyan,desc))
	return plotData


class MainWindow(QtGui.QMainWindow):
	"""docstring for MainWindow"""

	def clicked(self, plot, points):
		for p in self.lastClicked:
			p.resetPen()
		pointSet = set()
		for p in points:
			pointSet.add((p._data[0],p._data[1]))

		descList = []
		for p in pointSet:
			for desc in self.descDict.get(p) or []:
				descList.append(desc)

		descList.sort(key=lambda x: float(x[7]),reverse=True)

		print 'clicked %d'%(len(descList))
		for desc in descList :
			print desc

		for p in points:
			p.setPen('b', width=2)
		self.lastClicked = points

	def __init__(self):
		super(MainWindow, self).__init__()	

		self.lastClicked = []
		self.setWindowTitle('ScatterPlot')
		self.resize(700,700)
		self.show()

		view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
		self.setCentralWidget(view)

		w4 = view.addPlot()
		s4 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None))
		w4.addItem(s4)
		s4.sigClicked.connect(self.clicked)
		self.s4 = s4

		plotData = getPlotdata()
		self.descDict = {}
		for xset,yset,rgb,desc in plotData :
			s4.addPoints(x=xset, y=yset, brush=pg.mkBrush(rgb[0], rgb[1], rgb[2], 20))

			for x,y,d in zip(xset,yset,desc) :
				self.descDict[(x,y)] = (self.descDict.get((x,y)) or []) + [d]


		# s4.clear()


app = QtGui.QApplication([])
mw = MainWindow()

if __name__ == '__main__':
	import sys
	if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
		app.exec_()

		import pyqtgraph.exporters
		exporter = pg.exporters.ImageExporter(mw.s4)
		# exporter.parameters()['width'] = 1000
		exporter.export('test.png')