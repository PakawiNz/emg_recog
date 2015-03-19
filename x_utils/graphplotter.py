from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import datetime

ACCU_TRS = 80
TIME_TRS = 20

class TestResult(object):

	def __init__(self,idx,epoch,learn,momtm,hidd0,hidd1,time,accu,pcname,smth1,smth2,smth3,smth4,cutoff=True):
		self.idx = int(idx)
		self.epoch = int(epoch)
		self.learn = float(learn)
		self.momtm = float(momtm)
		self.hidd0 = int(hidd0)

		try :
			self.hidd1 = int(hidd1)
		except :
			self.hidd1 = 0

		try :
			self.time = datetime.datetime.strptime(time,'%M:%S.%f')
		except :
			self.time = datetime.datetime.strptime(time,'%M:%S:%f')

		self.time = self.time.minute * 60 + self.time.second

		self.accu = float(accu)

		if cutoff :
			if self.accu < ACCU_TRS :
				self.accu = ACCU_TRS

			if self.time > TIME_TRS*60 :
				self.time = TIME_TRS*60

		self.pcname = pcname
		self.smth1 = smth1
		self.smth2 = smth2
		self.smth3 = smth3
		self.smth4 = smth4

	def getParam(self):
		return [self.idx,self.epoch,self.learn,self.momtm,self.hidd0,self.hidd1,self.time,self.accu]

def readCSV(cutoff=True):
	afile = open('3stat/resultsum.csv','r')
	lines = afile.readlines()

	return map(lambda x : TestResult(*x.split(','),cutoff=cutoff) , lines)

def preparePoints(result,colorMode=0):
	pos = np.empty((len(result)+60, 3))
	size = np.empty((len(result)+60))
	color = np.empty((len(result)+60, 4))

	result = filter(lambda x : x.hidd1 == 0, result)
	xset = map(lambda x : x.learn,result)
	# yset = map(lambda x : x.learn,result)
	yset = map(lambda x : x.time,result)
	zset = map(lambda x : x.accu,result)

	minx,maxx = min(xset),max(xset)
	miny,maxy = min(yset),max(yset)
	minz,maxz = min(zset),max(zset)

	basex,rangex = (maxx+minx)/2.0,(maxx-minx)/2.0
	basey,rangey = (maxy+miny)/2.0,(maxy-miny)/2.0
	basez,rangez = (maxz+minz)/2.0,(maxz-minz)/2.0

	for i,inst in enumerate(zip(xset,yset,zset)):
		x,y,z = inst
		nx,ny,nz = (x-basex)/rangex,(y-basey)/rangey,(z-basez)/rangez
		pos[i] = (nx,ny,nz) 
		size[i] = 1
		if colorMode == 1 :
			color[i] = (
				(-nz + 1)/2, 
				(nz + 1)/2, 
				0, 0.5)
		else :
			color[i] = (
				(nx + 1)/2, 
				(ny + 1)/2, 
				(nz + 1)/2, 0.5)

	# RED (Y=1)
	for i in range(1,21):
		pos[-i] = (i/20.0,0,0)
		color[-i] = (1,0,0,1)
		size[-i] = 3

	# GREEN (Y=1)
	for i in range(1,21):
		pos[-i-20] = (0,i/20.0,0)
		color[-i-20] = (0,1,0,1)
		size[-i-20] = 3

	# BLUE (Z=1)
	for i in range(1,21):
		pos[-i-40] = (0,0,i/20.0)
		color[-i-40] = (0,0,1,1)
		size[-i-40] = 3

	return pos,size,color

def createUI():

	app = QtGui.QApplication([])
	w = gl.GLViewWidget()
	w.opts['distance'] = 100
	w.resize(700,700)
	w.show()
	w.setWindowTitle('GLScatterPlotItem')

	g = gl.GLGridItem()
	w.addItem(g)

	result = readCSV()

	# timeThreshold = 10*60
	# intime = filter(lambda x: x.time < timeThreshold, result)
	# maxacc = max(intime,key=lambda x: x.accu)
	# print maxacc.accu
	# print maxacc.idx
	# exit()

	pos,size,color = preparePoints(result,1)

	sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
	sp1.scale(20, 20, 20)
	w.addItem(sp1)
	return app

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
	import sys
	app = createUI()
	if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
		QtGui.QApplication.instance().exec_()