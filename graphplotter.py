from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import datetime

ACCU_TRS = 80
TIME_TRS = 20

class TestResult(object):

	def __init__(self,idx,epoch,learn,momtm,hidd1,hidd2,time,accu,pcname,smth1,smth2,smth3,smth4,):
		self.epoch = int(epoch)
		self.learn = float(learn)
		self.momtm = float(momtm)
		self.hidd1 = int(hidd1)

		try :
			self.hidd2 = int(hidd2)
		except :
			self.hidd2 = 0

		self.time = datetime.datetime.strptime(time,'%M:%S.%f')
		self.time = self.time.minute * 60 + self.time.second

		self.accu = float(accu)

		if self.accu < ACCU_TRS :
			self.accu = ACCU_TRS

		if self.time > TIME_TRS*60 :
			self.time = TIME_TRS*60


		self.pcname = pcname
		self.smth1 = smth1
		self.smth2 = smth2
		self.smth3 = smth3
		self.smth4 = smth4

def readCSV():
	afile = open('3stat/resultsum.csv','r')
	lines = afile.readlines()

	return map(lambda x : TestResult(*x.split(',')) , lines)

def preparePoints(result,colorMode=0):
	pos = np.empty((len(result)+60, 3))
	size = np.empty((len(result)+60))
	color = np.empty((len(result)+60, 4))

	xset = map(lambda x : x.hidd1,result)
	yset = map(lambda x : x.hidd2,result)
	zset = map(lambda x : x.accu,result)

	minx,maxx = min(xset),max(xset)
	miny,maxy = min(yset),max(yset)
	minz,maxz = min(zset),max(zset)

	basex,rangex = (maxx+minx)/2.0,(maxx-minx)/2.0
	basey,rangey = (maxy+miny)/2.0,(maxy-miny)/2.0
	basez,rangez = (maxz+minz)/2.0,(maxz-minz)/2.0

	for i,inst in enumerate(zip(xset,yset,zset)):
		x,y,z = inst
		pos[i] = ((x-basex)/rangex, (y-basey)/rangey, (z-basez)/rangez)
		size[i] = 1
		if colorMode == 1 :
			color[i] = (
				((basez-z)/rangez + 1)/2, 
				((z-basez)/rangez + 1)/2, 
				0, 0.5)
		else :
			color[i] = (
				((x-basex)/rangex + 1)/2, 
				((y-basey)/rangey + 1)/2, 
				((z-basez)/rangez + 1)/2, 0.5)

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

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 100
w.resize(700,700)
w.show()
w.setWindowTitle('GLScatterPlotItem')

g = gl.GLGridItem()
w.addItem(g)

result = readCSV()
pos,size,color = preparePoints(result,1)

sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
sp1.scale(20, 20, 20)
w.addItem(sp1)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
	import sys
	if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
		QtGui.QApplication.instance().exec_()