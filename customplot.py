import pyqtgraph as pg
import numpy as np

class SlidingPlot(pg.PlotWidget):
    """docstring for SlidingPlot"""
    def __init__(self,**kwargs):
        super(SlidingPlot, self).__init__(**kwargs)

        self.chunkSize = 256
        self.maxChunks = 10
        # Remove chunks after we have 10
        self.startTime = pg.ptime.time()
        self.curves = []
        self.data = np.empty((self.chunkSize+1,2))
        self.ptr = 0

        self.setLabel('bottom', 'Time', 's')
        self.setYRange(0, 1024)
        self.setXRange(-10, 0)

    def update(self,data_new):
        now = pg.ptime.time()
        for c in self.curves:
            c.setPos(-(now-self.startTime), 0)
        
        i = self.ptr % self.chunkSize
        if i == 0:
            curve = self.plot()
            self.curves.append(curve)
            last = self.data[-1]
            self.data = np.empty((self.chunkSize+1,2))        
            self.data[0] = last
            while len(self.curves) > self.maxChunks:
                c = self.curves.pop(0)
                self.removeItem(c)
        else:
            curve = self.curves[-1]

        self.data[i+1,0] = now - self.startTime
        self.data[i+1,1] = data_new
        curve.setData(x=self.data[:i+2, 0], y=self.data[:i+2, 1])
        self.ptr += 1
