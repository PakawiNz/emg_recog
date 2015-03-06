from emg_arff import getPath_stat
from emg_weka import WekaTrainer

from random import shuffle
import multiprocessing as mp
import itertools
import datetime
import threading
import sys

class AutoWekaWorker(object):
	"""docstring for AutoWekaWorker"""
	def __init__(self,exp,filename,cartesian,start):
		super(AutoWekaWorker, self).__init__()
		self.exp = exp
		self.filename = filename
		self.cartesian = cartesian[start:]
		self.count = start
		self.lock = threading.Lock()

	def work(self):
		# note=threading.current_thread().name
		note = ''
		
		while len(self.cartesian) > 0:
			self.lock.acquire()
			var = self.cartesian.pop(0)
			count = self.count
			self.count += 1
			print count,var
			self.lock.release()
			
			trainer = WekaTrainer(**var)
			start = datetime.datetime.now()
			stat = trainer.runWEKA(self.filename,True)
			duration = datetime.datetime.now()-start
			order = [count,var['EPOCH'],var['LEARNING_RATE'],var['MOMENTUM'],var['HIDDEN1'],var['HIDDEN2'],]
			csvdata = ",".join(map(str,list(order)+[duration]+stat+[trainer.WEKA_OPTION]))

			self.lock.acquire()
			afile = open(getPath_stat(self.exp,self.filename,note=note),'a+')
			afile.write(csvdata+'\n')
			afile.close()
			print csvdata
			self.lock.release()

def createCartesian(epoch,momentum,learning_rate,hidden0,hidden1):

	header = ['EPOCH','MOMENTUM','LEARNING_RATE','HIDDEN1','HIDDEN2']

	# cartesian = list(dict(zip(dicts, x)) for x in itertools.product(*list(dicts.itervalues())))
	cartesian = list(itertools.product(*[epoch,momentum,learning_rate,hidden0,hidden1]))
	cartesian = map(lambda x: dict(zip(header,x)), cartesian)
	# shuffle(cartesian)
	
	print "weka variation = %d"%len(cartesian)

	return cartesian

def multiAutoWEKA(exp,filename,threadAmount,start=0):
	# epoch 			= [500,1000,2000,4000]
	epoch 			= [100]
	momentum 		= [0.05,0.1,0.2,0.4,0.8,1.6,3.2]
	learning_rate 	= [0.05,0.1,0.2,0.4,0.8,1.6,3.2]
	hidden0 		= [4,5,6,7,8,9,10,12,14,16,20,24,30,36]
	hidden1 		= [None,5,7,9,12,15,18]

	cartesian = createCartesian(epoch, momentum, learning_rate, hidden0, hidden1)

	worker = AutoWekaWorker(exp, filename, cartesian, start)
	for i in range(threadAmount) :
		t = threading.Thread(target=worker.work,name="awk%d"%i)
		t.start()

if __name__ == '__main__':
	thread = 1
	start = 0

	if len(sys.argv) >= 2 :
		thread = int(sys.argv[1])
	if len(sys.argv) >= 3 :
		start = int(sys.argv[2])

	print 'thread amount = %d'%(thread)
	print 'start position = %d'%(start)

	multiAutoWEKA('Exp1', 'data10000', thread, start)