from emg_arff import getPath_stat
from emg_weka import WekaTrainer

from random import shuffle
import multiprocessing as mp
import itertools
import datetime
import threading

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
			print var
			self.lock.release()

			trainer = WekaTrainer(**var)
			start = datetime.datetime.now()
			stat = trainer.runWEKA(self.filename,True)
			duration = datetime.datetime.now()-start

			self.lock.acquire()
			order = [self.count,var['EPOCH'],var['LEARNING_RATE'],var['MOMENTUM'],var['HIDDEN1'],var['HIDDEN2'],]
			csvdata = ",".join(map(str,list(order)+[duration]+stat+[trainer.WEKA_OPTION]))
			afile = open(getPath_stat(self.exp,self.filename,note=note),'a+')
			afile.write(csvdata+'\n')
			afile.close()
			self.count += 1
			print csvdata
			self.lock.release()

def createCartesian(epoch,momentum,learning_rate,hidden0,hidden1):

	dicts = {
		'LEARNING_RATE': learning_rate,
		'MOMENTUM': momentum,
		'EPOCH': epoch,
		'HIDDEN1': hidden0,
		'HIDDEN2': hidden1,
		}

	cartesian = list(dict(zip(dicts, x)) for x in itertools.product(*list(dicts.itervalues())))
	# cartesian = list(itertools.product(*[[0],epoch,momentum,learning_rate,hidden0,hidden1]))
	# shuffle(cartesian)
	
	print "weka variation = %d"%len(cartesian)

	return cartesian

def multiAutoWEKA(exp,filename,threadAmount,start=0):
	epoch 			= [500,1000,2000,4000]
	momentum 		= [0.05,0.1,0.2,0.3,0.4,0.5,0.75,1.0,1.5,2.0]
	learning_rate 	= [0.05,0.1,0.2,0.3,0.4,0.5,0.75,1.0,1.5,2.0]
	hidden0 		= [4,5,6,7,8,9,10,12,14,16,18,20,23,26,29,32,36,40]
	hidden1 		= [None,5,6,7,8,9,10,12,14,16,18,20]

	cartesian = createCartesian(epoch, momentum, learning_rate, hidden0, hidden1)

	worker = AutoWekaWorker(exp, filename, cartesian, start)
	for i in range(threadAmount) :
		t = threading.Thread(target=worker.work,name="awk%d"%i)
		t.start()

if __name__ == '__main__':

	multiAutoWEKA('Exp1', 'data10000', 3, 3)