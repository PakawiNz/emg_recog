import importer
from emg_utils import getPath_stat
from emg_weka import WekaTrainer,isLinux
from emg_utils import current_milli_time

from random import shuffle
import multiprocessing as mp
import itertools
import threading
import sys,signal,os

POS_INF = 1000000

class AutoWekaWorker(object):
	"""docstring for AutoWekaWorker"""
	def __init__(self,exp,filename,cartesian,start,end,threadAmount):
		super(AutoWekaWorker, self).__init__()
		self.exp = exp
		self.filename = filename
		self.cartesian = cartesian[start:]
		self.start = start
		self.count = start
		self.end = end
		self.lock = threading.Lock()
		self.nextwrite = start
		self.buffer = {}
		self.threadAmount = threadAmount
		self.processStore = []
		self.statpath = getPath_stat(self.exp,self.filename)
		# self.statpath = self.getPath_dropbox(self.exp,self.filename,note='')

		if start < 0 :
			raise Exception('invalid start value')
		if end < 0 :
			raise Exception('invalid end value')

		afile = open(self.statpath,'a+')
		afile.close()

	def flush(self):
		self.writeout(force=True)

	def writeout(self,count=-1,csvdata='',force=False):
		if self.nextwrite == count :
			afile = open(self.statpath,'a+')
			afile.write(csvdata+'\n')
			afile.close()
			print csvdata
			self.nextwrite += 1
		else :
			self.buffer[count] = csvdata

		if force and not self.buffer.get(self.nextwrite) and self.nextwrite <= self.count:
			self.buffer[self.nextwrite] = "%d,no result"%(self.nextwrite)

		if self.buffer.get(self.nextwrite):
			self.writeout(self.nextwrite,self.buffer[self.nextwrite],force)

	def work(self):
		
		while len(self.cartesian) > 0:
			self.lock.acquire()
			count = self.count
			if count > self.end :
				self.writeout()
				self.lock.release()
				break
			else :
				var = self.cartesian.pop(0)
				self.count += 1
				print count,var
				self.lock.release()

			trainer = WekaTrainer(**var)
			start = current_milli_time()
			stat = trainer.runWEKA(self.filename,True,self.processStore)
			duration = current_milli_time()-start
			order = [count,var['EPOCH'],var['LEARNING_RATE'],var['MOMENTUM'],var['HIDDEN1'],var['HIDDEN2'],]
			csvdata = ",".join(map(str,list(order)+[duration]+stat+[trainer.WEKA_OPTION]))

			self.lock.acquire()
			self.writeout(count,csvdata)
			self.lock.release()

		self.lock.acquire()
		self.threadAmount -= 1
		if self.threadAmount == 0 :
			print "\n\n >>>>>> AUTOMATION FINISHED <<<<<<"
		self.lock.release()

def runEpoch(epoch):
	config = [
		[0.05,0.8,16,0],
		[3.2,3.2,20,0],
		[0.2,3.2,24,0],
		[1.6,3.2,20,0],
		[3.2,0.2,20,0],
		[3.2,1.6,20,0],
		[3.2,1.6,14,0],
		[0.1,3.2,16,0],
		[3.2,0.2,14,0],
		[0.1,0.8,20,0],]

	header = ['EPOCH','MOMENTUM','LEARNING_RATE','HIDDEN1','HIDDEN2']
	cartesian = map(lambda x : [x[0]]+x[1],itertools.product(epoch,config))
	cartesian = map(lambda x: dict(zip(header,x)), cartesian)
	# shuffle(cartesian)
	
	print "weka variation = %d"%len(cartesian)

	return cartesian

def createCartesian(epoch,momentum,learning_rate,hidden0,hidden1):

	header = ['EPOCH','MOMENTUM','LEARNING_RATE','HIDDEN1','HIDDEN2']

	# cartesian = list(dict(zip(dicts, x)) for x in itertools.product(*list(dicts.itervalues())))
	cartesian = list(itertools.product(epoch,momentum,learning_rate,hidden0,hidden1))
	cartesian = map(lambda x: dict(zip(header,x)), cartesian)
	# shuffle(cartesian)
	
	print "weka variation = %d"%len(cartesian)

	return cartesian

def multiAutoWEKA(exp,filename,threadAmount,start=0,end=100000):
	# epoch 			= [500,1000,2000,4000]
	# epoch 			= [500]
	# momentum 		= [0.05,0.1,0.2,0.4,0.8,1.6,3.2]
	# learning_rate 	= [0.05,0.1,0.2,0.4,0.8,1.6,3.2]
	# hidden0 		= [4,5,6,7,8,9,10,12,14,16,20,24,30,36]
	# hidden1 		= [0,5,7,9,12,15,18]
	epoch 			= [50,100,200,300,500,700,900,1200,1500,1800,2400,3000]
	learning_rate 	= [0.05]
	momentum 		= [0.4]
	hidden0 		= [16]
	hidden1 		= [0]

	print "WARNING : In order to exit, you should (type 'e') instead of (CTRL+C), unless some result may lost"
	if isLinux :
		print "WARNING : (Linux) You should kill the java process by yourself!! after exit with (type 'e')"

	# cartesian = createCartesian(epoch, momentum, learning_rate, hidden0, hidden1)
	cartesian = runEpoch(epoch)
	if end == POS_INF :
		end = len(cartesian)-1

	print 'thread amount = %d'%(thread)
	print 'start position = %d'%(start)
	print 'end position = %d'%(end)

	worker = AutoWekaWorker(exp, filename, cartesian, start, end, threadAmount)

	try:
		raw_input("Type anything to start... ")
	except Exception, e:
		print 'You should use terminal to run this program.'
		return

	for i in range(threadAmount) :
		t = threading.Thread(target=worker.work,name="awk%d"%i)
		t.daemon = True
		t.start()

	while True:
		kb = raw_input()
		if kb == 'e':
			print "EXITING"
			map(lambda x: x.terminate(), worker.processStore)
			worker.flush()
			sys.exit()
		elif worker.threadAmount == 0:
			print "EXITING"
			sys.exit()

		print "Automation is on progress (type 'e' to force exit)"

if __name__ == '__main__':

	thread = 1
	start = 0
	end = POS_INF

	if len(sys.argv) >= 2 :
		thread = int(sys.argv[1])
	if len(sys.argv) >= 3 :
		start = int(sys.argv[2])
	if len(sys.argv) >= 4 :
		end = int(sys.argv[3])

	multiAutoWEKA('Exp3', 'data10000', thread, start, end)