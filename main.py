from emg_serial import SerialManager
from emg_fft import FeatureExtractor
from emg_train import MotionModel
from emg_recog import recognize

import datetime
import msvcrt
import os
# asks whether a key has been acquired
def kbfunc():
    #this is boolean for whether the keyboard has bene hit
    x = msvcrt.kbhit()
    if x:
        #getch acquires the character encoded in binary ASCII
        ret = msvcrt.getch()
        ret = ret.decode()
    else:
        ret = None
    return ret

def start(sec,train_only,debug):
	ser = SerialManager()
	extr = FeatureExtractor(recognize,train_only=train_only,debug=debug)

	# start = datetime.datetime.now()
	
	if sec <= 0 :
		infinite = True
	else :
		infinite = False

	r = 0
	while infinite or r < sec :
		if kbfunc() == 's' :
			exit()
		for i in range(256) :
			pack = ser.recieve()
			extr.gather(pack.ch1)
		r += 1

	# interval = datetime.datetime.now() - start
	# print interval
	
	ser.close()
	return extr.storage
	
def trian(sec=10):
	storage = start(sec=sec,train_only=True,debug=True)
	storage = zip(*storage)

	model = MotionModel(MotionModel.FILE_FLEX)

	model.average = map(np.average ,storage)
	model.stdev = map(np.std ,storage)
	model.max = map(np.max ,storage)
	model.min = map(np.min ,storage)

	print "SUMMARIZED"
	printMagnitude(model.average)
	printMagnitude(model.stdev)

	model.save()

def play(sec=1):
	start(sec=sec,train_only=False,debug=True)

try :
	raw_input("--> type anything to start :: ")
	play(0)
except :
	play(10)