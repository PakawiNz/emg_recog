from emg_fft import FeatureExtractor
from emg_recog import Recognition

import itertools
import time
import numpy as np

current_milli_time = lambda: int(round(time.time() * 1000))

def features(dataset,OUTPUT_TYPE,CALC_SIZE,SLIDING_SIZE,FREQ_DOMAIN,TREND_CHUNK):
	extr = FeatureExtractor(
		OUTPUT_TYPE = OUTPUT_TYPE,
		CALC_SIZE = CALC_SIZE,
		SLIDING_SIZE = SLIDING_SIZE,
		FREQ_DOMAIN = FREQ_DOMAIN,
		TREND_CHUNK = TREND_CHUNK)

	calctime = (0,0)
	features = []
	supervised = 0

	for data in dataset :
		if type(data) == tuple :
			data,supervised = data[0],data[1]

		begin_calc = current_milli_time()
		result = extr.gather(data)
		if result and supervised:
			features.append((result,supervised))
			calctime[0] += current_milli_time() - begin_calc
			calctime[1] += 1

	time_fillin = float(CALC_SIZE)/256
	time_calc = float(calctime[0])/calctime[1]
	return features,time_fillin,time_calc

def training(features,EPOCHS) :
	for i in EPOCHS :
		recog = Recognition(len(features[0]))
		map(recog.addSample , *features)
		recog.training(i)


CALC_SIZE = [32,64,128,256,512,1024]
SLIDING_SIZE = [2,4,6,8,10,12,14,16,18,20]
FREQ_DOMAIN = [8,16,32,64,128]

OUTPUT_TYPE1 = [0,1]
TREND_CHUNK1 = [0]

OUTPUT_TYPE2 = [2]
TREND_CHUNK2 = [3,5,7,9,11,13,15,17,19,21]

AUTOMATION1 = [OUTPUT_TYPE1,CALC_SIZE,SLIDING_SIZE,FREQ_DOMAIN,TREND_CHUNK1]
AUTOMATION2 = [OUTPUT_TYPE2,CALC_SIZE,SLIDING_SIZE,FREQ_DOMAIN,TREND_CHUNK2]

from pybrain import structure as STRUCT

NN_LAYER = map(STRUCT.__getattribute__, 
	filter(lambda x: x.endswith('Layer'),dir(STRUCT)))
NN_CONNECTION = map(STRUCT.__getattribute__, 
	filter(lambda x: x.endswith('Connection'),dir(STRUCT)))
NN_NETWORK = map(STRUCT.__getattribute__, 
	filter(lambda x: x.endswith('Network'),dir(STRUCT)))
NN_HIDDEN = [(1,3),(1,6),(1,9)]

def automation() :
	count = 0
	cartesian_fft = itertools.product(*AUTOMATION1)
	for element in cartesian_fft:
		features(dataset,*element)
		print element
		count += 1
	print count

print "\n".join(map(str,NN_NETWORK))