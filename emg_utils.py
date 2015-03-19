import time
current_milli_time = lambda: int(round(time.time() * 1000))

import os
filepath = os.path.dirname(os.path.abspath(__file__))
os.chdir(filepath)
def mkdirs(path):
	dirname = os.path.dirname(path)
	if not os.path.exists(dirname):
	    os.makedirs(dirname)

def getPath_raw(filename):
	path = 'x_data/0raw/%s.txt'%(filename)
	mkdirs(path)
	return path

def getPath_csv(filename):
	path = 'x_data/1store/%s.csv'%(filename)
	mkdirs(path)
	return path

def getPath_arff(filename):
	path = 'x_data/2arff/%s.arff'%(filename)
	mkdirs(path)
	return path

def getPath_stat(exp,filename,note=0):
	path = 'x_data/3stat/%s/%s-%s-%s.csv'%(exp,filename,exp,note)
	mkdirs(path)
	return path

def getPath_train(filename):
	path = 'x_data/4train/%s.emg'%(filename)
	mkdirs(path)
	return path

# dropboxPath = '/home/pakawinz/Dropbox/'
# pcname = 'Acer'
# outputrange = '0-1000'

# def getPath_dropbox(exp,filename,note=0):
# 	path = '@Senior Project/03 Final/Stat/Stat-%s-%s.csv'%(outputrange,pcname)
# 	path = os.path.join(dropboxPath,path)
# 	mkdirs(path)
# 	return path

def createConfusionMatrix(trainset):
	ansset = set()
	[ansset.add(inst[-1]) for inst in trainset]
	confusionMatrix = {}
	for ans in ansset:
		confusionMatrix[ans] = {}
		for act in ansset:
			confusionMatrix[ans][act] = 0

	return confusionMatrix

def printConfusionMatric(confusionMatrix,delimit='\t'):
	matrixStr = "\nConfusion Matrix ::\n---" + delimit
	matrixStr += delimit.join([str(key) for key in confusionMatrix.iterkeys()])
	for ans,actSet in confusionMatrix.iteritems() :
		matrixStr += "\n%s"%(ans)
		for act,value in actSet.iteritems():
			matrixStr += delimit + "%d"%(actSet[act])

	print matrixStr

def statConfusionMatrix(confusionMatrix,asList=False):
	from itertools import product
	safeDivide = lambda x,y : y and float(x)/float(y)

	result = dict.fromkeys(confusionMatrix.iterkeys())
	keyList = list(result.iterkeys())
	for key in keyList :
		result[key] = {}
		r = result[key]

		negList = list(keyList)
		negList.remove(key)

		alltest = sum([confusionMatrix[i][j] for i,j in list(product(keyList,keyList))])
		r['TP'] = sum([confusionMatrix[i][j] for i,j in list(product([key],[key]))])
		r['TN'] = sum([confusionMatrix[i][j] for i,j in list(product(negList,negList))])
		r['FP'] = sum([confusionMatrix[i][j] for i,j in list(product(negList,[key]))])
		r['FN'] = sum([confusionMatrix[i][j] for i,j in list(product([key],negList))])
		r['TPR'] 	=	safeDivide(r['TP'],(r['TP']+r['FN']))
		r['TNR'] 	=	safeDivide(r['TN'],(r['TN']+r['FP']))
		r['PPV'] 	=	safeDivide(r['TP'],(r['TP']+r['FP']))
		r['NPV'] 	=	safeDivide(r['TN'],(r['TN']+r['FN']))
		r['ACCU'] 	=	safeDivide(r['TP'] + r['TN'],alltest)

	resultList = []
	if asList :
		resultList += [result[key]['TPR'] for key in keyList]
		resultList += [result[key]['TNR'] for key in keyList]
		resultList += [result[key]['PPV'] for key in keyList]
		resultList += [result[key]['NPV'] for key in keyList]
		resultList += [result[key]['ACCU'] for key in keyList]
		return resultList
	else :
		return result
