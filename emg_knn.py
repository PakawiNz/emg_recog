import csv
import random
import math
import operator
from emg_arff import getPath_csv
from emg_fft import current_milli_time
 
class KNN(object):
	@staticmethod
	def euclideanDistance(instance1, instance2, length):
		distance = 0
		for x in range(length):
			distance += pow((instance1[x] - instance2[x]), 2)
		return math.sqrt(distance)
	 
	@staticmethod
	def getResponse(neighbors):
		classVotes = {}
		for x in range(len(neighbors)):
			response = neighbors[x][-1]
			if response in classVotes:
				classVotes[response] += 1
			else:
				classVotes[response] = 1
		sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
		return sortedVotes[0][0]

	def __init__(self,csvFilename):
		self.filename = getPath_csv(csvFilename)
		self.trainingSet = []
		with open(self.filename, 'rb') as csvfile:
			lines = csv.reader(csvfile)
			for line in list(lines) :
				self.trainingSet.append(map(float,line))

	def getNeighbors(self, testInstance, k):
		distances = []
		length = len(testInstance)
		for x in range(len(self.trainingSet)):
			dist = KNN.euclideanDistance(testInstance, self.trainingSet[x], length)
			distances.append((self.trainingSet[x], dist))
		distances.sort(key=operator.itemgetter(1))
		neighbors = []
		for x in range(k):
			neighbors.append(distances[x][0])
		return neighbors

	def activate(self, testInstance):
		neighbors = self.getNeighbors(testInstance, 1)
		result = KNN.getResponse(neighbors)
		return result

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


def statConfusionMatric(confusionMatrix,asList=False):
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

def crossValidation(filename,k_fold=10,header=False):

	from random import shuffle
	knn = KNN(filename)
	shuffle(knn.trainingSet)
	shuffled = knn.trainingSet
	length = len(shuffled)/k_fold

	# conf = createConfusionMatrix(shuffled)
	# printConfusionMatric(conf,'\t\t')
	# statConfusionMatric(conf)
	# exit()

	for fold in range(k_fold):
		knn.trainingSet = shuffled[fold*length:(fold+1)*length]
		testSet = shuffled[:fold*length] + shuffled[(fold+1)*length:]

		conf = createConfusionMatrix(knn.trainingSet)
		accu = 0
		time = 0
		for test in testSet :
			begin = current_milli_time()
			result = knn.activate(test[:-1])
			time += current_milli_time() - begin

			if result == test[-1] :
				accu += 1
			conf[test[-1]][result] += 1

		stat = statConfusionMatric(conf,True)
		accu /= float(len(testSet))
		time /= float(len(testSet))
		print ",".join(map(str,[filename,length,len(shuffled),fold,accu,time] + stat))

if __name__ == '__main__':
	crossValidation('150305')
