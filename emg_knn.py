import random
import math
import operator
from emg_arff import getPath_arff,arffToData
from emg_fft import current_milli_time
from emg_utils import createConfusionMatrix,statConfusionMatrix
 
class KNN(object):
	@staticmethod
	def euclideanDistance(instance1, instance2, length):
		distance = sum(map(lambda x : pow((instance1[x] - instance2[x]), 2), range(length)))
		return math.sqrt(distance)
	 
	def __init__(self,arffFilename):
		self.filename = arffFilename
		self.trainingSet = arffToData(self.filename,True)
		self.responseSet = dict.fromkeys(zip(*self.trainingSet)[-1],0)
		self.inputLength = len(self.trainingSet[0])-1

	def getNeighbors(self, testInstance, k):
		if len(testInstance) != self.inputLength :
			raise Exception("test instance size is not matched with trained instance.")

		dict_fn = lambda trainInstance : (trainInstance, 
			KNN.euclideanDistance(testInstance, trainInstance, len(testInstance)))

		distances = map(dict_fn, self.trainingSet)
		distances.sort(key=operator.itemgetter(1))

		neighbors = []
		for x in range(k):
			neighbors.append(distances[x][0])
		return neighbors

	def getResponse(self,neighbors):
		classVotes = self.responseSet.copy()
		for x in range(len(neighbors)):
			response = neighbors[x][-1]
			classVotes[response] += 1
		result = max(classVotes.iteritems(), key=operator.itemgetter(1))
		return result[0]

	def activate(self, testInstance):
		neighbors = self.getNeighbors(testInstance, 1)
		result = self.getResponse(neighbors)
		return result

def crossValidation(filename,k_fold=10):
	from random import shuffle
	knn = KNN(filename)
	shuffle(knn.trainingSet)
	shuffled = knn.trainingSet
	length = len(shuffled)/k_fold

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

		stat = statConfusionMatrix(conf,True)
		accu /= float(len(testSet))
		time /= float(len(testSet))
		print ",".join(map(str,[filename,length,len(shuffled),fold,accu,time] + stat))

if __name__ == '__main__':
	crossValidation('data20000',20)
	# knn = KNN('data20000')
	# test = knn.trainingSet[0]
	# knn.activate(test[:-1])
