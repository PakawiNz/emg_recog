import importer
from emg_utils import current_milli_time,createConfusionMatrix,statConfusionMatrix,printConfusionMatric
from emg_weka import WekaTrainer
from emg_arff import arffToData

def changespace(x):
	ts,minTS,maxTS,minTR,maxTR = x

	return (ts-minTS)/(maxTS-minTS)*(maxTR-minTR) + minTR


def crosstest(trainfile,testfile,delimit='\t'):
	trainer = WekaTrainer()
	trainer.loadTrained(trainfile)
	network = trainer.buildNetwork()

	testSet = arffToData(testfile)

	mintest = map(min,zip(*[test[0] for test in testSet]))
	maxtest = map(max,zip(*[test[0] for test in testSet]))
	mintrain = trainer.minarray
	maxtrain = trainer.maxarray

	# chk = map(lambda x: map(changespace,zip(x,mintest,maxtest,mintrain,maxtrain)),[test[0] for test in testSet])
	# minchk = map(min,zip(*chk))

	conf = createConfusionMatrix(testSet)
	for test in testSet :
		adj_test = map(changespace,zip(test[0],mintest,maxtest,mintrain,maxtrain))
		action = network.activate(adj_test) + 1

		conf[test[1]][action] += 1

	stat = statConfusionMatrix(conf,True)
	print delimit.join(map(str,stat))
	printConfusionMatric(conf,delimit)

if __name__ == '__main__':
	from itertools import product
	files = ['150320', 'data10000','champ_CORE_00']
	cartesian = list(product(files,files))
	from random import shuffle
	shuffle(cartesian)
	for a,b in cartesian :
		print '\n ------------------------- \n%s\t%s\t'%(a,b),
		crosstest(a, b)
	# crosstest('150320', 'data10000')
	# crosstest('data10000', '150320')
	# crosstest('150320', '150320')
	# crosstest('champ_CORE_00', '150320')
	# crosstest('champ_CORE_00', 'champ_CORE_00')
	# crosstest('150320', 'champ_CORE_00')
	# crosstest('data10000', 'champ_CORE_00')