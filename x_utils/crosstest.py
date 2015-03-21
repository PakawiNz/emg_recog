import importer
from emg_utils import current_milli_time,createConfusionMatrix,statConfusionMatrix,printConfusionMatric
from emg_weka import WekaTrainer
from emg_arff import arffToData

def crosstest(trainfile,testfile,delimit='\t'):
	trainer = WekaTrainer()
	trainer.loadTrained(trainfile)
	network = trainer.buildNetwork()

	testSet = arffToData(testfile)
	conf = createConfusionMatrix(testSet)
	for test in testSet :
		action = network.activate(test[0]) + 1

		conf[test[1]][action] += 1

	stat = statConfusionMatrix(conf,True)
	print delimit.join(map(str,stat))
	printConfusionMatric(conf,delimit)

if __name__ == '__main__':
	from itertools import product
	files = ['150320', 'data10000','champ_CORE_00']
	cartesian = product(files,files)
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