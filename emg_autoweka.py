import itertools
import re
import os
import datetime

#Exp0 correct command /
# N_FOLD 	= [10]
# EPOCH 	= [50]
# MOMENTUM 	= [0.2]
# LEARNING_RATE = [0.3]
# HIDDEN0 	= [8]
# HIDDEN1 	= [0]

#Exp1 vary HIDDEN0, HIDDEN1 >>> test 1 or 2 Layer
# N_FOLD 	= [10]
# EPOCH 	= [500] 
# MOMENTUM 	= [0.1]
# LEARNING_RATE = [0.1]
# HIDDEN0 	= [8,7,6,5]
# HIDDEN1 	= [7,6,5]

#Exp2 vary learning rate & momentum & hidden0
N_FOLD 	= [10]
EPOCH 	= [500] 
MOMENTUM 	= [0.05,0.1,0.2,0.3,0.4,0.5]
LEARNING_RATE = [0.05,0.1,0.2,0.3,0.4,0.5]
HIDDEN0 	= []<<<<<<<<<<<<<<<<form Exp1
HIDDEN1 	= []<<<<<<<<<<<<<<<<form Exp1

#Ex3 vary EPOCH
# N_FOLD 	= [10,20,50,100] 
# EPOCH 	= [500,1000,2000,5000,800,10000]
# MOMENTUM 	= []<<<<<<<<<<<<<<<<form Exp2
# LEARNING_RATE = []<<<<<<<<<<<<<<<<form Exp2
# HIDDEN0 	= []
# HIDDEN1 	= []

AUTOMATION0 = [N_FOLD,EPOCH,MOMENTUM,LEARNING_RATE,HIDDEN0,HIDDEN1]
AUTOMATION1 = [N_FOLD,EPOCH,MOMENTUM,LEARNING_RATE,HIDDEN0]

WEKA_PATH = '-classpath "/Users/Gift/Downloads/meka-1.7.5/lib/weka.jar"' #for mac
# WEKA_PATH = '-classpath "C:\Program Files\Weka-3-6\weka.jar"'#for windows
WEKA_CLASS = 'weka.classifiers.functions.MultilayerPerceptron'

W0 = 0
X0 = 1

def autoWEKA(exp,filename):

	fn = filename.split('.')[0].split('/')[1]	
	afile = open('3stat/'+exp+'/'+fn+'-'+exp+'.csv','a+')
	count = 0

	cartesian0 = list(itertools.product(*AUTOMATION0)) #Epx1
	cartesian1 = list(itertools.product(*AUTOMATION1))
	print "weka variation = %d"%len(cartesian)

	print "START WEKA"

	# for var in cartesian0 :
	for var in cartesian1 :
		# WEKA_OPTION = ' -L %.2f -M %.2f -N %d -V 0 -S 0 -E 20 -H %d,%d -B -C -R -v -x %d'%(var[3],var[2],var[1],var[4],var[5],var[0])
		WEKA_OPTION = ' -L %.2f -M %.2f -N %d -V 0 -S 0 -E 20 -H %d -B -C -R -v -x %d'%(var[3],var[2],var[1],var[4],var[0])

		WEKA_CMD = " ".join(["java",WEKA_PATH,WEKA_CLASS,WEKA_OPTION,"-t",filename])
		count += 1
		print count
		# print WEKA_CMD
		start = datetime.datetime.now()
		result = os.popen(WEKA_CMD).read()
		# print result
		lines = result.splitlines()

		ACCU = re.search(r'Correctly Classified Instances\s+(\d+)\s+([\d.]+).*', result)
		EPE	 = 0
		MAE	 = re.search(r'Mean absolute error\s+([\d.]+).*', result)
		RMSE = re.search(r'Root mean squared error\s+([\d.]+).*', result)
		RAE	 = re.search(r'Relative absolute error\s+([\d.]+).*', result)
		RAE	 = re.search(r'Relative absolute error\s+([\d.]+).*', result)
		RRSE = re.search(r'Root relative squared error\s+([\d.]+).*', result)

		table = re.search(r'=== Detailed Accuracy By Class ===\s+([\w\t -]+)\s+([\d.\s]+).*', result)
		lines = re.split('\s+',table.group(2))
		data = [[],[],[],[],[],[],[],[],[]]
		for i,word in enumerate(lines):
			data[i%9].append(word)



		# csvdata = ",".join(map(str,list(var)+[datetime.datetime.now() - start]+[ACCU.group(2)]+[MAE.group(1)]+[RMSE.group(1)]+[RAE.group(1)]+[RRSE.group(1)]+data[2:5]+[WEKA_CMD]))
		csvdata = ",".join(map(str,list(var)+[]+[datetime.datetime.now() - start]+[ACCU.group(2)]+[MAE.group(1)]+[RMSE.group(1)]+[RAE.group(1)]+[RRSE.group(1)]+data[2:5]+[WEKA_CMD]))
		afile.write(csvdata)
	
	afile.close()
	return 1