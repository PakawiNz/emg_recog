import importer
from emg_utils import getPath_arff,getPath_csv,getPath_stat
from emg_weka import getWekaPath,getStat_WEKA

import itertools
import re,os
import datetime

#Exp0 correct command /
# N_FOLD 	= [2] #[10]
# EPOCH 	= [50]
# MOMENTUM 	= [0.2]
# LEARNING_RATE = [0.3]
# HIDDEN0 	= [8]
# HIDDEN1 	= [2] #None

#Exp1 vary HIDDEN0, HIDDEN1 >>> test 1 or 2 Layer
# N_FOLD 	= [10]
# EPOCH 	= [500] 
# MOMENTUM 	= [0.1]
# LEARNING_RATE = [0.1]
# HIDDEN0 	= [9,8,7,6,5,4,3,2,1]
# HIDDEN1 	= [7,6,5]

#Exp2 vary learning rate & momentum & hidden0
# N_FOLD 	= [10]
# EPOCH 	= [500] 
# MOMENTUM 	= [0.05,0.1,0.2,0.3,0.4,0.5]
# LEARNING_RATE = [0.05,0.1,0.2,0.3,0.4,0.5]
# HIDDEN0 	= [8]#<<<<<<<<<<<<<<<<form Exp1
# HIDDEN1 	= [6]#<<<<<<<<<<<<<<<<form Exp1

#Ex3 vary EPOCH
# N_FOLD 	= [10] #[20]
# EPOCH 	= [500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2500,3000]#[500,1000,2000,5000]
# MOMENTUM 	= [0.1]#<<<<<<<<<<<<<<<<form Exp2
# LEARNING_RATE = [0.1]#<<<<<<<<<<<<<<<<form Exp2
# HIDDEN0 	= [8]
# HIDDEN1 	= [6]

#Ex4 vary Dataset amount
N_FOLD 	= [10]#<<<<<<<<<<<<<<<<form Exp3
EPOCH 	= [2000]#<<<<<<<<<<<<<<<<form Exp3
MOMENTUM 	= [0.1]
LEARNING_RATE = [0.1]
HIDDEN0 	= [8]
HIDDEN1 	= [6]

if HIDDEN1 == None :
	AUTOMATION = [N_FOLD,EPOCH,MOMENTUM,LEARNING_RATE,HIDDEN0]
else :
	AUTOMATION = [N_FOLD,EPOCH,MOMENTUM,LEARNING_RATE,HIDDEN0,HIDDEN1]

WEKA_PATH = getWekaPath()
WEKA_CLASS = 'weka.classifiers.functions.MultilayerPerceptron'
WEKA_OPTION = ''

W0 = 0
X0 = 1

def autoWEKA(exp,filename,note=0):
	count = 0

	cartesian = list(itertools.product(*AUTOMATION))
	print "weka variation = %d"%len(cartesian)
	print "START WEKA"

	for var in cartesian :
		stat = open(getPath_stat(exp,filename,note=note),'w')
		if len(AUTOMATION) == 6 :
			WEKA_OPTION = ' -L %.2f -M %.2f -N %d -V 0 -S 0 -E 20 -H %d,%d -B -C -R -v -x %d'%(var[3],var[2],var[1],var[4],var[5],var[0])
		elif len(AUTOMATION) == 5 :
			WEKA_OPTION = ' -L %.2f -M %.2f -N %d -V 0 -S 0 -E 20 -H %d -B -C -R -v -x %d'%(var[3],var[2],var[1],var[4],var[0])
		else :
			print 'AUTOMATION confuse -- please check AUTOMATION configuration'
			return 
		WEKA_CMD = " ".join(["java",WEKA_PATH,WEKA_CLASS,WEKA_OPTION,"-t",getPath_arff(filename)])
		count += 1
		print count
		print WEKA_OPTION+' '+getPath_arff(filename) 
		start = datetime.datetime.now()
		result = os.popen(WEKA_CMD).read()
		# print result
		outStat = getStat_WEKA(result,1)

		if len(AUTOMATION) == 6 :
			csvdata = ",".join(map(str,
				list(var)+[datetime.datetime.now() - start]+outStat+[WEKA_OPTION]))
		elif len(AUTOMATION) == 5 :
			csvdata = ",".join(map(str,
				list(var)+["-"]+[datetime.datetime.now() - start]+outStat+[WEKA_OPTION]))
		stat.write(csvdata+'\n')
		stat.close()
	

if __name__ == '__main__':
	#autoDataTest
	# autoWEKA('noop','data5000')
	# autoWEKA('noop','150206')

	# autoWEKA('Exp0','data5000')
	# autoWEKA('Exp0','data10000','')

	# autoWEKA('Exp1','data10000','')
	
	# autoWEKA('Exp2','data10000','')

	# autoWEKA('Exp3','data10000','')

	# autoWEKA('Exp4','data5000')
	autoWEKA('Exp4','data10000')
	autoWEKA('Exp4','data15000')
	# autoWEKA('Exp4','data20000')
	autoWEKA('Exp4','data25000')
	# autoWEKA('Exp4','data30000')
	# autoWEKA('Exp4','data500')
	# autoWEKA('Exp4','data1000')
	# autoWEKA('Exp4','data1500')
	# autoWEKA('Exp4','data2000')
	# autoWEKA('Exp4','data2500')
	# autoWEKA('Exp4','data3000')
	# autoWEKA('Exp4','data3500')
	# autoWEKA('Exp4','data4000')
