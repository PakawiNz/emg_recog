import itertools
import re,sys,os
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
N_FOLD 	= [10] 
EPOCH 	= [500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2500,3000]#[500,1000,2000,5000]
MOMENTUM 	= [0.1]#<<<<<<<<<<<<<<<<form Exp2
LEARNING_RATE = [0.1]#<<<<<<<<<<<<<<<<form Exp2
HIDDEN0 	= [8]
HIDDEN1 	= [6]

#Ex4 vary Dataset amount
# N_FOLD 	= []#<<<<<<<<<<<<<<<<form Exp3
# EPOCH 	= []#<<<<<<<<<<<<<<<<form Exp3
# MOMENTUM 	= [0.1]
# LEARNING_RATE = [0.1]
# HIDDEN0 	= [8]
# HIDDEN1 	= [6]

AUTOMATION = [N_FOLD,EPOCH,MOMENTUM,LEARNING_RATE,HIDDEN0,HIDDEN1]
# AUTOMATION = [N_FOLD,EPOCH,MOMENTUM,LEARNING_RATE,HIDDEN0]

if sys.platform.startswith('win'):
	WEKA_PATH = '-classpath "C:\Program Files\Weka-3-6\weka.jar"' #for windows
elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
	WEKA_PATH = '-classpath "/Users/Gift/Downloads/meka-1.7.5/lib/weka.jar"' #for mac
elif sys.platform.startswith('darwin'):
	WEKA_PATH = '-classpath "/Users/Gift/Downloads/meka-1.7.5/lib/weka.jar"' #for mac
else:
	raise EnvironmentError('Unsupported class path')
WEKA_CLASS = 'weka.classifiers.functions.MultilayerPerceptron'

W0 = 0
X0 = 1

def autoWEKA(exp,filename):
	if sys.platform.startswith('win'):
		fn = x.split('.')[0].split('\\')[1] # for windows
	elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
		fn = filename.split('.')[0].split('/')[1]	
	elif sys.platform.startswith('darwin'):
		fn = filename.split('.')[0].split('/')[1]	
	else:
		raise EnvironmentError('Unsupported path')
	count = 0

	cartesian = list(itertools.product(*AUTOMATION))
	print "weka variation = %d"%len(cartesian)

	print "START WEKA"

	for var in cartesian :
		afile = open('3stat/'+exp+'/'+fn+'-'+exp+'.csv','a+')
		WEKA_OPTION = ' -L %.2f -M %.2f -N %d -V 0 -S 0 -E 20 -H %d,%d -B -C -R -v -x %d'%(var[3],var[2],var[1],var[4],var[5],var[0])
		# WEKA_OPTION = ' -L %.2f -M %.2f -N %d -V 0 -S 0 -E 20 -H %d -B -C -R -v -x %d'%(var[3],var[2],var[1],var[4],var[0])

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
		data = [[],[],[],[],[],[],[]]
		for i,word in enumerate(lines):
			data[i%7].append(word)

		csvdata = ",".join(map(str,list(var)+[datetime.datetime.now() - start]+[ACCU.group(2)]+[EPE]+[MAE.group(1)]+[RMSE.group(1)]+[RAE.group(1)]+[RRSE.group(1)]+data[2]+data[3]+data[4]+[WEKA_OPTION]))
		# csvdata = ",".join(map(str,list(var)+["-"]+[datetime.datetime.now() - start]+[ACCU.group(2)]+[EPE]+[MAE.group(1)]+[RMSE.group(1)]+[RAE.group(1)]+[RRSE.group(1)]+data[2]+data[3]+data[4]+[WEKA_OPTION]))
		afile.write(csvdata+'\n')
		afile.close()
	
	return 1

if __name__ == '__main__':
	import glob
	ctype = 0
	
	#autoDataTest
	# autoWEKA('Exp0','2arff/data10000-0.arff')

	# autoWEKA('Exp1','2arff/data10000-0.arff')
	
	# autoWEKA('Exp2','2arff/data10000-0.arff')

	autoWEKA('Exp3','2arff/data10000-0.arff')

	# autoWEKA('Exp4','2arff/data10000-0.arff')
	# autoWEKA('Exp4','2arff/data15000-0.arff')
	# autoWEKA('Exp4','2arff/data20000-0.arff')
	# autoWEKA('Exp4','2arff/data25000-0.arff')
	# autoWEKA('Exp4','2arff/data30000-0.arff')
	# autoWEKA('Exp4','2arff/data35000-0.arff')
	# autoWEKA('Exp4','2arff/data5000-0.arff')
