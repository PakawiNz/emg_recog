import itertools
import re,sys,os
import datetime
from emg_arff import getPath_arff,getPath_csv,getPath_stat

def getWekaPath() :
	if sys.platform.startswith('win'):
		WEKA_PATH = '-classpath "C:\Program Files\Weka-3-7\weka.jar"' #for windows
	elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin') or sys.platform.startswith('darwin'):
		WEKA_PATH = '-classpath "/Users/Gift/Downloads/meka-1.7.5/lib/weka.jar"' #for mac
	else:
		raise EnvironmentError('Unsupported class path')
	return WEKA_PATH

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
DATA = [[],[],[],[],[],[],[],[],[]]

W0 = 0
X0 = 1

def getStat_WEKA(resultString,isList=0): #isList == 1 : list of stat, 0 : dict of stat
	ACCU = re.search(r'Correctly Classified Instances\s+(\d+)\s+([\d.]+).*', resultString).group(2)
	EPE	 = ''
	MAE	 = re.search(r'Mean absolute error\s+([\d.]+).*', resultString).group(1)
	RMSE = re.search(r'Root mean squared error\s+([\d.]+).*', resultString).group(1)
	RAE	 = re.search(r'Relative absolute error\s+([\d.]+).*', resultString).group(1)
	RRSE = re.search(r'Root relative squared error\s+([\d.]+).*', resultString).group(1)

	table = re.search(r'=== Detailed Accuracy By Class ===\s+([\w\t -]+)\s+([\d.\s?]+).*', resultString)
	print '\t\t '+table.group(2)
	lines = re.split('\s+',table.group(2))
	for i,word in enumerate(lines):
		DATA[i%len(DATA)].append(word)

	listStat = [ACCU]+[EPE]+[MAE]			\
			+[RMSE]+[RAE]+[RRSE]			\
			+DATA[2]+DATA[3]+DATA[4]

	dictStat = {'ACC':ACCU,'EPE':EPE,'MAE':MAE,									\
				'RMS':RMSE,'RAE':RAE,'RRS':RRSE,	\
				'1PS':map(float,DATA[2]),'2RC':map(float,DATA[3]),'3FM':map(float,DATA[4])}
	if isList : 
		return listStat
	else :
		return dictStat

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
