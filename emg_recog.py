from emg_train import MotionModel
import numpy as np

motion = MotionModel(MotionModel.FILE_FLEX)
motion = motion.load()

def z_calc(x,mean,stdev) :
	result = -np.square(x-mean) / (2.0 * np.square(stdev))
	result = np.exp(result) #/ (stdev * np.sqrt(2 * np.pi))
	return result

def recognize(fft_result) :
	FREQ_DOMAIN = len(fft_result)
	result = 0
	# use normal distribution equation
	for i in range(FREQ_DOMAIN) :
		z_score = z_calc(fft_result[i], motion.average[i], motion.stdev[i])
		result += z_score
		# print z_score

	result /= FREQ_DOMAIN
	result = 0 if result < 0.1 else result

	return result