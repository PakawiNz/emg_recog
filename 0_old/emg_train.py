import cPickle as pickle
import os

class MotionModel(object):
	FILE_REST = 'train_REST.emg'
	FILE_FLEX = 'train_FLEX.emg'
	FILE_EXTD = 'train_EXTD.emg'
	FILE_CIRI = 'train_CIRI.emg'
	FILE_CIRO = 'train_CIRO.emg'

	def __init__(self,filename):
		self.filename = filename
		if os.path.isfile(filename) :
			self.loadable = True
		else :
			self.loadable = False

	def load(self):
		if not self.loadable : 
			raise Exception("Can't load motion which never saved.")
		with open(self.filename, 'rb') as input:
			return pickle.load(input)

	def save(self):
		self.loadable = True
		with open(self.filename, 'wb') as output:
			pickler = pickle.Pickler(output, -1)
			pickler.dump(self)