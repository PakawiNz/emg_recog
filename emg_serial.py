import serial,sys
import os,glob,numpy as np

BAUDRATE = 57600
CENTER_VAL = 512

PACKETLEN = 17
IDXCH1H,IDXCH1L = 4,5
IDXCH2H,IDXCH2L = 6,7
IDXCH3H,IDXCH3L = 8,9
IDXCH4H,IDXCH4L = 10,11
IDXCH5H,IDXCH5L = 12,13
IDXCH6H,IDXCH6L = 14,15

class NoiseFilter(object) :
	def __init__(self,attrlist,size=100):
		self.size = size
		self.registered = zip(attrlist,map(lambda x : [], attrlist))
		self.process = self.stabilize

	def stabilize(self,package):
		for target,buffer in self.registered :
			data = package.__getattribute__(target)
			buffer.append(data)

			if len(buffer) > self.size :
				buffer.pop(0)

			avr = np.average(buffer)
			package.__setattr__(target,data-(avr-CENTER_VAL))

class Stabilizer(object) :
	def __init__(self,attrlist,size=6):
		self.size = size
		self.registered = zip(attrlist,map(lambda x : [], attrlist))
		self.process = self.stabilize

	def stabilize(self,package):
		for target,buffer in self.registered :
			data = package.__getattribute__(target)
			buffer.append(data)

			if len(buffer) > self.size :
				buffer.pop(0)

			avr = np.average(buffer)
			package.__setattr__(target,data-(avr-CENTER_VAL))

class SerialManager(object) :

	def __init__(self,stabilize=True):
		ports = SerialManager.serial_ports()
		ser = serial.Serial(
			port=ports[-1],
			baudrate=BAUDRATE,
		)

		ser.close()
		ser.open()

		self.ser = ser

		self.preprocessor = []
		if stabilize :
			self.preprocessor.append(Stabilizer(['ch1']))

	@staticmethod
	def serial_ports():
		if sys.platform.startswith('win'):
			ports = ['COM' + str(i + 1) for i in range(256)]
		elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
			ports = glob.glob('/dev/tty[A-Za-z]*')
		elif sys.platform.startswith('darwin'):
			ports = glob.glob('/dev/tty.*')
		else:
			raise EnvironmentError('Unsupported platform')

		result = []
		for port in ports:
			try:
				s = serial.Serial(port)
				s.close()
				result.append(port)
			except (OSError, serial.SerialException):
				pass
		return result

	def recieve(self):
		try :
			package = Package(self.ser.read(PACKETLEN))
			for proc in self.preprocessor :
				proc.process(package)
			return package
		except :
			self.ser.read(1)
			return self.recieve()

	def close(self):
		self.ser.close()

class Package(object) :
	
	@staticmethod
	def isValid(raw_package):
		if len(raw_package) != PACKETLEN :
			return False
		if ord(raw_package[0]) != 0xa5 :
			return False
		if ord(raw_package[1]) != 0x5a :
			return False

		return True

	def __init__(self,raw_package):
		if not Package.isValid(raw_package) :
			raise Exception('Package is not valid.')

		self.ch1 = (ord(raw_package[IDXCH1H]) << 8) | ord(raw_package[IDXCH1L])
		self.ch2 = (ord(raw_package[IDXCH2H]) << 8) | ord(raw_package[IDXCH2L])
		self.ch3 = (ord(raw_package[IDXCH3H]) << 8) | ord(raw_package[IDXCH3L])
		self.ch4 = (ord(raw_package[IDXCH4H]) << 8) | ord(raw_package[IDXCH4L])
		self.ch5 = (ord(raw_package[IDXCH5H]) << 8) | ord(raw_package[IDXCH5L])
		self.ch6 = (ord(raw_package[IDXCH6H]) << 8) | ord(raw_package[IDXCH6L])
