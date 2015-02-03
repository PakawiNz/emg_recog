import serial,sys
import os
import glob

# COMPORT = '/dev/tty.usbmodemfa131'
COMPORT = '/dev/tty.usbmodemfd121'
BAUDRATE = 57600

PACKETLEN = 17
IDXCH1 = (4,5)

class SerialManager(object) :

	def __init__(self):
		print SerialManager.serial_ports()
		ser = serial.Serial(
			port=COMPORT,
			baudrate=BAUDRATE,
		)

		ser.close()
		ser.open()

		self.ser = ser

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
		if ord(raw_package[0]) != 165 :
			return False
		if ord(raw_package[1]) != 90 :
			return False

		return True

	def __init__(self,raw_package):
		if not Package.isValid(raw_package) :
			raise Exception('Package is not valid.')

		ch1h,ch1l = IDXCH1
		self.ch1 = (ord(raw_package[ch1h]) << 8) | ord(raw_package[ch1l])


