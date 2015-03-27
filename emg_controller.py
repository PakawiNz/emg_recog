import SendKeys

forward = SendKeys.str2keys('w')[0]
backward = SendKeys.str2keys('s')[0]
turnleft = SendKeys.str2keys('a')[0]
turnright = SendKeys.str2keys('d')[0]

actionMapper = {1:turnleft,2:backward,3:turnright,4:forward}

class KeyController(object):

	def __init__(self):
		self.lastmove = 0

	def control(self,action):
		if action in actionMapper :
			newmove = actionMapper[action][0]
			if self.lastmove != newmove :
				SendKeys.key_up(self.lastmove)

			SendKeys.key_down(newmove)
			self.lastmove = newmove
		else :
			SendKeys.key_up(self.lastmove)

if __name__ == "__main__" :
	print 'noob'
	import time
	kc = KeyController()
	time.sleep(2)
	kc.control(1)
	time.sleep(5)
	kc.control(0)