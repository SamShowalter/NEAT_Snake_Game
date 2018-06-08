

class Segment():

	def __init__(self,X,Y):
		self.x = X 
		self.y = Y

	def updateLocation(self,X,Y):
		self.x = X
		self.y = Y

	def samePoint(self, seg):
		if (self.x == seg.x and self.y == seg.y):
			return True
		return False

	def updateLocation(self,seg):
		self.x = seg.x
		self.y = seg.y

	def shiftLocation(self,shiftX,shiftY):
		self.x += shiftX
		self.y += shiftY
	