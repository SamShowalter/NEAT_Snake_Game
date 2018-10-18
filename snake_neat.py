import numpy as np

class Snake():

	def __init__(self, seg_size, direction = 2):
		self.start_direction = direction 
		self.direction = direction
		self.fitness = 0
		self.seg_size = seg_size
		self.body = []

		self.direction_opposites = {1:0,
									3:2,
									2:3,
									0:1}

	def addSegment(self, segment):
		self.body.append(segment) 

	def reset_snake(self, color_reset = False):
		self.direction = self.start_direction
		self.fitness = 0
		self.body = []
		self.dead = False

		if color_reset:
			de=("%02x"%np.random.randint(0,255))
			re=("%02x"%np.random.randint(0,255))
			we=("%02x"%np.random.randint(0,255))
			self.color = "#" + de+re+we

	def move_helper(self,move_x,move_y):
		
		for seg_index in reversed(range(1, len(self.body))):
			self.body[seg_index].updateLocation(self.body[seg_index-1])

		self.body[0].shiftLocation(move_x*(self.seg_size),move_y*(self.seg_size))

		#Print statement for debugging
		#print(self.body[0].x,self.body[0].y)

	def change_direction(self, direction):
		self.direction = direction

	def move(self):

		if (self.direction == 0):			#UP
			self.move_helper(0,-1)	
		elif (self.direction == 1):		#DOWN
			self.move_helper(0,1)
		elif (self.direction == 2):		#LEFT
			self.move_helper(-1,0)
		elif (self.direction == 3):		#RIGHT
			self.move_helper(1,0)





