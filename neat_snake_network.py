import pandas as pd
import numpy as np 
from snake_neat import Snake

class Neat_Snake(Snake):

	def __init__(self, snake_length, network, Snake):
		self.fitness = 0
		self.dead = False
		self.food_dist = 1
		self.snake = Snake
		self.snake.__init__(self, snake_length)
		
		self.network = network

	def updateFitness(self):
		pass
