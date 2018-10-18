import numpy as np
import sys
import copy
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from tkinter import * 
from segment import Segment
from snake_neat import Snake
from neat_nn_scratch import NN_Scratch
from neat_snake_network import Neat_Snake
import time
#matplotlib.use("Qt4Agg")

class SnakeGame(Frame):
	"""
	The Tkinter UI, responsible for drawing the board and accepting user input.
	"""
	def __init__(self,parent, edge_buffer = 30, snake_pop = 2, num_epochs = 2, snake_length = 5, seg_size = 15):
		Frame.__init__(self,parent)
		self.WIDTH = parent.winfo_screenwidth()/2
		self.HEIGHT = parent.winfo_screenheight() - 90
		self.parent = parent
		self.snake_length = snake_length
		self.live_snakes = snake_pop
		self.num_epochs = num_epochs
		self.last_shimmy_gen = 0
		self.best_fitness_gen = 0
		self.best_fitness = -np.Inf
		self.snake_pop = snake_pop
		self.new_snakelist = []
		self.first_run = True
		self.game_over = False
		self.snakelist = []
		self.seg_size = seg_size
		self.edge_buffer = edge_buffer
		self.solved = False
		self.lost = False
		self.food_dist = 1
		self.parent.geometry('%dx%d+0+0' % (self.WIDTH,self.HEIGHT))

		#Create options for where the snake can attach
		self.seg_opts = [(1,0),(0,1),(-1,0),(0,-1)]				#Right, Down, Left, Up
		self.__play_game()

	def __change_coord(self,snake):
		#print(event.keysym)
		action = snake.network.feed_forward(snake.input)
		if snake.direction_opposites[snake.direction] == action[0]:
			return

		snake.direction = action[0]


	def __initUI(self):
		#Give the window a title and pack it
		#Also make button frames so it looks good
		self.parent.title("Snake Hunger Games (I volunteer as tribute!)")
		self.pack(fill=BOTH)
		

		#Create a canvas and pack it
		self.canvas = Canvas(self,
                             width=self.WIDTH,
                             height=self.HEIGHT)
		self.canvas.pack(fill=BOTH, side=TOP)

		#Pack other frames
		self.__draw_game()
		self.__initialize_snakes()
		self.__make_food()
		self.__draw_snakes()		


	def __get_closest_snake(self, snake):
		snake.close_snake_dict = {"Left": np.Inf,
								 "Right": np.Inf,
								 "Up":np.Inf,
								 "Down":np.Inf}

		head_x = snake.body[0].x
		head_y = snake.body[0].y
		for i in range(1, len(snake.body)):

			snake_x = snake.body[i].x
			snake_y = snake.body[i].y 

			# For Up and down
			if (snake_x == head_x):
				if (snake_y < head_y):
					snake.close_snake_dict["Up"] = min(snake.close_snake_dict["Up"], 
													  (head_y - snake_y)//self.seg_size)
				else:
					snake.close_snake_dict["Down"] = min(snake.close_snake_dict["Down"], 
													  (snake_y - head_y)//self.seg_size)

			#Left and Right
			elif (snake_y == head_y):
				if (snake_x < head_x):
					snake.close_snake_dict["Left"] = min(snake.close_snake_dict["Left"], 
													  (head_x - snake_x)//self.seg_size)
				else:
					snake.close_snake_dict["Right"] = min(snake.close_snake_dict["Right"], 
													  (snake_x - head_x)//self.seg_size)

			#Make sure snake bits right behind don't freak snake out
			snake.close_snake_dict[snake.direction_opposites[snake.direction]] = np.Inf

			# For Down

	#Gets manhattan distance of head of the snake from each of the walls
	def __get_input(self, snake):

		# Get distances from different walls
		self.left_wall = (snake.body[0].x - self.edge_buffer//2)//self.seg_size
		self.right_wall = (self.WIDTH - self.edge_buffer - snake.body[0].x)//self.seg_size
		self.top_wall = (snake.body[0].y - self.edge_buffer//2)//self.seg_size
		self.bottom_wall = (self.HEIGHT - self.edge_buffer - snake.body[0].y)//self.seg_size

		self.left_danger = float(min(snake.close_snake_dict["Left"], self.left_wall))
		self.right_danger = float(min(snake.close_snake_dict["Right"], self.right_wall))
		self.up_danger = float(min(snake.close_snake_dict["Up"], self.top_wall))
		self.down_danger = float(min(snake.close_snake_dict["Down"], self.bottom_wall))

		#Get food distances
		self.left_food = (snake.body[0].x - self.food.x)//self.seg_size
		self.right_food = (self.food.x - snake.body[0].x)//self.seg_size
		self.up_food = (snake.body[0].y - self.food.y)//self.seg_size
		self.down_food = (self.food.y - snake.body[0].y)//self.seg_size

		direction_one_hot = np.zeros(4)
		direction_one_hot[snake.direction] = 1

		irrelevant_dist = 500

		if self.left_food < 0:
			self.left_food = irrelevant_dist
		if self.right_food < 0:
			self.right = irrelevant_dist
		if self.up_food < 0:
			self.up_food = irrelevant_dist
		if self.down_food < 0:
			self.down_food = irrelevant_dist

		# boolean_dist = 1

		# if self.left_food < 0:
		# 	self.left_food = boolean_dist
		# if self.right_food < 0:
		# 	self.right = boolean_dist
		# if self.up_food < 0:
		# 	self.up_food = boolean_dist
		# if self.down_food < 0:
		# 	self.down_food = boolean_dist

		danger_score = 1

		if self.left_danger > 1:
			self.left_danger = 0.0
		if self.right_danger > 1:
			self.right_danger = 0.0
		if self.up_danger > 1:
			self.up_danger = 0.0
		if self.down_danger > 1:
			self.down_danger = 0.0

		self.left_danger = min(self.left_danger, 1.0)
		self.right_danger = min(self.right_danger, 1.0)
		self.up_danger = min(self.up_danger, 1.0)
		self.down_danger = min(self.down_danger, 1.0)



		snake.input = np.array([direction_one_hot[0],
								direction_one_hot[1],
								direction_one_hot[2],
								direction_one_hot[3],
								self.left_danger,
								self.right_danger,
								self.up_danger,
								self.down_danger,
								self.left_food,
								self.right_food,
								self.up_food,
								self.down_food])

	def __initialize_snakes(self):
		'''
		Creates the snake that will be used in the game
		'''
		self.live_snakes = self.snake_pop

		for i in range(self.snake_pop):

			#Initialize
			snake = None

			if self.first_run:
				snake = Neat_Snake(self.seg_size, NN_Scratch(12, 1, [30], 4), Snake)
				
				#snake = Neat_Snake(self.seg_size, NN_Scratch(8, 2, [20,50], 4),Snake)# network_filename = "Neat_Snake_Network_i8_h20_o4__2018-10-17_13.49.26"), Snake)

				for key in snake.network.network.keys():
					if any(i in key for i in ["w", "b"]):
						snake.network.network[key] += np.random.uniform(-1, 1, size = snake.network.network[key].shape)

				de=("%02x"%np.random.randint(0,255))
				re=("%02x"%np.random.randint(0,255))
				we=("%02x"%np.random.randint(0,255))
				snake.color = "#" + de+re+we

				#snake = Neat_Snake(self.seg_size, NN_Scratch(8, 1, [50], 4), Snake)
			else:
				snake = self.snakelist[i]

			# print(self.WIDTH - self.edge_buffer - 20)
			# print(self.edge_buffer)

			#Snake head creation at some random middle point (multiple of the seg_size)
			snake_head = Segment(450,450)

			#Create snake by starting with head
			snake.addSegment(snake_head)
			
			#Everything references the location before it
			for i in range(1,self.snake_length):
				for j in self.seg_opts:
					#print(self.snake.body[i-1].x + j[0]*15)
					#print(self.snake.body[i-1].y + j[1]*15)
					if (self.__valid_position(snake.body[i-1].x + j[0]*15,snake.body[i-1].y + j[1]*15,snake)):
						newSeg = Segment(snake.body[i-1].x + j[0]*15,snake.body[i-1].y + j[1]*15)
						snake.addSegment(newSeg)
						#print("GOOD SPOT!")
						break

			#add the snake to the snakelist
			if self.first_run:
				self.snakelist.append(snake)

		if self.first_run:
			self.first_run = False


	def __make_food(self):

		while(True):
			randX = np.random.randint(self.edge_buffer//15, (self.WIDTH - self.edge_buffer//2)//15)*15
			randY = np.random.randint(self.edge_buffer//15, (self.HEIGHT - self.edge_buffer//2)//15)*15

			#Potential location for food
			tempLoc= Segment(randX, randY)

			#Check to verify a valid position
			if self.__valid_position_food(randX, randY) and not self.__onSnake(tempLoc):
				self.food = tempLoc
				return

	def __dist_to_food(self, snake):
		old_food_dist = snake.food_dist
		snake.food_dist = abs(snake.body[0].x - self.food.x) + abs(snake.body[0].y - self.food.y)
		snake.food_dist_dif = (snake.food_dist - old_food_dist)

	def __update_fitness(self, snake):
		if snake.food_dist_dif < 0:
			snake.fitness += 2*len(snake.body)
		elif snake.food_dist_dif > 0:
			snake.fitness -= 4*len(snake.body)

	def __draw_food(self):
		self.canvas.create_oval(self.food.x + 3, self.food.y+ 3, self.food.x+13, self.food.y+13, fill = "red", tags = "food")

	def __snake_grow(self, snake):

		lastPosX = snake.body[-1].x
		lastPosY = snake.body[-1].y

		for angle_i in range(len(self.seg_opts)):
			newPosX = lastPosX + self.seg_size*self.seg_opts[angle_i][0]
			newPosY = lastPosY + self.seg_size*self.seg_opts[angle_i][1]

			newSeg = Segment(newPosX, newPosY)

			if self.__valid_position(newPosX,newPosY, snake) and not self.__onSnake(newSeg):
				snake.addSegment(newSeg)
				return

	def __draw_snakes(self):
		for snake in self.snakelist:
			if (not snake.dead):
				for i in snake.body:
					self.canvas.create_rectangle(i.x,i.y, i.x + self.seg_size, i.y + self.seg_size, fill = snake.color, tags = "snake")

	def __valid_position_food(self, posX, posY):

		if posX <= self.edge_buffer or posX >= (self.WIDTH - self.edge_buffer/2):
			#print("Bad X coordinate")
			return False
		if posY <= self.edge_buffer or posY >= (self.HEIGHT - self.edge_buffer/2):
			#print("Bad Y coordinate")
			return False

		return True

	def __valid_position(self, posX, posY, snake):
		for segment in snake.body:
			#print(segment.x, segment.y)
			if posX == segment.x and posY == segment.y:
				#print("Interferes with other snake stuff")
				return False

		if posX <= self.edge_buffer or posX >= (self.WIDTH - self.edge_buffer/2):
			#print("Bad X coordinate")
			return False
		if posY <= self.edge_buffer or posY >= (self.HEIGHT - self.edge_buffer/2):
			#print("Bad Y coordinate")
			return False

		return True

	def __outOfBounds(self, snake):
		if (snake.body[0].x > (self.WIDTH - self.edge_buffer) or 
			snake.body[0].x < (self.edge_buffer//2)):
			return True

		elif (snake.body[0].y > (self.HEIGHT - self.edge_buffer) or 
			snake.body[0].y < (self.edge_buffer//2)):
			return True

		return False

	def __onSnake(self, seg):
		for snake in self.snakelist:
			for seg_index in range(len(snake.body)):
				if seg.samePoint(snake.body[seg_index]):
					return True
			return False

	def __onBody(self, snake):
		head = snake.body[0]
		for seg_index in range(1,len(snake.body)):
			if head.samePoint(snake.body[seg_index]):
				return True
		return False

	def __reproduce_probs(self):
		min_fitness = np.Inf
		fitnesses = []

		for snake in self.snakelist:
			fitnesses.append(snake.fitness)

		fitnesses = np.array(fitnesses)
		min_fitness = fitnesses.min()

		if fitnesses.max() > self.best_fitness:
			self.best_fitness = fitnesses.max()
			self.alpha = self.snakelist[np.where(fitnesses == self.best_fitness)[0][0]]
			self.alpha.reset_snake()
			self.best_fitness_gen = self.generation

		if min_fitness < 0:
			fitnesses += abs(min_fitness) + 1

		self.reproduce_probs = np.cumsum(fitnesses / (fitnesses.sum()))
		# print(self.reproduce_probs)
		# print(np.argmax(self.reproduce_probs > 0.5))

	def __choose_parents(self):
		mom_index = np.argmax(self.reproduce_probs > np.random.random())
		dad_index = mom_index
		while dad_index == mom_index:
			dad_index = np.argmax(self.reproduce_probs > np.random.random())

		mom = self.snakelist[mom_index]
		dad = self.snakelist[dad_index]

		return mom, dad

	def __crossover(self, mom, dad, rd = 0.25, ws = 0.85, ns = 1):

		def weight_swap(mom, child, layer_num, node_num):
			input_size = child.network.num_inputs
			num_swaps = np.random.randint(input_size)
			swap_indices = np.random.randint(input_size, size = num_swaps)

			for swap_index in swap_indices:

				#Child gets something from its momma
				child.network.network["w" + str(layer_num)][swap_index,node_num] = \
				mom.network.network["w" + str(layer_num)][swap_index, node_num]

		def node_swap(mom, child, layer_num, node_num):
			child.network.network["w" + str(layer_num)][:,node_num] = mom.network.network["w" + str(layer_num)][:,node_num]

		def random_difference(mom, dad, child, layer_num, biases = False):
			dict_key = "w"
			if biases:
				dict_key = "b"

			#Update weights
			weight_dif = dad.network.network[dict_key + str(layer_num)] - mom.network.network[dict_key + str(layer_num)]
			child.network.network[dict_key + str(layer_num)] += np.random.uniform(-1,1,size = weight_dif.shape) * weight_dif

		#Find the parents, and make the child the dad to start with
		child = copy.deepcopy(dad)

		for layer in range(child.network.num_hidden_layers):
			rand_choice = np.random.random()

			random_difference(mom, dad, child, layer, biases = True)

			if (rand_choice < rd):
				random_difference(mom,dad, child, layer)
				break

			else:

				for node in range(child.network.hidden_layer_nodelist[layer]):
					rand_choice = np.random.random()
					if (rand_choice < ws and rand_choice > rd):
						weight_swap(mom, child, layer, node)
					elif (rand_choice < ns and rand_choice > rd):
						node_swap(mom, child, layer, node)

		return self.__mutate(child)

		
	def __mutate(self, snake, full_shimmy = False, mutate_odds = 0.02):

		def normal_dist_shimmy(snake_input, mutate_arr):
			snake_input += mutate_array * np.random.randn(*snake_input.shape) 

		if full_shimmy:
			for layer in range(snake.network.num_hidden_layers):
				snake.network.network["w" + str(layer)] += np.random.normal(0, 2, size = snake.network.network["w" + str(layer)].shape)
				snake.network.network["b" + str(layer)] += np.random.normal(0, 2, size = snake.network.network["b" + str(layer)].shape)


		for layer in range(snake.network.num_hidden_layers):
			mutate_array = np.random.rand(*snake.network.network["w" + str(layer)].shape)
			mutate_set = (mutate_array < mutate_odds)
			normal_dist_shimmy(snake.network.network["w" + str(layer)], mutate_set)

		return snake


	def __repopulate(self):
		shimmy = False
		self.__reproduce_probs()
		self.new_snakelist.append(self.alpha)
		
		if (int(self.generation) - int(self.best_fitness_gen) >= 10) and (int(self.generation) - int(self.last_shimmy_gen) >= 10):
			self.last_shimmy_gen = self.generation
			shimmy = True
			for i in range(1, self.snake_pop//2):
				child = copy.deepcopy(self.alpha)
				child = self.__mutate(child, full_shimmy = True)
				child.reset_snake(color_reset = True)
				self.new_snakelist.append(child)


		start = 1
		if shimmy:
			start = self.snake_pop//2
		for i in range(start, self.snake_pop):
			mom, dad = self.__choose_parents()
			child = self.__crossover(mom, dad)
			child.reset_snake()
			self.new_snakelist.append(child)

		self.snakelist = self.new_snakelist
		self.new_snakelist = []



	def __draw_fitnesses(self):
		# self.canvas.create_text( 400,60,text = "Distance from left danger: " + str(self.left_danger), tags = "dists", font = ("Arial",13,"bold"))
		# self.canvas.create_text( 400,80,text = "Distance from right danger: " + str(self.right_danger), tags = "dists", font = ("Arial",13,"bold"))
		# self.canvas.create_text( 400,100,text = "Distance from up danger: " + str(self.up_danger), tags = "dists", font = ("Arial",13,"bold"))
		# self.canvas.create_text( 400,120,text = "Distance from down danger: " + str(self.down_danger), tags = "dists", font = ("Arial",13,"bold"))
		#step = 110

		self.canvas.create_text( 110,30,text = "Current Generation: " + str(self.generation), tags = "dists", font = ("Arial",13,"bold"))
		self.canvas.create_text( 310,30,text = "Max Fitness: " + str(self.best_fitness), tags = "dists", font = ("Arial",13,"bold"))
		self.canvas.create_text( 360,50,text = "Generation of Best Fitness: " + str(self.best_fitness_gen), tags = "dists", font = ("Arial",13,"bold"))


		# for i in range(len(self.snakelist)):
		# 	snake = self.snakelist[i]
		# 	if i % 10 == 0 and i > 0:
		# 		step = step + 200			
		# 	self.canvas.create_text(step, 30 + (i%10)*20,text = "Snake %s Fitness: %s"%(i,str(snake.fitness)), tags = "dists", font = ("Arial",13,"bold"))
			
			

	def __draw_game(self):
		"""
		Draws grid divided with blue lines into 3x3 squares
		"""

		#Top line
		self.canvas.create_line(0, 0, self.WIDTH, 0, fill="black", width = self.edge_buffer)
		#Bottom line
		self.canvas.create_line(0, self.HEIGHT, self.WIDTH, self.HEIGHT, fill="black", width = self.edge_buffer)
		#Left line
		self.canvas.create_line(0, 0, 0, self.HEIGHT, fill="black", width = self.edge_buffer)
		#Top line
		self.canvas.create_line(self.WIDTH, 0, self.WIDTH, self.HEIGHT, fill="black", width = self.edge_buffer)

	def __play_game(self):

		for i in range(self.num_epochs):
			self.generation = str(i)

			#print("New Epoch")
			#Draw the game and play it
			if i > 0:
				self.__repopulate()
			self.__initUI()
			#print("After epoch 2")
			timeout = time.time() + 3
			while (self.live_snakes > 0):
				snake_count = 0
				for snake in self.snakelist:
					if not snake.dead:
						snake_count += 1
				if snake_count == 0:
					self.canvas.destroy()
					break


				if time.time() >= timeout:
					print("Timeout implemented")
					self.canvas.destroy()
					break
				self.canvas.update()
				self.canvas.delete("snake")
				self.canvas.delete("dists")
				self.__draw_food()
				self.__draw_snakes()
				self.__draw_fitnesses()


				for snake in self.snakelist:
					if (not snake.dead):
						#print("Someone not dead")
						self.__dist_to_food(snake)
						self.__get_closest_snake(snake)
						self.__get_input(snake)
						self.__change_coord(snake)
						snake.move()
						self.__update_fitness(snake)
						
					#print("One Iteration")
					#self.change_coord(event)

						if (self.__outOfBounds(snake) or self.__onBody(snake)):
							#print("Snake died")
							snake.dead = True
							self.live_snakes -= 1
							#print("Live Snakes: %s"%(self.live_snakes))

							if (self.live_snakes <= 0):
								#print("GEN: %s"%(self.generation))
								#print("All snakes dead")
								self.canvas.destroy()
								break

							continue

						if (snake.body[0].samePoint(self.food)):
							self.canvas.delete("food")
							timeout = time.time() + 3
							snake.fitness += (1000 * (len(snake.body)/self.snake_length))
							self.__make_food()
							self.__snake_grow(snake)

					if (self.live_snakes <= 0):
							print("All snakes dead")
							self.canvas.destroy()
							break

						#Sys.exit(1)

				#Controls how fast the snake moves
				#time.sleep(0.01)

		self.alpha.network.save_network()
		self.parent.destroy()

			




#Run the file
#Fancy way to have this be imported for other modules too
if __name__ == '__main__':

	root = Tk()
	snake_game = SnakeGame(root, snake_length = 4, snake_pop = 100, num_epochs = 500)
	root.mainloop()

	
	
