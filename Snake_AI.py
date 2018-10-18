import numpy as np
import pandas as pd
from tkinter import * 
from segment import Segment
import os
from snake import Snake
import time
#matplotlib.use("Qt4Agg")

class SnakeGame(Frame):
	"""
	The Tkinter UI, responsible for drawing the board and accepting user input.
	"""
	def __init__(self,parent, edge_buffer = 30, snake_length = 5, seg_size = 15):
		Frame.__init__(self,parent)
		self.WIDTH = parent.winfo_screenwidth()/2
		self.HEIGHT = parent.winfo_screenheight() - 90
		self.parent = parent
		self.snake_length = snake_length
		self.seg_size = seg_size
		os.chdir("C:\\Users\\sshowalter\\Documents\\My_Documents\\Repos\\NEAT_Snake_Game\\output\\")
		self.training_data = np.load("neat_train_data.npy").tolist()
		self.test_data = np.load("neat_test_data.npy").tolist()
		self.edge_buffer = edge_buffer
		self.solved = False
		self.lost = False

		self.parent.geometry('%dx%d+0+0' % (self.WIDTH,self.HEIGHT))
		self.parent
		self.__initUI()


	def change_coord(self,event):
		#print(event.keysym)
		print(event.keysym)
		if self.snake.direction_opposites[self.snake.direction] == event.keysym:
			return

		self.snake.direction = event.keysym


	def __initUI(self):
		#Give the window a title and pack it
		#Also make button frames so it looks good
		self.parent.title("Snake AI Game")
		self.pack(fill=BOTH)
		

		#Create a canvas and pack it
		self.canvas = Canvas(self,
                             width=self.WIDTH,
                             height=self.HEIGHT)
		self.canvas.pack(fill=BOTH, side=TOP)

		self.parent.bind_all("<Key>", self.change_coord)

		#Pack other frames

		#Draw the game and play it
		self.__draw_game()
		self.__initialize_snake()
		self.__make_food()

		self.__draw_snake()
		self.__play_game()

	def __get_closest_snake(self):
		self.close_snake_dict = {"Left": np.Inf,
								 "Right": np.Inf,
								 "Up":np.Inf,
								 "Down":np.Inf}

		head_x = self.snake.body[0].x
		head_y = self.snake.body[0].y
		for i in range(1, len(self.snake.body)):

			snake_x = self.snake.body[i].x
			snake_y = self.snake.body[i].y 

			# For Up and down
			if (snake_x == head_x):
				if (snake_y < head_y):
					self.close_snake_dict["Up"] = min(self.close_snake_dict["Up"], 
													  (head_y - snake_y)//self.seg_size)
				else:
					self.close_snake_dict["Down"] = min(self.close_snake_dict["Down"], 
													  (snake_y - head_y)//self.seg_size)

			#Left and Right
			elif (snake_y == head_y):
				if (snake_x < head_x):
					self.close_snake_dict["Left"] = min(self.close_snake_dict["Left"], 
													  (head_x - snake_x)//self.seg_size)
				else:
					self.close_snake_dict["Right"] = min(self.close_snake_dict["Right"], 
													  (snake_x - head_x)//self.seg_size)

			#Make sure snake bits right behind don't freak snake out
			self.close_snake_dict[self.snake.direction_opposites[self.snake.direction]] = np.Inf

	#Gets manhattan distance of head of the snake from each of the walls
	def __get_input(self):

		self.__get_closest_snake()

		# Get distances from different walls
		self.left_wall = (self.snake.body[0].x - self.edge_buffer//2)//self.seg_size
		self.right_wall = (self.WIDTH - self.edge_buffer - self.snake.body[0].x)//self.seg_size
		self.top_wall = (self.snake.body[0].y - self.edge_buffer//2)//self.seg_size
		self.bottom_wall = (self.HEIGHT - self.edge_buffer - self.snake.body[0].y)//self.seg_size

		self.left_danger = float(min(self.close_snake_dict["Left"], self.left_wall))
		self.right_danger = float(min(self.close_snake_dict["Right"], self.right_wall))
		self.up_danger = float(min(self.close_snake_dict["Up"], self.top_wall))
		self.down_danger = float(min(self.close_snake_dict["Down"], self.bottom_wall))

		#Get food distances
		self.left_food = max((self.snake.body[0].x - self.food.x)//self.seg_size, -1)
		self.right_food = max((self.food.x - self.snake.body[0].x)//self.seg_size, -1)
		self.up_food = max((self.snake.body[0].y - self.food.y)//self.seg_size, -1)
		self.down_food = max((self.food.y - self.snake.body[0].y)//self.seg_size, -1)

		self.input = np.array([self.left_danger,
								self.right_danger,
								self.up_danger,
								self.down_danger,
								self.left_food,
								self.right_food,
								self.up_food,
								self.down_food])
		self.training_data.append(self.input)

	def __initialize_snake(self):
		'''
		Creates the snake that will be used in the game
		'''
		self.snake = Snake(self.seg_size)
		# print(self.WIDTH - self.edge_buffer - 20)
		# print(self.edge_buffer)

		#Snake head creation at some random middle point (multiple of the seg_size)
		snake_head = Segment(450,450)

		#Create snake by starting with head
		self.snake.addSegment(snake_head)

		#Create options for where the snake can attach
		seg_opts = [(1,0),(0,1),(-1,0),(0,-1)]
		
		#Everything references the location before it
		for i in range(1,self.snake_length):
			for j in seg_opts:
				#print(self.snake.body[i-1].x + j[0]*15)
				#print(self.snake.body[i-1].y + j[1]*15)
				if (self.__valid_position(self.snake.body[i-1].x + j[0]*15,self.snake.body[i-1].y + j[1]*15)):
					newSeg = Segment(self.snake.body[i-1].x + j[0]*15,self.snake.body[i-1].y + j[1]*15)
					self.snake.addSegment(newSeg)
					#print("GOOD SPOT!")
					break

	def __make_food(self):

		while(True):
			randX = np.random.randint(self.edge_buffer//15, (self.WIDTH - self.edge_buffer//2)//15)*15
			randY = np.random.randint(self.edge_buffer//15, (self.HEIGHT - self.edge_buffer//2)//15)*15

			#Potential location for food
			tempLoc= Segment(randX, randY)

			if self.__valid_position(randX, randY) and not self.__onSnake(tempLoc):
				self.food = tempLoc
				return

	def __draw_food(self):
		self.canvas.create_oval(self.food.x + 3, self.food.y+ 3, self.food.x+13, self.food.y+13, fill = "red", tags = "food")

	def __snake_grow(self):

		lastPosX = self.snake.body[-1].x
		lastPosY = self.snake.body[-1].y

		possible_angles = [(0,1), (0,-1), (1,0), (-1,0)]

		for angle_i in range(len(possible_angles)):
			newPosX = lastPosX + self.seg_size*possible_angles[angle_i][0]
			newPosY = lastPosY + self.seg_size*possible_angles[angle_i][1]

			newSeg = Segment(newPosX, newPosY)

			if self.__valid_position(newPosX,newPosY) and not self.__onSnake(newSeg):
				self.snake.addSegment(newSeg)
				return

	def __draw_snake(self):
		for i in self.snake.body:
			self.canvas.create_rectangle(i.x,i.y, i.x + self.seg_size, i.y + self.seg_size, fill = "blue", tags = "snake")

	def __valid_position(self, posX, posY):
		for segment in self.snake.body:
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

	def __outOfBounds(self):
		if (self.snake.body[0].x > (self.WIDTH - self.edge_buffer) or 
			self.snake.body[0].x < (self.edge_buffer//2)):
			return True

		elif (self.snake.body[0].y > (self.HEIGHT - self.edge_buffer) or 
			self.snake.body[0].y < (self.edge_buffer//2)):
			return True

	def __onSnake(self, seg):

		for seg_index in range(len(self.snake.body)):
			if seg.samePoint(self.snake.body[seg_index]):
				return True
		return False

	def __onBody(self):
		head = self.snake.body[0]
		for seg_index in range(1,len(self.snake.body)):
			if head.samePoint(self.snake.body[seg_index]):
				return True
		return False

	def __draw_distances(self):
		self.canvas.create_text( 400,60,text = "Distance from left wall: " + str(self.left_dist), tags = "dists", font = ("Arial",13,"bold"))
		self.canvas.create_text( 400,80,text = "Distance from right wall: " + str(self.right_dist), tags = "dists", font = ("Arial",13,"bold"))
		self.canvas.create_text( 400,100,text = "Distance from top wall: " + str(self.top_dist), tags = "dists", font = ("Arial",13,"bold"))
		self.canvas.create_text( 400,120,text = "Distance from bottom wall: " + str(self.bottom_dist), tags = "dists", font = ("Arial",13,"bold"))


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

		while (not self.solved and not self.lost):
			self.canvas.update()
			self.canvas.delete("snake")
			self.canvas.delete("dists")
			self.__get_input()
			self.test_data.append(self.snake.int_direction[self.snake.direction])
			self.snake.move()
			self.__draw_snake()
			self.__draw_food()
			#self.__draw_distances()
			#print("One Iteration")
			#self.change_coord(event)

			if (self.__outOfBounds() or self.__onBody()):
				self.lost = True
				print("Game Over")
				print(len(self.training_data))
				os.chdir("C:\\Users\\sshowalter\\Documents\\My_Documents\\Repos\\NEAT_Snake_Game\\output\\")
				np.save("neat_train_data", self.training_data)
				np.save("neat_test_data", np.array(self.test_data))
				print(len(self.test_data))
				sys.exit(1)
				return

			if (self.snake.body[0].samePoint(self.food)):
				self.canvas.delete("food")
				self.__make_food()
				self.__snake_grow()

			if (self.solved or self.lost):
				return

			
			#Controls how fast the snake moves
			time.sleep(0.07)




#Run the file
#Fancy way to have this be imported for other modules too
if __name__ == '__main__':
	root = Tk()
	snake_game = SnakeGame(root, snake_length = 50)

	root.mainloop()
	# os.chdir("C:\\Users\\sshowalter\\Documents\\My_Documents\\Repos\\NEAT_Snake_Game\\output\\")
	# a = np.load("neat_train_data.npy")
	# b = np.load("neat_test_data.npy")

	# n_values = 4
	# print(b)
	# print(np.eye(n_values)[b])

	
	
