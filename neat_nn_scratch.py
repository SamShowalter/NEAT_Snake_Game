###########################################################################
#
# Neural Network Implementation from Scratch
# -- Built for a WMP KTS --
#
# Author: Sam Showalter
# Date: October 11, 2018
#
###########################################################################

###########################################################################
# Module and library imports 
###########################################################################

#Logistic and system-based imports
import os
import datetime as dt 
import copy
import sys
import pickle as pkl

#Data Science and predictive libraries
import pandas as pd 
import numpy as np 

###########################################################################
# Data formatting and restructuring for analysis 
###########################################################################


###########################################################################
# Class and constructor
###########################################################################


class NN_Scratch():
	"""
	This class implements single and multi-layer neural networks (perceptrons)
	from scratch. Primarily, this class is to be used for teaching purposes, though
	with a small amount of tweaking it could be applied to a client project. Networks
	are optimized with backpropagation via batch gradient descent. Other methods of 
	backpropagation will be implemented in the future.

	Attributes:

		network_name:				Name of the network (usually tied with type of data)
		nerwork_filename:			Filename of compressed network that will be imported
		directory: 					Directory that networks will be saved off to
		
			### Data attributes to train neural network (inputs) ###
		
		train_data:					Training dataset, feature set as a numpy Array
		labels:						Output labels, usually as one-hot encoded array
									Regression would NOT use a one-hot encoded array
		train_verbose:				If someone wants to see the progress and loss/accuracy during training
		update_learn_rate:			Boolean on if the learning rate should be updated while training occurs

			### Neural Network structure attributes ###

		num_inputs:					Number of input features into the network
		num_outputs:				Number of class labels that can be predicted
		num_hidden_layers:			Number of hidden layers in the network (must be at least 1)
		hidden_layer_nodelist:		The number of nodes to have in each hidden layer
									Number of nodes is sequential with the layers


			### Activation and backpropagation attributes ###

		activation_dict:			Dictionary storing all activation functions and derivatives
		activate:					Chosen activation function
		derivActivate:				Chosen actication function derivative
		epsilon:					Learning rate for gradient descent
		reg_lambda:					Regularization strength

			### Network output and evaluation attributes ###

		log_loss:					Log loss function to determine entropy loss (how good is fit?)
		log_loss_dif:				Difference from last log_loss to new one (is it decreasing?)
		results:					Results from feed forward process in neural network
									This is a one-hot encoded array usually. Unless specified otherwise
		preds:						Predictions derived from one-hot encoded results array
		accuracy:					Accuracy calculation

	
	"""

	def __init__(	self, 
					num_inputs, 
					num_hidden_layers, 
					hidden_layer_nodelist, 
					num_outputs,
					activation_function = "tanh",
					network_name = "Neat_Snake_Network",
					network_filename = None,
					directory = "C:\\Users\\sshowalter\\Documents\\My_Documents\\Repos\\BA_Source_Code\\Neural_Networks\\output"):
		"""
		Constructor for Neural Network from scratch class. Given the inputs, the neural network is
		created and the appropriate activation functions are chosen.

		Args:
				
				network_name:				Name of the network (usually tied with type of data)
				nerwork_filename:			Filename of compressed network that will be imported
				directory: 					Directory that networks will be saved off to

					### Data attributes to train neural network (inputs) ###
				
				train_data:					Training dataset, feature set as a numpy Array
				labels:						Output labels, usually as one-hot encoded array
											Regression would NOT use a one-hot encoded array

					### Neural Network structure attributes ###

				num_inputs:					Number of input features into the network
				num_hidden_layers:			Number of hidden layers in the network (must be at least 1)
				hidden_layer_nodelist:		The number of nodes to have in each hidden layer
											Number of nodes is sequential with the layers
				num_outputs:				Number of class labels that can be predicted

					### Activation and backpropagation attributes ###

				activation_function:		Activation function keyword. Corresponds to key in activation_dict
				epsilon:					Learning rate for gradient descent
				reg_lambda:					Regularization strength

		"""

		#Initialize neural network
		self.directory = directory
		self.network_name = network_name
		self.network = {}

		#Activation dictionary; stores all activation functions
		self.activation_dict = {"tanh": self.tanh,
								"derivtanh": self.derivTanh}
		#Network dimension information
		self.num_inputs = num_inputs
		self.num_hidden_layers = num_hidden_layers
		self.hidden_layer_nodelist = hidden_layer_nodelist
		self.num_outputs = num_outputs

		#Activation function attribution from dictionary
		self.activate = self.activation_dict[activation_function]
		self.derivActivate = self.activation_dict["deriv" + activation_function]

		#Create the neural network. Initialize random weights or import
		if network_filename is None:
			self.make_network()

		#Load the network if one already exists
		else:
			#print("Loading Network")
			self.load_network(network_filename)


###########################################################################
# Orchestration functions for training and testing
###########################################################################



###########################################################################
# Network creation and forward propagation
###########################################################################

	def make_network(self):
		"""
		Create the neural network structure. All weights and biases are initialized
		as random weights and zeros, respectively. 
		"""

		#Add first set of weights that connect to the input
		self.network["w0"] = np.random.rand(self.num_inputs, self.hidden_layer_nodelist[0]) #* self.epsilon
		self.network["b0"] = np.zeros((1, self.hidden_layer_nodelist[0]))

		#Add all intermediate hidden layers
		for i in range(self.num_hidden_layers - 1):
			self.network["w" + str(i + 1)] = np.random.rand(self.hidden_layer_nodelist[i], self.hidden_layer_nodelist[i+1]) #* self.epsilon
			self.network["b" + str(i + 1)] = np.zeros((1,self.hidden_layer_nodelist[i+1]))

		#Add weights that go to output layer
		self.network["w" + str(self.num_hidden_layers)] = np.random.rand(self.hidden_layer_nodelist[-1], self.num_outputs) #* self.epsilon
		self.network["b" + str(self.num_hidden_layers)] = np.zeros((1,self.num_outputs))
	

	def feed_forward(self, data):
		"""
		Feed forward process of pushing feature input through the network to get
		output probabilities and predictions. This is also called as the "predict"
		function for final output, and is the reason we feed in test input and test output.

		Args:
				test:			Boolean to determine if this is a prediction after training
				test_input:		test_input to replace training data if test == True
				test_output:	test_output to replace testing labels if test == True

		"""

		#Initialize throughput for this analysis
		throughput = None

		#For all layers, including output layer
		for i in range(self.num_hidden_layers + 1):

			#If this is the input layer
			if (i == 0):
				throughput = np.matmul(data, self.network["w" + str(i)]) + self.network["b" + str(i)]

			#If this is a hidden layer
			else:
				throughput = np.matmul(self.network["a" + str(i - 1)], self.network["w" + str(i)]) + self.network["b" + str(i)]

			#Add the intermediate layer to the NN dictionary store
			self.network["z" + str(i)] = throughput

			#Ase activation function and add the results to NN
			if i < self.num_hidden_layers:
				throughput = self.activate(throughput)
				self.network["a" + str(i)] = throughput

		# Determine the output probabilities of each class with softmax
		self.probs = self.softmax(throughput)

		#Determine the predictions by converting the probabilities into output
		return self.probs.argmax(axis = 1)


###########################################################################
# Supplemental network functions
###########################################################################

	def softmax(self, 
				throughput):
		"""
		Softmax function to determine pseudo probabilities for output.

		Args:

			throughput: 		Output of the feed-forward network. One-hot encoded. [num_examples x num_features]

		Returns:

			softmax probabilities as a one-hot encoded array. [num_examples x num_features]

		"""
		e_x = np.exp(throughput - np.max(throughput))
		return e_x / e_x.sum(axis = 1)[:,None]

	def save_network(self):
		"""
		Pickles and saves the neural network to be used for later if necessary.
		This way you can train a network and save it for a rainy day :).
		"""

		#Change to the correct directory
		os.chdir(self.directory)

		#Create the file name
		filename = (self.network_name + "_i" + str(self.num_inputs) +  "_h" + ".".join([str(i) for i in self.hidden_layer_nodelist]) 
					+ "_o" + str(self.num_outputs) + "_" + str(dt.datetime.now().strftime("_%Y-%m-%d_%H.%M.%S")))

		#Save off the neural network
		with open(filename + '.pickle', 'wb') as network_name:
			pkl.dump(self.network, network_name, protocol=pkl.HIGHEST_PROTOCOL)



	def load_network(	self, 
						filename):
		"""
		Loads in a network so you do not have to initialize and train
		one using random weights.

		Args:

			filename: 			Name of the file to be read in as the network
		"""

		#Change to the correct directory
		os.chdir(self.directory)

		#Read in the network
		with open(filename + ".pickle", 'rb') as network:
			self.network = pkl.load(network)

###########################################################################
# Activation functions, all stored in activation_dict
###########################################################################

	def tanh(	self, 
				input_arr):
		"""
		Tanh() activation function.

		Args:

			input: Input to be altered by activation function
		"""
		return np.tanh(input_arr)

	def derivTanh(	self, 
					input_arr):
		"""
		Tanh() derivative activation function for backpropagation

		Args:

			input: Input to be altered by activation function for backprop
		"""
		return 1 - np.power(np.tanh(input_arr), 2)


###########################################################################
# Main method for testing
###########################################################################



	
