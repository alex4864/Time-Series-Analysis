import numpy as np
import math
import random

class Neuron:
	LEARNING_RATE = .01 # often expressed as lambda, the constant rate at which weights are modified

	def __init__(self):
		self.downstreamNeurons = [] # set of neurons that have this neuron as an input
		self.upstreamNeurons = [] # set of neurons that make up this neuron's inputs
		self.net = None  # weighted sum of all inputs, used in error calculation
		self.output = None  # output of the neuron
		self.delta = None  # the negative of the partial of error with respect to net

		self.threshold = None
		self.weights = None

	# evaluates inputs from upstreamNeurons, putting the result in self.output
	def evaluate_output(self):
		inputs = []
		for neuron in self.upstreamNeurons:
			inputs.append(neuron.get_output())

		self.net = sum([i * j for (i, j) in zip(self.weights, inputs)]) + self.threshold
		self.output = self.sigmoid(self.net)

	def evaluate_delta(self):
		if self.net is None:
			raise Exception("self.net has not yet been calculated")

		downstreamDeltas = 0
		for neuron in self.downstreamNeurons:
			downstreamDeltas += neuron.get_weighted_delta(self)
		self.delta = self.sigmoid_prime(self.net) * downstreamDeltas

	def update_weights(self):
		if self.delta is None:
			raise Exception("self.delta has not yet been calculated")

		for i in range(0, len(self.weights)):
			self.weights[i] = self.weights[i] + self.LEARNING_RATE * self.delta * self.upstreamNeurons[i].get_output()

		self.threshold = self.threshold + self.LEARNING_RATE * self.delta

	def reset(self):
		self.net = None
		self.output = None
		self.delta = None

	def sigmoid(self, x):
		return (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x))

	def sigmoid_prime(self, x):
		return 2 / (math.exp(x) + math.exp(-x))

	def get_weights(self):
		return self.weights

	def get_output(self):
		return self.output

	def get_weighted_delta(self, upstreamNeuron):
		index = self.upstreamNeurons.index(upstreamNeuron)

		return self.delta * self.weights[index]

	def set_upstream_neurons(self, upstreamNeurons):
		self.upstreamNeurons = upstreamNeurons
		self.initialize_weights(len(upstreamNeurons))

	def initialize_weights(self, numWeights):
		self.threshold = random.random() * 2 - 1
		self.weights = []
		for i in range(0, numWeights):
			self.weights.append(random.random() * 2 - 1)

	def set_downstream_neurons(self, downstreamNeurons):
		self.downstreamNeurons = downstreamNeurons