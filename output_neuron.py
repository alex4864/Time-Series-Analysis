import numpy as np
import math
import random
from neuron import Neuron

# class behaves very similarly to neuron, but with different delta calculations
class OutputNeuron(Neuron):
	LEARNING_RATE = .01 # often expressed as lambda, the constant rate at which weights are modified

	def __init__(self):
		super().__init__()

	def evaluate_delta(self, targetOutput):
		self.delta = (targetOutput - self.output) * self.sigmoid_prime(self.net)