import networkx as nx
from neuron import Neuron
from input_neuron import InputNeuron
from output_neuron import OutputNeuron

class Network:

	def __init__(self, topology):
		if not nx.is_directed_acyclic_graph(topology):
			raise Exception('Graph must be directed acyclic graph')

		neurons = []
		self.inputNeurons = []
		self.hiddenNeurons = []
		self.outputNeurons = []
		for node, nodeData in topology.nodes(data=True):
			if nodeData['type'] == 'input':
				newNeuron = InputNeuron()
				neurons.append(newNeuron)
				self.inputNeurons.append(newNeuron)

			if nodeData['type'] == 'hidden':
				newNeuron = Neuron()
				neurons.append(newNeuron)
				self.hiddenNeurons.append(newNeuron)

			if nodeData['type'] == 'output':
				newNeuron = OutputNeuron()
				neurons.append(newNeuron)
				self.outputNeurons.append(newNeuron)

		# some weirdness here because networkx graphs start at 1 and python lists start at 0
		for node in topology.nodes:
			for u, v in topology.out_edges(node):
				neurons[u - 1].add_downstream_neuron(neurons[v - 1])
				neurons[v - 1].add_upstream_neuron(neurons[u - 1])

		for neuron in self.hiddenNeurons:
			neuron.initialize_weights()

		for neuron in self.outputNeurons:
			neuron.initialize_weights()

	def evaluate(self, inputs):
		output = self.feed_forward(inputs)
		self.reset_neurons()
		return output

	def learn(self, inputs, targetOutputs):
		if len(inputs) != len(self.inputNeurons):
			raise Exception("Size of inputs (" + str(len(inputs)) + ") does not match input layer (" + str(len(self.inputLayer)) + ")")

		if len(targetOutputs) != len(self.outputNeurons):
			raise Exception("Size of targetOutputs (" + str(len(targetOutputs)) + ") does not match input layer (" + str(len(self.outputLayer)) + ")")

		self.feed_forward(inputs)
		self.back_propagate(targetOutputs)
		self.update_weights()
		self.reset_neurons()

	def feed_forward(self, inputs):
		for i, input in enumerate(inputs):
			self.inputNeurons[i].set_input(input)

		outputs = []
		for neuron in self.outputNeurons:
			outputs.append(neuron.get_output())
		return outputs

	def back_propagate(self, targetOutputs):
		for i, targetOutput in enumerate(targetOutputs):
			self.outputNeurons[i].evaluate_delta(targetOutputs[i])

		for neuron in self.hiddenNeurons:
			neuron.evaluate_delta()

	def update_weights(self):
		for neuron in self.hiddenNeurons:
			neuron.update_weights()
		for neuron in self.outputNeurons:
			neuron.update_weights()

	def reset_neurons(self):
		for neuron in self.hiddenNeurons:
			neuron.reset()
		for neuron in self.outputNeurons:
			neuron.reset()