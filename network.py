from neuron import Neuron
from input_neuron import InputNeuron
from output_neuron import OutputNeuron

class Network:

	def __init__(self, numInputs, hiddenLayers, numOutputs):
		#initialize neurons for each layer
		self.inputLayer = []
		for _ in range(numInputs):
			self.inputLayer.append(InputNeuron())

		self.hiddenLayers = []
		for i, layer in enumerate(hiddenLayers):
			self.hiddenLayers.append([])
			for neuron in range(layer):
				self.hiddenLayers[i].append(Neuron())

		self.outputLayer = [OutputNeuron()]

		if len(self.hiddenLayers) == 1:
			for neuron in self.hiddenLayers[0]:
				neuron.set_upstream_neurons(self.inputLayer)
				neuron.set_downstream_neurons(self.outputLayer)
		else:
			for neuron in self.hiddenLayers[0]:
				neuron.set_upstream_neurons(self.inputLayer)
				neuron.set_downstream_neurons(self.hiddenLayers[1])
			for i, layer in enumerate(self.hiddenLayers[1:-1]):
				for neuron in layer:
					neuron.set_upstream_neurons(self.hiddenLayers[i])
					neuron.set_downstream_neurons(self.hiddenLayers[i + 2])
			for neuron in self.hiddenLayers[-1]:
				neuron.set_upstream_neurons(self.hiddenLayers[-2])
				neuron.set_downstream_neurons(self.outputLayer)

		for neuron in self.outputLayer:
			neuron.set_upstream_neurons(self.hiddenLayers[-1])

	def evaluate(self, inputs):
		output = self.feed_forward(inputs)
		self.reset_neurons()
		return output

	def learn(self, inputs, targetOutputs):
		if len(inputs) != len(self.inputLayer):
			raise Exception("Size of inputs (" + str(len(inputs)) + ") does not match input layer (" + str(len(self.inputLayer)) + ")")

		if len(targetOutputs) != len(self.outputLayer):
			raise Exception("Size of targetOutputs (" + str(len(targetOutputs)) + ") does not match input layer (" + str(len(self.outputLayer)) + ")")

		self.feed_forward(inputs)
		self.back_propagate(targetOutputs)
		self.update_weights()
		self.reset_neurons()

	def feed_forward(self, inputs):
		for i, input in enumerate(inputs):
			self.inputLayer[i].set_input(input)

		for neuron in self.inputLayer:
			neuron.evaluate_output()

		for layer in self.hiddenLayers:
			for neuron in layer:
				neuron.evaluate_output()

		for neuron in self.outputLayer:
			neuron.evaluate_output()

		outputs = []
		for neuron in self.outputLayer:
			outputs.append(neuron.get_output())
		return outputs

	def back_propagate(self, targetOutputs):
		for i in range(0, len(targetOutputs)):
			self.outputLayer[i].evaluate_delta(targetOutputs[i])

		for layer in reversed(self.hiddenLayers):
			for neuron in layer:
				neuron.evaluate_delta()

	def update_weights(self):
		for neuron in self.outputLayer:
			neuron.update_weights()

		for layer in reversed(self.hiddenLayers):
			for neuron in layer:
				neuron.update_weights()

	def reset_neurons(self):
		for neuron in self.outputLayer:
			neuron.reset()

		for layer in self.hiddenLayers:
			for neuron in layer:
				neuron.reset()

		for neuron in self.inputLayer:
			neuron.reset()