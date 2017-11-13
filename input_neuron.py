class InputNeuron:

	def set_input(self, input):
		self.input = input

	def get_output(self):
		return self.input

	def reset(self):
		self.input = None

	def add_upstream_neuron(self, neuron):
		return

	def add_downstream_neuron(self, neuron):
		return