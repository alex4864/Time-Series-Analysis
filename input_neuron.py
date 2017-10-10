class InputNeuron:

	def set_input(self, input):
		self.input = input

	def get_output(self):
		return self.input

	def reset(self):
		self.input = None