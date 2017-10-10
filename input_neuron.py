class InputNeuron:

	def set_input(self, input):
		self.input = input

	def evaluate_output(self):
		self.output =  self.input

	def get_output(self):
		return self.output

	def reset(self):
		self.input = None