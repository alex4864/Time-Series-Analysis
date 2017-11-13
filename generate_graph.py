import networkx as nx

def generate_shallow_network(inputs, hidden, outputs):
	G = nx.DiGraph()

	inputRange = range(0, inputs)
	hiddenRange = range(inputs, inputs + hidden)
	outputRange = range(inputs + hidden, inputs + hidden + 1)

	G.add_nodes_from(inputRange, type='input')
	G.add_nodes_from(hiddenRange, type='hidden')
	G.add_nodes_from(outputRange, type='output')

	for i in inputRange:
		for j in hiddenRange:
			G.add_edge(i, j)

	for i in hiddenRange:
		for j in outputRange:
			G.add_edge(i, j)

	return G

def generate_sparse_series_network(inputs):
	layerRanges = []
	layerRanges.append(range(inputs))
	total = inputs
	for i in range(inputs):
		rangeSize = inputs - i
		layerRanges.append(range(total, total + rangeSize))
		total += rangeSize

	G = nx.DiGraph()
	G.add_nodes_from(layerRanges[0], type = 'input')
	for i in range(1, inputs):
		G.add_nodes_from(layerRanges[i], type = 'hidden')
	G.add_nodes_from(layerRanges[inputs], type = 'output')

	# connections from input layer to first computational layer
	for node in layerRanges[0]:
		G.add_edge(node, node + inputs)

	# connections within layers
	for layerIndex, layerRange in enumerate(layerRanges[1:-1]):
		for node in layerRange[:-1]:
			G.add_edge(node, node + 1)
			G.add_edge(node, node + (inputs - layerIndex))
		for node in layerRange[1:]:
			G.add_edge(node, node + (inputs - layerIndex - 1))

	return G