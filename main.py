import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from network import Network
from generate_data import generate_examples_from_series
from generate_graph import generate_sparse_series_network, generate_shallow_network

def label_to_color(label):
	red = (label + 1) / 2
	green = 0
	blue = 1 - (label + 1) / 2
	alpha = 1
	return [red, green, blue, alpha]

def plot_data(data):
	xCoords = []
	yCoords = []
	labels = []
	for point in data:
		xCoords.append(point['coord'][0])
		yCoords.append(point['coord'][1])
		labels.append(point['label'])

	colors = []
	for label in labels:
		colors.append(label_to_color(label))

	plt.scatter(xCoords, yCoords, c=colors)

def plot_network(network):
	x = y = np.arange(-3.0, 3.0, 0.05)
	X, Y = np.meshgrid(x, y)
	Z = np.array([network.evaluate([x, y]) for (x,y) in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

	levels = [0]

	plt.contour(X, Y, Z, levels, colors='black')

trainingData = np.sin(np.arange(0, 5, .05))
trainingExamples = generate_examples_from_series(trainingData, 5, 7, 2)
validationData = np.sin(np.arange(0, 5, .05))
validationExamples = generate_examples_from_series(trainingData, 5, 7, 2)

graph = generate_shallow_network(7, 7, 1)
net = Network(graph)

for i in range(0, 1000):
	for example in trainingExamples:
		net.learn(example['inputs'], [example['label']])

errors = []
for i, example in enumerate(trainingExamples):
	output = net.evaluate(example['inputs'])
	errors.append(np.abs(output - example['label']))

print('Average error: ' + str(sum(errors) / len(errors)))