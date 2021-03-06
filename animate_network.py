import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from network import Network
from generate_data import generate_data

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

	return plt.scatter(xCoords, yCoords, c=colors)

def plot_network(network):
	x = y = np.arange(-3.0, 3.0, 0.05)
	X, Y = np.meshgrid(x, y)
	Z = np.array([network.evaluate([x, y]) for (x,y) in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

	levels = [0]

	plt.contour(X, Y, Z, levels, colors='black')


clusters = [[2, 2], [-1, 1], [0, 2], [1, -1], [-1, -2]]
labels = [1, 1, -1, -1, -1]

data = generate_data(clusters, labels, 1, 20)

net = Network()

fig,ax = plt.subplots()
plot_data(data)

def animate(i):
	for x in range (0, 1):
		for j in range (0, len(data)):
			net.learn(data[j]['coord'], [data[j]['label']])

	ax.clear()
	plot_data(data)

	x = y = np.arange(-3.0, 3.0, 0.05)
	X, Y = np.meshgrid(x, y)
	Z = np.array([net.evaluate([x, y]) for (x, y) in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

	cont = ax.contour(X, Y, Z, [0], color='black')
	return cont

ani = animation.FuncAnimation(fig, animate, 80)

Writer = animation.writers['imagemagick']
writer = Writer(fps=8, metadata=dict(artist='Alex Shadley'), bitrate=1800)

ani.save('classification.gif', writer=writer)