from NeuralNetStructure import NeuralNetwork
import numpy as np


#BASIC XOR FUNCTION

inputs = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
outputs = np.array([[1], [1], [0], [0]])


nn = NeuralNetwork(input_nodes=2, hidden_nodes=8, output_nodes=1, learning_rate=0.03)

losses = nn.train(inputs, outputs, iterations=5000)

nn.plot_loss(losses)

predictions = nn.forward_propogation(inputs)
print("Predictions:")
print(predictions)
