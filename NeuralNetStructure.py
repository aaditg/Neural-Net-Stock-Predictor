import numpy as np
import matplotlib.pyplot as plt

def weight_init(layer_in, layer_out):
    limit = np.sqrt(6. / (layer_in + layer_out))
    return np.random.uniform(-limit, limit, (layer_in, layer_out))

def he_init(layer_in, layer_out):
    limit = np.sqrt(2. / layer_in)
    return np.random.randn(layer_in, layer_out) * limit



class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.01):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate


        self.weights_input_hidden = weight_init(self.input_nodes, self.hidden_nodes)
        self.bias_hidden = np.zeros((1, hidden_nodes))
        self.weights_hidden_output = weight_init(self.hidden_nodes, self.output_nodes)
        self.bias_output = np.zeros((1, output_nodes))

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_gradient(x):
        x = np.clip(x, -500, 500)
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid * (1 - sigmoid)

#relu will be used instead of sigmoid in some cases

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_gradient(x):
        return np.where(x > 0, 1, 0)

    def forward_propogation(self, data):

        self.hidden_layer_input = np.dot(data, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.predictions = self.sigmoid(self.output_layer_input)
        return self.predictions

    def backward_propogation(self, data, target):


        error_output = -(target - self.predictions) * self.sigmoid_gradient(self.output_layer_input)
        gradient_w_ho = np.dot(self.hidden_layer_output.T, error_output)
        error_hidden = np.dot(error_output, self.weights_hidden_output.T) * self.sigmoid_gradient(self.hidden_layer_input)
        gradient_w_ih = np.dot(data.T, error_hidden)

        return gradient_w_ih, gradient_w_ho

    def update_weights(self, grad_w_ih, grad_w_ho):

        self.weights_input_hidden -= self.learning_rate * grad_w_ih
        self.weights_hidden_output -= self.learning_rate * grad_w_ho




    @staticmethod
    def compute_cost(true_values, predictions):
        return 0.5 * np.sum((true_values - predictions) ** 2)

    def train(self, data, target, iterations=5000):

        losses = []
        for epoch in range(iterations):
            
            self.forward_propogation(data)
            grad_w_ih, grad_w_ho = self.backward_propogation(data, target)
            
            self.update_weights(grad_w_ih, grad_w_ho)
            
            loss = self.compute_cost(target, self.predictions)
            losses.append(loss)
        
        return losses

    def plot_loss(self, losses):

        plt.plot(range(len(losses)), losses)
        plt.title('Training Loss Over Iterations')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.show()
