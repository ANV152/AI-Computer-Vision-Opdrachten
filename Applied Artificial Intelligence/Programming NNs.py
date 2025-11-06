import math
from loadDataset import *
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def adjust_learning_rate(initial_eta, gradient, threshold=1e-6):
    """
    Gebaseerd op de gradient wordt de learning rate aangepast
    
    Args:
        initial_eta (float): Initial learning rate.
        gradient (float): Gradient magnitude voor de huidige update.
        threshold (float): Minimaale waarde om delen door nul te voorkomen.
        
    Returns:
        float: aangepaste learning rate.
    """
    gradient_magnitude = max(abs(gradient), threshold)
    
    # If gradient is small, reduce learning rate significantly
    if abs(gradient) < threshold:
        return initial_eta * 0.1  # Slow down learning
    return initial_eta / (1 + gradient_magnitude)


class Perceptron:
    def __init__(self):
        self.inputs = []
        self.weights = []
        self.bias = 0.4
        self.output = None  # Placeholder for output


        # gebruikt voor backpropagation:
        self.delta = 0.0  # Gradient for weight updates
        self.z = 0.0  # Linear combination of inputs
        self.eta = 0.01  # Learning rate
    def initialize_weights(self):
        if len(self.inputs) == 0:
            raise ValueError("Inputs cannot be an empty list.")
        
        self.weights = [random.random() for _ in self.inputs] # Normalize weights

    def print_n_inputs(self):

        return print("Perceptron input length", len(self.inputs))
            
    def relu(self, x):
        """ReLU activation function."""
        return max(0, x)
    def calc_z(self):
        z = 0
        for i in range(len(self.inputs)):
            input_value = self.inputs[i].output if isinstance(self.inputs[i], Perceptron) else self.inputs[i]
            z += input_value * self.weights[i]
        self.z = z + self.bias
        return self.z

    def activation(self):
        z = self.calc_z()
        self.output = sigmoid(z)  # Sigmoid activation
        # self.output = self.relu(z)
        return self.output

    def update_w_b(self):
        """
        Gewichten en bias van de neuron worden hier geüpdated. 
        """     
        # adjusted_eta = adjust_learning_rate(self.eta, self.delta)
        
        for i in range(len(self.weights)):
            input_value = self.inputs[i].output if isinstance(self.inputs[i], Perceptron) else self.inputs[i]
            self.weights[i] += self.eta * self.delta * input_value

        # Update bias
        self.bias += self.eta * self.delta


class NN:
    def __init__(self):
        self.inputs = None  # Input values or input perceptrons
        self.hidden_Ls = []  # List of hidden layers (each layer is a list of perceptrons)
    
    def add_layer(self, n_neurons):
        """
        NN trainen op basis van de input_data en epochs

        Args:
            n_neurons(int): aantal neuron in de nieuwe toegevoegde layer

        Raises:
            ValueError: als er geen layers meer in het netwerk zijn gevonden, dan gooien we een error
        """
        if n_neurons <= 0:
            raise ValueError("Number of neurons must be a positive integer.")

        perceptrons = [Perceptron() for _ in range(n_neurons)]
        for perceptron in perceptrons:
            if len(self.hidden_Ls) == 0:
                # can't initialize weights yet if inputs is None. In this way we can adjust easily the input size later when training starts
                perceptron.inputs = self.inputs 
            else:
                perceptron.inputs = self.hidden_Ls[-1]
                perceptron.initialize_weights()
        self.hidden_Ls.append(perceptrons)

    def forward(self):
        """Forward propagation wordt hier uitgevoerd
        """
        # Forward propagate through all layers
        for i, layer in enumerate(self.hidden_Ls):
            #TODO: The weights are being reset in set_inputs each time. we need to fix that
            outputs = []
            for perceptron in layer:
                perceptron.activation()
                outputs.append(perceptron.output)
            # print(f"Layer {i} inputs: {[p.output if isinstance(p, Perceptron) else p for p in perceptron.inputs]}")
            # print(f"Layer {i} weights: {[p.weights for p in layer]}")
            # print(f"Layer {i} outputs : {outputs}")
            # print("-----")
    def train(self, input_data, epochs=1):
        """
        NN trainen op basis van de input_data en epochs

        Args:
            input_data (list): Een list van tuples [(input_vector, label_vector), ...].
            epochs (int): aantal keers dat het netwerk door de input_data heen draait.

        Raises:
            ValueError: als er geen layers meer in het netwerk zijn gevonden, dan gooien we een error
        """
        if len(self.hidden_Ls) == 0:
            raise ValueError("No layers found in the network.")
        input_size = len(input_data[0][0])
        if self.hidden_Ls[0][0].inputs is None:
            self.inputs = [0 for _ in range(input_size)]
            for perceptron in self.hidden_Ls[0]:
                perceptron.inputs = self.inputs
                perceptron.initialize_weights()
        for epoch in range(epochs):
            for n, y in input_data:  # Iterate over inputs and labels
                self.inputs = n
                for perceptron in self.hidden_Ls[0]:
                    perceptron.inputs = self.inputs
                self.forward()
                
                # Update deltas for the output layer
                for i in range(len(self.hidden_Ls[-1])):
                    output_perceptron = self.hidden_Ls[-1][i]

                    sig_derivative = sigmoid_derivative(output_perceptron.z)

                    error = y[i] - output_perceptron.output
                    output_perceptron.delta = sig_derivative * (error)
                    # print(error, " of perc. ", output_perceptron.output)
                    # print("weigt before: ", output_perceptron.weights[0])
                    # output_perceptron.update_w_b()
                    # print("weigt after: ", output_perceptron.weights[0])

                 # Backpropagate to hidden layers
                for i in range(len(self.hidden_Ls) - 2, -1, -1):  # Skip output layer
                    for j in range(len(self.hidden_Ls[i])):

                        hidden_perceptron = self.hidden_Ls[i][j]
                        sig_derivative = sigmoid_derivative(hidden_perceptron.z)
                        som = 0

                        for nxt_per in self.hidden_Ls[i+1]:
                            som += nxt_per.weights[j] * nxt_per.delta # som van de weights en deltas

                        hidden_perceptron.delta = sig_derivative * som
                        # print("weight before: ", hidden_perceptron.weights[0])
                        # hidden_perceptron.update_w_b()
                        # print("weigt after: ", hidden_perceptron.weights[0])
                # Update weights and biases for all perceptrons
                for layer in self.hidden_Ls:
                    for perceptron in layer:
                        # print("weigt before: ", perceptron.weights[0])
                        perceptron.update_w_b()
                        # print("weigt after: ", perceptron.weights[0])

    def predict(self, inputs):
        """
        Voorspelling van de NN op basis van de gegeven inputs
        
        Args:
            inputs (list): Lijst van input waardes.
        
        Returns:
            int: De index van de voorspelde class (0, 1, or 2).
        """
        self.inputs = inputs
        self.forward()  
        
        outputs = [perceptron.output for perceptron in self.hidden_Ls[-1]] #ouput layer uit het netwerk ophalen
        # print(outputs)
        # Find the index of the perceptron with the highest output
        print(outputs)
        predicted_class = outputs.index(max(outputs))
        return predicted_class
#TODO: MSE Toepassen in de predict  functie. 
#        Error moet ergens worden opgeslagen
#        en verwerkt met de mean squere error 
#        de afgeleide Hoe schuin de afgeleide wordt

nn = NN()
nn.add_layer(4)  # Hidden layer with 4 perceptrons
nn.add_layer(5) # hidden layer met 5 perceptrons
nn.add_layer(3)  # Output layer with 3 perceptrons
nn.train(train_data, epochs= 50)
# print(test_data[0])
# predicted_class = nn.predict(test_data[0][0])
# actual_class = test_data[0][1].index(1)
# print(f"Input: {test_data[0]}, Predicted: {predicted_class}, Actual: {actual_class}")
# test
correct_predictions = 0

for input_data, label in test_data:
    predicted_class = nn.predict(input_data)
    actual_class = label.index(1)  # Get the index of the true class from the one-hot label

    print(f"Input: {input_data}, Predicted: {predicted_class}, Actual: {actual_class}")

    if predicted_class == actual_class:
        correct_predictions += 1

accuracy = (correct_predictions / len(test_data)) * 100
print("Calculated accuracy: ", round(accuracy), " %")
