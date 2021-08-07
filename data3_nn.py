import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn() for i in range(6)])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derrorpred = 2 * (prediction - target)
        dpredlay1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derrorpred * dpredlay1 * dlayer1_dbias
        )
        derror_dweights = (
            derrorpred * dpredlay1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors

df = pd.read_csv("data3.txt").iloc[:, 0].str.split(" ", expand=True).reset_index()
df = df.iloc[:,0].str.split(" ",expand=True).iloc[1:,:7]
# seting column names
columns =  ["i1","i2","i3","i4","i5","i6","o"]
df.columns = columns

# splitting data set into train and test
x = df.to_numpy()
indices = np.random.permutation(x.shape[0])
training_idx, test_idx = indices[:80], indices[80:]
training, test = x[training_idx,:], x[test_idx,:]
train_input = training[:,:6].astype(float)
train_output = training[:,6:].astype(float)
learn = 0.01
nn = NeuralNetwork(learn)
training_error = nn.train(train_input, train_output, 100000)

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.title("Performance of NN with learning rate "+str(learn))
plt.show()

for i in test:
    print(nn.predict(i.astype(float)[:6]), i[6:])

