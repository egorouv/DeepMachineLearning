import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from task2.samples import generate_two_gauss_data
from task2.samples import generate_spiral_data
from task2.samples import generate_circle_data
from task2.samples import generate_xor_data


class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10, activation='step', method='theorem'):
        self.weights = np.zeros(input_size)
        self.w0 = 0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.method = method
        self.activation = activation

    def step(self, x):
        return np.where(x >= 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_back(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def predict(self, x):
        return self.step(np.dot(x, self.weights) + self.w0) if self.activation == 'step' \
            else self.sigmoid(np.dot(x, self.weights) + self.w0)

    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - self.sigmoid_back(prediction) if self.method == 'gradient' else prediction
                self.weights += self.learning_rate * error * X[i]
                self.w0 += self.learning_rate * error


samples = 500
noise = 0.1

data = generate_two_gauss_data(samples, noise)

X = np.array([[point[0], point[1]] for point in data])
y = np.array([point[2] for point in data])

perceptron = Perceptron(input_size=2)

start_time = time()
perceptron.train(X, y)
end_time = time()

predictions = [perceptron.predict(x) for x in X]


# # Преобразование непрерывных предсказаний в бинарный формат с использованием порогового значения
# binary_predictions = [1 if p >= 0.5 else 0 for p in predictions]
#
# matrix = confusion_matrix(y, binary_predictions)
# accuracy = accuracy_score(y, binary_predictions) * 2.0

matrix = confusion_matrix(y, predictions)
accuracy = accuracy_score(y, predictions) * 2.0

print("perceptron")
print(matrix[0][1], " ", matrix[0][2])
print()
print(matrix[2][1], " ", matrix[2][2])
print("accuracy =", accuracy)
print("time =", end_time - start_time, "seconds")

# disp = ConfusionMatrixDisplay(matrix).plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix")
# plt.xlabel(accuracy)
# plt.show()

