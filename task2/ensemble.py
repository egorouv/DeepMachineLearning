import numpy as np
from time import time
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from task2.perceptron import Perceptron
from task2.samples import generate_two_gauss_data
from task2.samples import generate_spiral_data
from task2.samples import generate_circle_data
from task2.samples import generate_xor_data


class EnsemblePerceptron:
    def __init__(self, input_size, num_perceptrons, learning_rate=0.1, epochs=10):
        self.perceptrons = [Perceptron(input_size, learning_rate, epochs) for _ in range(num_perceptrons)]

    def train(self, X, y):
        subset_size = len(X) // len(self.perceptrons)
        for i, perceptron in enumerate(self.perceptrons):
            start_idx = i * subset_size
            end_idx = (i + 1) * subset_size if i < len(self.perceptrons) - 1 else len(X)
            subset_X = X[start_idx:end_idx]
            subset_y = y[start_idx:end_idx]
            perceptron.train(subset_X, subset_y)

    def predict(self, X):
        predictions = [perceptron.predict(X) for perceptron in self.perceptrons]
        return np.mean(predictions, axis=0) >= 0.5


samples = 500
noise = 0.1

data = generate_two_gauss_data(samples, noise)

X = np.array([[point[0], point[1]] for point in data])
y = np.array([point[2] for point in data])

ensemble = EnsemblePerceptron(input_size=2, num_perceptrons=5)

start_time = time()
ensemble.train(X, y)
end_time = time()

predictions = ensemble.predict(X)

matrix = confusion_matrix(y, predictions)
accuracy = accuracy_score(y, predictions) * 2.0

print("\nensemble")
print(matrix[0][1], " ", matrix[0][2])
print()
print(matrix[2][1], " ", matrix[2][2])
print("accuracy =", accuracy)
print("time =", end_time - start_time, "seconds")
