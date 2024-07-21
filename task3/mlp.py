import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from task3.samples import generate_two_gauss_data
from task3.samples import generate_xor_data
from task3.samples import generate_circle_data
from task3.samples import generate_spiral_data
from sklearn.model_selection import KFold


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_func):
        super(MultiLayerPerceptron, self).__init__()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                if activation_func == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation_func == 'tanh':
                    layers.append(nn.Tanh())
                elif activation_func == 'relu':
                    layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def prepare_data(generator, num_samples, noise):
    data = generator(num_samples, noise)
    inputs = torch.tensor([[point[0], point[1]] for point in data], dtype=torch.float32)
    labels = torch.tensor([point[2] for point in data], dtype=torch.long)
    return list(zip(inputs, labels))


def train_model(model, train_data, learning_rate, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, labels in train_data:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.unsqueeze(0), labels.unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")


def test_model(model, test_data):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_data:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 0)
            total += labels.numel()
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def k_fold_cross_validation(model_class, data, k, learning_rate, epochs, input_size, hidden_sizes, output_size, activation_func):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_index, val_index) in enumerate(kf.split(data)):
        print(f"Fold {fold + 1}/{k}")

        train_data = [data[i] for i in train_index]
        val_data = [data[i] for i in val_index]

        model = model_class(input_size, hidden_sizes, output_size, activation_func)

        train_model(model, train_data, learning_rate, epochs)
        accuracy = test_model(model, val_data)
        accuracies.append(accuracy)

        print(f"Validation Accuracy for fold {fold + 1}: {accuracy}%")

    print(f"Average Accuracy: {np.mean(accuracies)}%")
    return np.mean(accuracies)


samples = 500
noise = 0.1

data_generators = [
    ("Два гауссовских распределения", generate_two_gauss_data),
    ("Спираль", generate_spiral_data),
    ("Окружность", generate_circle_data),
    ("XOR", generate_xor_data)
]

activation_functions = ['sigmoid', 'tanh', 'relu']
results = {}

for name, generator in data_generators:
    print(f"Testing dataset: {name}")
    data = prepare_data(generator, samples, noise)
    input_size = 2
    hidden_sizes = [3, 3]
    output_size = 2
    learning_rate = 0.01
    epochs = 50
    k = 5

    for activation_func in activation_functions:
        print(f"Using activation function: {activation_func}")
        accuracy = k_fold_cross_validation(MultiLayerPerceptron, data, k, learning_rate, epochs, input_size,
                                           hidden_sizes, output_size, activation_func)
        results[(name, activation_func)] = accuracy

for key, value in results.items():
    print(f"Dataset: {key[0]}, Activation Function: {key[1]}, Average Accuracy: {value}%")
