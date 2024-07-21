import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import torchvision.datasets.mnist as mnist
import torchvision.transforms as trnsfrms


class Perceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Linear(input_size, output_size, dtype=torch.float64)

    def forward(self, X):
        pred = self.layers(X)
        return torch.softmax(pred, dim=1)

    def fit(self, X: torch.Tensor, Y: torch.Tensor, epoch, batch_size, lr):
        N = X.shape[0]
        X = X.type(torch.float64)

        optimizer = optim.SGD(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for i in range(epoch):
            indices = torch.randperm(N)
            X, Y = X[indices], Y[indices]
            for j in range(0, N - batch_size, batch_size):
                X_batch, Y_batch = X[j:j + batch_size], Y[j:j + batch_size]
                self.layers.zero_grad()
                Y_pred = self(X_batch)
                loss = criterion(Y_pred, Y_batch)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        X = X.type(torch.float64)
        pred = self(X)
        return torch.argmax(pred, dim=1)


class Ensemble:
    def __init__(self, models, output_size, train_size):
        self.output_size = output_size
        self.train_size = train_size
        self.models = models

    def fit(self, X: torch.Tensor, Y: torch.Tensor, epoch, batch_size, lr):
        N = X.shape[0]
        n = int(self.train_size * N)
        for model in self.models:
            indices = torch.randperm(N)
            model.fit(X[indices[:n]], Y[indices[:n]], epoch, batch_size, lr)

    def predict(self, X):
        X = X.type(torch.float64)
        preds = torch.zeros((X.shape[0], self.output_size))
        for model in self.models:
            pred = model(X)
            preds += pred
        return torch.argmax(preds, dim=1)


def draw_heatmap(y_pred, y_true):
    with torch.no_grad():
        matrix = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(matrix, display_labels=range(10)).plot(
            include_values=True, cmap='Blues')
        plt.show()


transform = trnsfrms.Compose([
    trnsfrms.ToTensor()
])

full_train_set = mnist.MNIST('./data', download=True, train=True, transform=transform)
full_test_set = mnist.MNIST('./data', download=True, train=False, transform=transform)
full_train_x, full_train_y = full_train_set.data.reshape(-1, 28 * 28), full_train_set.targets
full_test_x, full_test_y = full_test_set.data.reshape(-1, 28 * 28), full_test_set.targets

digit_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
train_ind = [i for i, label in enumerate(full_train_y) if label in digit_labels]
test_ind = [i for i, label in enumerate(full_test_y) if label in digit_labels]

dig_train_x, dig_train_y = full_train_x[train_ind], full_train_y[train_ind]
dig_test_x, dig_test_y = full_test_x[test_ind], full_test_y[test_ind]

dig_train_y_encoded = torch.zeros(dig_train_y.shape[0], 10, dtype=torch.float64)
dig_train_y_encoded[torch.arange(dig_train_y.shape[0]), dig_train_y] = 1

dig_test_y_encoded = torch.zeros(dig_test_y.shape[0], 10, dtype=torch.float64)
dig_test_y_encoded[torch.arange(dig_test_y.shape[0]), dig_test_y] = 1

models = [Perceptron(28 * 28, 10) for _ in range(10)]
ensemble = Ensemble(models, 10, 0.8)
ensemble.fit(dig_train_x, dig_train_y_encoded, 10, 100, 0.003)
dig_test_pred = ensemble.predict(dig_test_x)

print(f'Accuracy: {accuracy_score(dig_test_y, dig_test_pred)}')
draw_heatmap(dig_test_pred, dig_test_y)
