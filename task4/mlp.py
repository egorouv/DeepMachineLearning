import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class MLP(nn.Module):
    def __init__(self, hidden_layers_num=3, neyrons_num=5, activation_func=nn.ReLU):
        super().__init__()
        hidden_layers = []
        for i in range(hidden_layers_num - 1):
            hidden_layers.append(nn.Linear(neyrons_num, neyrons_num, dtype=torch.float64))
            hidden_layers.append(activation_func())
        self.model = nn.Sequential(
            nn.Linear(1, neyrons_num, dtype=torch.float64),
            activation_func(),
            *hidden_layers,
            nn.Linear(neyrons_num, 1, dtype=torch.float64),
        )

    def forward(self, x):
        return self.model(x)

    def fit(self, X: torch.Tensor, Y: torch.Tensor, epochs=3000, batch_size=32, lr=0.03):
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            indices = torch.randperm(X.shape[0])
            x, y = X[indices], Y[indices]
            for i in range(0, X.shape[0], batch_size):
                x_batch, y_batch = x[i:i + batch_size], y[i:i + batch_size]

                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        return self.forward(X)


def draw_polinom_regression(x1, y1, x2, y2, xx, yy, title):
    plt.scatter(xx, yy, color="blue")
    plt.plot(x1, y1, color="green")
    plt.plot(x2, y2, color="red")

    plt.ylim(y1.min() - 1, y1.max() + 1)
    plt.title(title)
    plt.show()


a, b, c, d = np.random.uniform(low=-3, high=3, size=4)
x1 = np.linspace(-1, 1, 1000)
y1 = a * x1 ** 3 + b * x1 ** 2 + c * x1 + d

N = 20
xx = np.random.uniform(low=-1, high=1, size=N)
yy = a * xx ** 3 + b * xx ** 2 + c * xx + d + np.random.normal(0, 0.5, N)

x2 = np.linspace(-1, 1, 10000)

num_of_layers = 3
neyrons_num = 5
lr = 0.03
activation_func = nn.Tanh

mlp = MLP(num_of_layers, neyrons_num, activation_func)
mlp.fit(torch.tensor(xx).reshape(-1, 1), torch.tensor(yy).reshape(-1, 1), batch_size=16, epochs=1000, lr=lr)
y2 = mlp.predict(torch.tensor(x2).reshape(-1, 1)).detach().numpy()

draw_polinom_regression(x1, y1, x2, y2, xx, yy, f'Layers: {num_of_layers}, Neurons: {neyrons_num}, LR: {lr}')

num_of_layers = 6
neyrons_num = 10
lr = 0.03
activation_func = nn.Tanh

mlp = MLP(num_of_layers, neyrons_num, activation_func)
mlp.fit(torch.tensor(xx).reshape(-1, 1), torch.tensor(yy).reshape(-1, 1), batch_size=16, epochs=10000, lr=lr)
y2 = mlp.predict(torch.tensor(x2).reshape(-1, 1)).detach().numpy()

draw_polinom_regression(x1, y1, x2, y2, xx, yy, f'Layers: {num_of_layers}, Neurons: {neyrons_num}, LR: {lr}')
