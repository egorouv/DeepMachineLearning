import numpy as np
import matplotlib.pyplot as plt


def generate_data(func, num_samples, epsilon_0):
    x = np.random.uniform(-1, 1, num_samples)
    epsilon = np.random.uniform(-epsilon_0, epsilon_0, num_samples)
    #epsilon = np.random.normal(0, 1, num_samples) * epsilon_0
    y = func(x) + epsilon
    return x, y


def original_function(x):
    return x * np.sin(2 * np.pi * x)


def polynomial(x, w):
    return sum(w[i] * x ** i for i in range(len(w)))


def main():
    x, y = generate_data(original_function, 30, 0.3)

    m = 7

    a = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            a[i, j] = np.sum(x ** (i + j))

    b = np.zeros(m + 1)
    for i in range(m + 1):
        b[i] = np.sum(x ** i * y)

    w = np.linalg.solve(a, b)

    plt.scatter(x, y, label='Выборка')
    x_range = np.linspace(-1.2, 1.2, 100)
    plt.plot(x_range, original_function(x_range), color='red', label='Истинная функция')
    plt.plot(x_range, polynomial(x_range, w), color='green', label='Полином степени ' + str(m))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-1, 1)
    plt.ylim(-2, 2)
    plt.title('Полиномиальная регрессия (степень ' + str(m) + ')')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
