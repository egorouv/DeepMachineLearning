import numpy as np
import matplotlib.pyplot as plt


def generate_uniform_data(func, num_samples, epsilon_0):
    x = np.random.uniform(-1, 1, num_samples)
    epsilon = np.random.uniform(-epsilon_0, epsilon_0, num_samples)
    y = func(x) + epsilon
    return x, y


def generate_normal_data(func, num_samples, epsilon_0):
    x = np.random.uniform(-1, 1, num_samples)
    epsilon = np.random.normal(0, 1, num_samples) * epsilon_0
    y = func(x) + epsilon
    return x, y


def f1(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def f2(x):
    return x * np.sin(2 * np.pi * x)


def main():
    a, b, c, d = np.random.uniform(-3, 3, 4)

    x1_uniform, y1_uniform = generate_uniform_data(lambda x: f1(x, a, b, c, d), 100, 0.1)
    x1_normal, y1_normal = generate_normal_data(lambda x: f1(x, a, b, c, d), 100, 0.1)
    x2_uniform, y2_uniform = generate_uniform_data(f2, 100, 0.1)
    x2_normal, y2_normal = generate_normal_data(f2, 100, 0.1)

    plt.figure(figsize=(12, 10))

    plt.subplot(221)
    plt.scatter(x1_uniform, y1_uniform, color='blue', label='Выборка с ошибкой')
    plt.plot(np.linspace(-1, 1, 100),
             f1(np.linspace(-1, 1, 100), a, b, c, d), color='red', label='Истинная функция')
    plt.title('Полиномиальная функция (равномерное распределение)')
    plt.legend()

    plt.subplot(222)
    plt.scatter(x1_normal, y1_normal, color='green', label='Выборка с ошибкой')
    plt.plot(np.linspace(-1, 1, 100),
             f1(np.linspace(-1, 1, 100), a, b, c, d), color='red', label='Истинная функция')
    plt.title('Полиномиальная функция (нормальное распределение)')
    plt.legend()

    plt.subplot(223)
    plt.scatter(x2_uniform, y2_uniform, color='orange', label='Выборка с ошибкой')
    plt.plot(np.linspace(-1, 1, 100),
             f2(np.linspace(-1, 1, 100)), color='red', label='Истинная функция')
    plt.title('Синусоидальная функция (равномерное распределение)')
    plt.legend()

    plt.subplot(224)
    plt.scatter(x2_normal, y2_normal, color='purple', label='Выборка с ошибкой')
    plt.plot(np.linspace(-1, 1, 100),
             f2(np.linspace(-1, 1, 100)), color='red', label='Истинная функция')
    plt.title('Синусоидальная функция (нормальное распределение)')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
