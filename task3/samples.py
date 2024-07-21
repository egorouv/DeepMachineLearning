import numpy as np
import matplotlib.pyplot as plt


def generate_two_gauss_data(num_samples, noise):
    points = []
    variance_scale = lambda x: 0.5 + 3.5 * (x / 0.5)

    def gen_gauss(cx, cy, label):
        nonlocal points
        for i in range(num_samples // 2):
            x = np.random.normal(cx, variance_scale(noise))
            y = np.random.normal(cy, variance_scale(noise))
            points.append((x, y, label))

    gen_gauss(2, 2, 1)
    gen_gauss(-2, -2, 0)
    return points


def generate_spiral_data(num_samples, noise):
    points = []

    def gen_spiral(delta_t, label):
        nonlocal points
        for i in range(num_samples // 2):
            r = i / (num_samples // 2) * 5
            t = 1.75 * i / (num_samples // 2) * 2 * np.pi + delta_t
            x = r * np.sin(t) + np.random.uniform(-1, 1) * noise
            y = r * np.cos(t) + np.random.uniform(-1, 1) * noise
            points.append((x, y, label))

    gen_spiral(0, 1)
    gen_spiral(np.pi, 0)
    return points


def generate_circle_data(num_samples, noise):
    points = []
    radius = 5

    def get_circle_label(p):
        return 1 if np.sqrt(p[0] ** 2 + p[1] ** 2) < (radius * 0.5) else 0

    for _ in range(num_samples // 2):
        r = np.random.uniform(0, radius * 0.5)
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noise_x = np.random.uniform(-radius, radius) * noise
        noise_y = np.random.uniform(-radius, radius) * noise
        label = get_circle_label((x + noise_x, y + noise_y))
        points.append((x, y, label))

    for _ in range(num_samples // 2):
        r = np.random.uniform(radius * 0.7, radius)
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noise_x = np.random.uniform(-radius, radius) * noise
        noise_y = np.random.uniform(-radius, radius) * noise
        label = get_circle_label((x + noise_x, y + noise_y))
        points.append((x, y, label))

    return points


def generate_xor_data(num_samples, noise):
    def get_xor_label(p):
        return 1 if p[0] * p[1] >= 0 else 0

    points = []
    for _ in range(num_samples):
        x = np.random.uniform(-5, 5)
        padding = 0.3
        x += padding if x > 0 else -padding
        y = np.random.uniform(-5, 5)
        y += padding if y > 0 else -padding
        noise_x = np.random.uniform(-5, 5) * noise
        noise_y = np.random.uniform(-5, 5) * noise
        label = get_xor_label((x + noise_x, y + noise_y))
        points.append((x, y, label))

    return points


def plot_data(data, title):
    positive_points = np.array([point[:2] for point in data if point[2] == 1])
    negative_points = np.array([point[:2] for point in data if point[2] == 0])

    plt.scatter(positive_points[:, 0], positive_points[:, 1])
    plt.scatter(negative_points[:, 0], negative_points[:, 1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.show()


samples = 500
noise = 0.1

data_generators = [
    ("Два гауссовских распределения", generate_two_gauss_data, samples, noise),
    ("Спираль", generate_spiral_data, samples, noise),
    ("Окружность", generate_circle_data, samples, noise),
    ("XOR", generate_xor_data, samples, noise)
]

# for name, data_generator, num_samples, noise in data_generators:
#     data = data_generator(num_samples, noise)
#     plot_data(data, name)


def draw_sample(X, Y, num_images_to_show=5):
    fig, axes = plt.subplots(1, num_images_to_show, figsize=(30, 12))
    pil = ToPILImage()
    for i in range(num_images_to_show):
        image, label = X[i], Y[i]
        image = pil(image.reshape(28, 28))
        axes[i].imshow(image, cmap='inferno')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.show()