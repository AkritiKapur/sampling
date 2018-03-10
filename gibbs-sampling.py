import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

MEAN_MATRIX = [1, 0]
CORRELATION_MATRIX = [[1, -0.5], [-0.5, 3]]


def draw_gibbs_samples(iterations, mu, sigma, start_from):
    theta_1 = -1
    theta_2 = -3

    theta_1_sigma = sigma[0][1] / sigma[1][1]
    correlation = (sigma[0][1] * sigma[0][1]) / (sigma[0][0] * sigma[1][1])
    theta_1_correlation = (1 - correlation) * sigma[0][0]
    theta_2_sigma = sigma[1][0] / sigma[0][0]
    theta_2_correlation = (1 - correlation) * sigma[1][1]

    samples_theta_1 = []
    samples_theta_2 = []

    for i in range(iterations):
        theta_1 = np.random.normal(mu[0] + theta_1_sigma*(theta_2-mu[1]), theta_1_correlation)
        theta_2 = np.random.normal(mu[1] + theta_2_sigma*(theta_1-mu[0]), theta_2_correlation)

        samples_theta_1.append(theta_1)
        samples_theta_2.append(theta_2)

    return samples_theta_1, samples_theta_2


def plot_frequency_histogram(samples, mu, sigma, start, end):
    plt.hist(samples, bins=50, normed=True)
    x = np.linspace(start, end, 100)
    plt.plot(x, norm.pdf(x, mu, sigma), color='cyan')
    plt.show()


if __name__ == '__main__':
    x1_distribution, x2_distribution = draw_gibbs_samples(500000, MEAN_MATRIX, CORRELATION_MATRIX, start_from=1000)
    plot_frequency_histogram(x1_distribution, MEAN_MATRIX[0], CORRELATION_MATRIX[0][0], -2, 4)
    plot_frequency_histogram(x2_distribution, MEAN_MATRIX[1], CORRELATION_MATRIX[1][1], -10, 10)
