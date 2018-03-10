import numpy as np

from scipy import stats

MAJOR_MAP = {1: 'comp', 0: 'bus'}
UNIVERSITY_MAP = {1: 'cu', 0: 'metro'}


def get_intelligence_sample(mu=100, sigma=15):
    """
    Intelligence is an individual's Intelligence Quotient (IQ), typically in the 70-130 range.
    Gets 1 sample for the provided mean and standard deviation of the normal distribution.
    :param mu: Mean of the normal distribution
    :param sigma: Standard deviation of the distribution
    :return:
    """
    return np.random.normal(mu, sigma)


def get_university_sample(I):
    probability = 1 / (1 + np.exp(-(I - 100) / 5))
    if np.random.rand() < probability:
        return 1
    return 0


def get_major_sample(I):
    probability = 1 / (1 + np.exp(-(I - 110) / 5))
    if np.random.rand() < probability:
        return 1
    return 0


def get_salary_sample(I, major, university):
    return np.random.gamma(0.1 * I + major + 3 * university, 5)


def draw_samples(num_samples):
    samples = []
    for i in range(num_samples):
        I = get_intelligence_sample()
        university = get_university_sample(I)
        major = get_major_sample(I)
        salary = get_salary_sample(I, major, university)
        weight = stats.gamma.pdf(salary, 5)
        sample = (I, major, university, salary, weight)
        samples.append(sample)

    return samples


def get_posterior(major, university, salary, samples):
    count_numerator = 0
    count_denominator = 0
    for sample in samples:
        if round(sample[3]) == salary:
            count_denominator += sample[4]
        if sample[1] == major and sample[2] == university and round(sample[3]) == salary:
            count_numerator += sample[4]

    return count_numerator / count_denominator


def task(salary, samples):
    print(get_posterior(major=0, university=1, salary=salary, samples=samples))
    print(get_posterior(major=0, university=0, salary=salary, samples=samples))
    print(get_posterior(major=1, university=0, salary=salary, samples=samples))
    print(get_posterior(major=1, university=1, salary=salary, samples=samples))


def likelihood_weighting():
    samples = draw_samples(100000)
    task(120, samples)
    print('*'*100)
    task(60, samples)
    print('*' * 100)
    task(20, samples)

if __name__ == '__main__':
    likelihood_weighting()
