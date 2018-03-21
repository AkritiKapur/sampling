import numpy as np
import pymc3 as pm
import theano.tensor as tt
import pandas as pd

import matplotlib.pyplot as plt
from pymc3 import Uniform, Normal, Model, DiscreteUniform, find_MAP, Bernoulli

G_conditioned = np.array([[0.9, 0.1],
                 [0.1, 0.9]])

G = np.array([0.5, 0.5])

mean = np.array([50, 60])
x2_g = [0.12615662, 0.00085003]


def get_samples():

    with Model() as gene_model:
        # normal priors on the mean and variance of blood pressures for various genes.
        g1 = Bernoulli("g1", 0.5)

        g2 = Bernoulli("g2", pm.math.switch(tt.eq(g1, 0), 0.1, 0.9))
        g3 = Bernoulli("g3", pm.math.switch(tt.eq(g1, 0), 0.1, 0.9))

        mean_g1 = pm.math.switch(tt.eq(g1, 0), 50, 60)
        mean_g2 = pm.math.switch(tt.eq(g2, 0), 50, 60)
        mean_g3 = pm.math.switch(tt.eq(g3, 0), 50, 60)

        x1 = Normal("x1", mean_g1, np.sqrt(10))
        x2 = Normal("x2", mean_g2, np.sqrt(10), observed=50)
        x3 = Normal("x3", mean_g3, np.sqrt(10))

    with gene_model:
        # obtain starting values via MAP
        start = find_MAP(model=gene_model)
        # start = 100
        # instantiate sampler
        step = pm.Metropolis()

        # draw 2000 posterior samples
        gene_trace = pm.sample(5000, step=step, start=start)

    from pymc3 import traceplot
    traceplot(gene_trace)
    print(pm.summary(gene_trace))
    plt.show()

    return gene_trace


def task1(samples):
    g1s = samples['g1'][:].tolist()
    numerator = sum(1 for i in g1s if i==1)
    print(numerator/len(g1s))

def task2(samples):



if __name__ == '__main__':
    trace = get_samples()
    task1(trace)
