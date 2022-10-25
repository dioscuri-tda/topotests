#
# Some helpers implementation of the varius probablity distribution functions that were used to run
# TopoTests and Kolmogorov-Smirnov tests.
#
# This is actually not a part of the TopoTest package per-se
#

import numpy as np
import scipy.stats as st


class GaussianMixture:
    def __init__(self, locations, scales, probas):
        """
        Implementation of Gaussian Mixtures (univariate)

        :param locations: vector of location parameters
        :param scales: vector of scale parameters
        :param probas: vector of mixture coefficients
        """
        if not (len(locations) == len(scales) and len(scales) == len(probas)):
            raise ValueError("Wrong number of components for Gaussian Mixture")
        self.locations = locations
        self.scales = scales
        self.n_gauss = len(locations)
        self.gauss_rv = [st.norm(loc, scale) for loc, scale in zip(locations, scales)]
        probas_sum = np.sum(probas)
        probas = [proba / probas_sum for proba in probas]
        self.probas = probas

    def rvs(self, size):
        """
        Draw a random sample from the GaussianMixture distribution

        :param size: size of a sample
        :return: array: (size, dim) - random sample
        """
        inds = st.rv_discrete(values=(range(self.n_gauss), self.probas)).rvs(size=size)
        samples = [self.gauss_rv[ind].rvs(size=1)[0] for ind in inds]
        return np.array(samples)

    def cdf(self, pts):
        """
        Cumulative Distribution Function of the GaussianMixture distribution.

        :param pts: list of points in which CDF should be determined
        :return: list of CDF values at pts
        """

        cdf = 0
        for p, rv in zip(self.probas, self.gauss_rv):
            cdf += p * rv.cdf(pts)
        return cdf


class GaussianMixtureMulti:
    def __init__(self, locations, covs, probas, label=None):
        """
        Implementation of Gaussian Mixtures (multivariate)

        :param locations: vector of location parameters
        :param covs: vector of covarance matrices
        :param probas: vector of mixture coefficients
        :param label: string used to label given distribution
        """
        # locations - vector of location parameters
        # covs - vector of covarance matrices
        # probas - vector of mixture coefficients
        if not (len(locations) == len(covs) and len(covs) == len(probas)):
            raise ValueError("Wrong number of components for Gaussian Mixture")
        self.locations = locations
        self.covs = covs
        self.n_gauss = len(locations)
        self.gauss_rv = [st.multivariate_normal(mean=loc, cov=cov) for loc, cov in zip(locations, covs)]
        probas_sum = np.sum(probas)
        probas = [proba / probas_sum for proba in probas]
        self.probas = probas
        self.label = label

    def rvs(self, size):
        """
        Draw a random sample from the GaussianMixtureMulti distribution

        :param size: size of a sample
        :return: array: (size, dim) - random sample
        """
        inds = st.rv_discrete(values=(range(self.n_gauss), self.probas)).rvs(size=size)
        samples = [self.gauss_rv[ind].rvs(size=1) for ind in inds]  # this is slow but good enough for now
        return np.array(samples)

    def cdf(self, pts):
        """
        Cumulative Distribution Function of the GaussianMixtureMulti distribution.

        :param pts: list of points in which CDF should be determined
        :return: list of CDF values at pts
        """
        cdf = 0
        for p, rv in zip(self.probas, self.gauss_rv):
            cdf += p * rv.cdf(pts)
        return cdf


class MultivariateDistribution:
    """
    General class to represent the multivariate distributions that are Cartesian products of
    (uncorrelated) univariates.
    Mind: It is not possible to introduce the correlation in this setup.

    """

    def __init__(self, univariates, label=None, shift_and_scale=False):
        """

        :param univariates: list of univarites distributions - each must provide a rvs() method
        :param label: str: (optional) identifier of the distribution
        :param shift_and_scale: bool should the distribution be standardised (by subtracting the marginals means and
                dividing by marginals standard deviations).
        """
        self.univariates = univariates
        self.label = label
        self.dim = len(univariates)
        self.shift_and_scale = shift_and_scale
        self.shift_vec = []
        self.scale_vec = []
        for uni in self.univariates:
            if self.shift_and_scale:
                self.shift_vec.append(uni.stats(moments="m"))
                self.scale_vec.append(np.sqrt(uni.stats(moments="v")))
            else:
                self.shift_vec.append(0)
                self.scale_vec.append(1)
        self.shift_vec = np.array(self.shift_vec)
        self.scale_vec = np.array(self.scale_vec)

    def rvs(self, size):
        """
        Draw a random sample from the distribution

        :param size: size of a sample
        :return: array: (size, dim) - random sample
        """
        sample = []
        if self.shift_and_scale:
            for univariate, shift_val, scale_val in zip(self.univariates, self.shift_vec, self.scale_vec):
                sample.append((univariate.rvs(size) - shift_val) / scale_val)
        else:
            for univariate in self.univariates:
                sample.append(univariate.rvs(size))
        return np.transpose(sample)

    def cdf(self, pts):
        """
        Cumulative Distribution Function of the distribution.
        FIXME: this work only for multivariate distributions with diagonal covariance matrix
        No correlations between axies are allowed

        :param pts: list of points in which CDF should be determined
        :return: list of CDF values at pts
        """
        if self.dim == 1:
            pts = pts * self.scale_vec[0]
            pts = pts + self.shift_vec
            pts = [pts]
        else:
            for i in range(self.dim):
                pts[i] = pts[i] * self.scale_vec[i]
            pts = pts + self.shift_vec
        cdf = 1
        for pt, univariate in zip(pts, self.univariates):
            cdf *= univariate.cdf(pt)
        return cdf


class MultivariateGaussian:
    """
    Implementation of the MG distribution considered in the paper

    """

    def __init__(self, dim, a, label=None):
        self.dim = dim
        self.label = label
        self.cov = np.ones((dim, dim)) * a + np.identity(dim) * (1 - a)
        self.mean = [0] * dim
        self.rv = st.multivariate_normal(mean=self.mean, cov=self.cov)

    def rvs(self, size):
        """
        Draw a random sample from the MultivariateGaussian

        :param size: size of a sample
        :return: array: (size, dim) - random sample
        """
        return self.rv.rvs(size)

    def cdf(self, pts):
        """
        Cumulative Distribution Function of the MultivariateGaussian

        :param pts: list of points in which CDF should be determined
        :return: list of CDF values at pts
        """
        return self.rv.cdf(pts)
