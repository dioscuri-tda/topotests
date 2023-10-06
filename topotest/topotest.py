from .ecc import *
import pandas as pd
import numpy as np
import tqdm
import scipy.interpolate as spi
from scipy._lib._bunch import _make_tuple_bunch


def sample_standarize(sample):
    return (sample - np.mean(sample, axis=0)) / np.std(sample, axis=0)


TopoTestResult = _make_tuple_bunch("TopoTestResult", ["statistic", "pvalue"])


class TopoTestOnesample:
    """
        Class to represent one-sample TopoTest

    """

    def __init__(
        self,
        n: int,
        dim: int,
        significance_level: float = 0.05,
        norm: str = "sup",
        method: str = "approximate",
        scaling: float = 1.0,
        standarize: bool = False,
    ):
        """
        :param n: sample size
        :param dim: dimension  of the sample data = dimension of the null distribution
        :param significance_level: significance level
        :param norm: which norm (distance) between sample ECC and expected ECC to use.
                    Possible options are 'sup', 'l1' or 'l2'
        :param method: which method is used to construct average ECC.
                    Possible options are 'approximate' or exact.
                    Option 'exact' is time and memory(!!) consuming.
        :param scaling: Should the data be scaled prior to building the ECC.
                    There is really no point to use that other than examining asymptotic theory
        :param standarize: Should the input data be standardised by subtracting mean and dividing by standard deviation
        """
        self.fitted = False
        self.sample_pts_n = n
        self.dim = dim
        self.significance_level = significance_level
        self.method = method
        self.norm = norm
        self.standarize = standarize
        self.scaling = scaling
        if norm not in ["sup", "l1", "l2"]:
            raise ValueError(f"norm must be sup or l1 or l2, got {norm} instead")
        if method not in ["approximate", "exact"]:
            raise ValueError(f"method must be approximate or exact, got {method} instead")
        if method == "approximate":
            self.representation = ecc_representation(self.norm, mode="approximate")
        if method == "exact":
            self.representation = ecc_representation(self.norm, mode="exact")

        # fields below are set by fit procedure()
        self.representation_distance = None
        self.representation_threshold = None
        self.representation_signature = None

    def fit(self, rv, n_signature: int = 1000, n_test: int = None):
        """
        Fitting/Training phase of the TopoTest.
        Purpose of this method is to compute the average ECC and threshold value, that is used later to decide
        whether the null hypothesis H0 should be rejected or not.

        n_signature random samples from rv are drawn to compute
        the average ECC. Later another batch of n_test samples is computed to estimate the threshold value.

        :param rv: random number generator that provides a rvs(size=) method. Most likely you would like to go with
               sth like scipy.stats.rv_continuous object e.g. scipy.stats.norm
        :param n_signature: number of random samples drawn to compute average ECC
        :param n_test: number of sample drawn to compute the threshold value. If None is passed samples drawn to compute average ECC are used.
        """
        # generate signature samples and test sample
        if n_test is not None and n_test < 2:
            n_test = None

        samples = [rv.rvs(size=self.sample_pts_n) * self.scaling for i in range(n_signature)]
        if n_test:
            samples_test = [rv.rvs(size=self.sample_pts_n) * self.scaling for i in range(n_test)]
        else:
            samples_test = None
        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]
            if n_test:
                samples_test = [sample_standarize(sample) for sample in samples_test]

        # get signatures representations of both samples
        if n_test:
            self.representation.fit(samples)
            self.representation_distance, self.representation_signature = self.representation.transform(samples_test)
        else:
            self.representation_distance, self.representation_signature = self.representation.fit_transform(samples)
        self.representation_threshold = np.quantile(self.representation_distance, 1 - self.significance_level)
        self.fitted = True

    def predict(self, samples):
        """
        :param samples: list of arrays, sample or samples for which the test is run
        :return: list of booleans (True for accept the H0, False otherwise), list of p-values
        """
        if not self.fitted:
            raise RuntimeError("Cannot run predict(). Run fit() first!")
        if not isinstance(samples, list):
            samples = [samples]

        # check if input samples are np.ndarry or pandas.core.series
        for sample in samples:
            if not isinstance(sample, (np.ndarray, pd.core.frame.DataFrame)):
                raise ValueError("All samples must be np.ndarry or pd.core.frame.DataFrame")

        # check if all samples are of proper size
        if self.dim == 1:
            samples = [sample.reshape(-1, 1) for sample in samples]

        samples = [sample * self.scaling for sample in samples]

        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]

        distance_predict, _ = self.representation.transform(samples)

        # reject_h0 = [dp > self.representation_threshold for dp in distance_predict]
        # calculate pvalues
        pvals = [np.mean(self.representation_distance > dp) for dp in distance_predict]
        if len(samples) > 1:
            return TopoTestResult(distance_predict, pvals)
        else:
            return TopoTestResult(distance_predict[0], pvals[0])


def TopoTestTwosample(X1, X2, norm="sup", loops=500, n_interpolation_points=2000, verbose=False):
    """
    Function to run twos-sample TopoTest

    :param X1: first sample
    :param X2: second sample
    :param norm: norm (=distance) used to measure the distance between ECC curves
    :param loops: how many iterations of the permutation test to perform
    :param n_interpolation_points: number of points in which ECCs are interpolated
    :param verbose: should the output be produced?
    :return: value of D statistcs and pvalue
    """

    def _get_ecc(point_cloud, filtration_max=None):
        """
        Computes the normalized ECC for the point clound.

        :param point_cloud: point cloud for which the ECC must be constructed
        :param filtration_max: maximal value of filtration. This is added to the ECC to provide equal support for
                all considered ECCs
        :return: normalized ECC
        """
        ecc = np.array(compute_ecc_contributions_alpha(point_cloud))
        ecc[:, 1] = np.cumsum(ecc[:, 1])
        if filtration_max is not None:
            ecc = np.vstack([ecc, [filtration_max, 0]])
        return ecc

    def _dist_ecc(ecc1, ecc2):
        """
        Compute the distance between two ECCs assuming the jumping points are in the same locations

        :param ecc1: first ecc
        :param ecc2: second ecc
        :return: distance, controlled by norm parameter of the TopoTestTwosample function
        """
        # ecc1 and ecc2 are of equal length and have jumps in the same location
        if norm == "sup":
            return np.max(np.abs(ecc1[:, 1] - ecc2[:, 1]))
        if norm == "l1":
            return np.trapz(np.abs(ecc1[:, 1] - ecc2[:, 1]), x=ecc1[:, 0])
        if norm == "l2":
            return np.trapz((ecc1[:, 1] - ecc2[:, 1]) ** 2, x=ecc1[:, 0])

    def _interpolate(ecc, filtration_grid):
        """
        Interpolate the ECC on the filtration_grid.

        :param ecc: ECC to be interpolated
        :param filtration_grid: grid of points on which the ECC should be interpolated
        :return: interpolation of ECC over the filtration_grid

        """
        interpolator = spi.interp1d(ecc[:, 0], ecc[:, 1], kind="previous")
        y = interpolator(filtration_grid)
        return np.column_stack([filtration_grid, y])

    if len(X1.shape) == 1:
        X1 = X1.reshape(-1, 1)
    if len(X2.shape) == 1:
        X2 = X2.reshape(-1, 1)

    # run the two-sample test
    # construct trial ECC only to get the filtration_max and filtration_grid
    eccX1 = _get_ecc(point_cloud=X1)
    eccX2 = _get_ecc(point_cloud=X2)
    filtration_max = max(np.max(eccX1[:, 0]), np.max(eccX2[:, 0]))
    filtration_gird = np.linspace(0, filtration_max, n_interpolation_points)

    # construct the actual ECCs from sample data points. they are defined over the same filtration grid
    eccX1 = _interpolate(_get_ecc(point_cloud=X1, filtration_max=filtration_max), filtration_grid=filtration_gird)
    eccX2 = _interpolate(_get_ecc(point_cloud=X2, filtration_max=filtration_max), filtration_grid=filtration_gird)
    sample_dist = _dist_ecc(ecc1=eccX1, ecc2=eccX2)

    # glue sample together to and resample from it
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    X12 = np.vstack([X1, X2])
    distances = []
    for _ in tqdm.tqdm(range(loops), disable=not verbose):
        inds = np.random.permutation(n1 + n2)
        x1 = X12[inds[:n1]]
        x2 = X12[inds[n1:]]
        y1 = _interpolate(_get_ecc(point_cloud=x1, filtration_max=filtration_max), filtration_grid=filtration_gird)
        y2 = _interpolate(_get_ecc(point_cloud=x2, filtration_max=filtration_max), filtration_grid=filtration_gird)
        distances.append(_dist_ecc(ecc1=y1, ecc2=y2))
    pval = np.mean(distances > sample_dist)

    return TopoTestResult(sample_dist, pval)
