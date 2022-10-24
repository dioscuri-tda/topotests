from ecc import *
import pandas as pd


def sample_standarize(sample):
    return (sample - np.mean(sample, axis=0)) / np.std(sample, axis=0)


class TopoTestOnesample:
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
        Class to represent one-sample TopoTest

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

    def fit(self, rv, n_signature: int = 1000, n_test: int = 1000):
        """
        Fitting/Training phase of the TopoTest.
        Purpose of this method is to compute the average ECC and threshold value, that is used later to decide
        whether the null hypothesis H0 should be rejected or not.

        n_signature random samples from rv are drawn to compute
        the average ECC. Later another batch of n_test samples is computed to estimate the threshold value.

        :param rv: random number generator that provides a rvs(size=) method. Most likely you would like to go with
               sth like scipy.stats.rv_continuous object e.g. scipy.stats.norm
        :param n_signature: number of random samples drawn to compute average ECC
        :param n_test: number of sample drawn to compute the threshold value
        """
        # generate signature samples and test sample
        samples = [rv.rvs(size=self.sample_pts_n) * self.scaling for i in range(n_signature)]
        samples_test = [rv.rvs(size=self.sample_pts_n) * self.scaling for i in range(n_test)]
        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]
            samples_test = [sample_standarize(sample) for sample in samples_test]

        # get signatures representations of both samples
        self.representation.fit(samples)
        (self.representation_distance, self.representation_signature) = self.representation.transform(samples_test)
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

        accpect_h0 = [dp < self.representation_threshold for dp in distance_predict]
        # calculate pvalues
        pvals = [np.mean(self.representation_distance > dp) for dp in distance_predict]

        return accpect_h0, pvals


# TODO: refactor this
def TopoTestTwosample(X1, X2, norm="sup", loops=100):
    n_grids = 2000
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    def _get_ecc(X, epsmax=None):
        ecc = np.array(compute_ECC_contributions_alpha(X))
        ecc[:, 1] = np.cumsum(ecc[:, 1])
        if epsmax is not None:
            ecc = np.vstack([ecc, [epsmax, 1]])
        return ecc

    def _dist_ecc(ecc1, ecc2):
        # ecc1 and ecc2 are of equal length and have jumps in the same location
        if norm == "sup":
            return np.max(np.abs(ecc1[:, 1] - ecc2[:, 1]))
        if norm == "l1":
            return np.trapz(np.abs(ecc1[:, 1] - ecc2[:, 1]), x=ecc1[:, 0])
        if norm == "l2":
            return np.trapz((ecc1[:, 1] - ecc2[:, 1]) ** 2, x=ecc1[:, 0])

    def _interpolate(ecc, epsgrid):
        interpolator = spi.interp1d(ecc[:, 0], ecc[:, 1], kind="previous")
        y = interpolator(epsgrid)
        return np.column_stack([epsgrid, y])

    ecc1 = _get_ecc(X1)
    ecc2 = _get_ecc(X2)
    epsmax = max(np.max(ecc1[:, 0]), np.max(ecc2[:, 0]))
    epsgrid = np.linspace(0, epsmax, n_grids)
    ecc1 = _interpolate(_get_ecc(X1, epsmax), epsgrid)
    ecc2 = _interpolate(_get_ecc(X2, epsmax), epsgrid)
    D = _dist_ecc(ecc1, ecc2)

    X12 = np.vstack([X1, X2])
    distances = []
    for _ in range(loops):
        inds = np.random.permutation(n1 + n2)
        x1 = X12[inds[:n1]]
        x2 = X12[inds[n1:]]
        y1 = _interpolate(_get_ecc(x1, epsmax), epsgrid=epsgrid)
        y2 = _interpolate(_get_ecc(x2, epsmax), epsgrid=epsgrid)
        distances.append(_dist_ecc(y1, y2))
    pval = np.mean(distances > D)
    return D, pval, distances
