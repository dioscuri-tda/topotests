import numpy as np
import gudhi as gd
import scipy.interpolate as spi
import random


def compute_ECC_contributions_alpha(point_cloud: np.ndarray):
    """
    Computes controbutions to ECC build on top of Alpha Complex

    :param point_cloud: data points (array)
    :return: list of tuples. First element of each tuple is a radius at which
            ECC jumps, second element is a jump height
    """
    if len(point_cloud.shape) == 1:
        point_cloud = point_cloud.reshape(-1, 1)
    alpha_complex = gd.AlphaComplex(points=point_cloud)
    simplex_tree = alpha_complex.create_simplex_tree()

    ecc = {}

    for s, f in simplex_tree.get_filtration():
        dim = len(s) - 1
        ecc[f] = ecc.get(f, 0) + (-1) ** dim

    # remove the contributions that are 0
    to_del = []
    for key in ecc:
        if ecc[key] == 0:
            to_del.append(key)
    for key in to_del:
        del ecc[key]

    return sorted(list(ecc.items()), key=lambda x: x[0])


class ecc_representation:
    """
    A class to represent an ECC

    Attributes
    ----------
    xs : list of float
        List of ECC jumping x coordinates
    representation: list of float
        List that represent the ECC values at jumping points xs
    max_range : float
        Max filtration value for the ECC
    n_interpolation_points : int
        How many points is used to represent the ECC
    norm: str
        Which norm (distance) is used to compute the distance between two ECC curves
        options are 'sup', 'l1', 'l2'
    mode: str
        How jumping points of the ecc are determined. ECC is a cadlag function therefore when computing sum/average
        of ECCs that have jumping point in different locations one needs to decide how to merge them.
        Options are:
            * 'exact': jumping points of the two ECCs will be joined (i.e. jumps of ECC1+ECC2 occurs in jumping points
            of ECC1 and jumping points of ECC2. This leads to huge memory consumption when average over many ECCs.
            * 'approximate': there is a list of jumping points (selected during the pilot run) and ECCs are
            interpolated over that grid.
        In practise only 'approximate mode should be used'
    approximate_n_pilot: int
        Number of trials ECCs that are used to determine the optimal locations of grid points.
    fitted: bool
        Was the fit() method run?

    Methods
    -------
    fit(samples):
        Construct the average ECC based on list of samples provided.

    transform(samples):
        Returns an ECC computed, for each sample in list of samples, on the same grid as during the fitting.

    """

    def __init__(self, norm="sup", n_interpolation_points=20000, mode="approximate"):
        self.xs = None
        self.representation = None
        self.max_range = -np.Inf
        self.n_interpolation_points = n_interpolation_points
        self.norm = norm
        self.mode = mode
        self.approximate_n_pilot = 100
        self.fitted = False

    def fit(self, samples):
        # TODO: this implementation is memomy-ineffcient - we don't have to store all ecc to compute the mean
        # TODO: this becomes a problem for Rips
        # TODO: To be fixed in later releases
        self.max_range = -np.Inf
        eccs = []
        jumps = set()
        if self.mode == "exact":
            for sample in samples:
                ecc = np.array(compute_ECC_contributions_alpha(sample))
                ecc[:, 1] = np.cumsum(ecc[:, 1])
                jumps.update(ecc[:, 0])  # FIXME: ecc[:, 0] is stored in eccs anyway
                self.max_range = max(self.max_range, ecc[-1, 0])
                eccs.append(ecc)
            self.xs = np.sort(list(jumps))
            self.representation = self.xs * 0
            # extend all ecc so that it include the max_range
            for ecc in eccs:
                ecc_extended = np.vstack([ecc, [self.max_range, 1]])
                interpolator = spi.interp1d(ecc_extended[:, 0], ecc_extended[:, 1], kind="previous")
                y_inter = interpolator(self.xs)
                self.representation += y_inter
            self.representation /= len(samples)
        else:
            # find jump positions based on given number of trial ecc curves
            approximate_n_trials = np.min([self.approximate_n_pilot, len(samples)])
            trial_samples = random.choices(samples, k=approximate_n_trials)
            jumps = set()
            for sample in trial_samples:
                ecc = np.array(compute_ECC_contributions_alpha(sample))
                self.max_range = max([self.max_range, ecc[-1, 0]])
                jumps.update(ecc[:, 0])
            jumps = np.sort(list(jumps))
            jumps_step = int(len(jumps) / self.n_interpolation_points)
            jumps_step = max(jumps_step, 1)
            self.xs = jumps[::jumps_step]
            self.representation = self.xs * 0
            # interpolate ECC curves on the grid
            for sample in samples:
                ecc = np.array(compute_ECC_contributions_alpha(sample))
                ecc[:, 1] = np.cumsum(ecc[:, 1])
                # cut ecc on self.max_range
                range_ind = ecc[:, 0] < self.max_range
                ecc = ecc[range_ind, :]
                # add ecc max to the end
                ecc_extended = np.vstack([ecc, [self.max_range, 1]])
                interpolator = spi.interp1d(ecc_extended[:, 0], ecc_extended[:, 1], kind="previous")
                y_inter = interpolator(self.xs)
                self.representation += y_inter
            self.representation /= len(samples)
        self.fitted = True

    def transform(self, samples):
        if not self.fitted:
            raise RuntimeError("Run fit() before transform()")

        dist = []
        representations = []
        for sample in samples:
            ecc = np.array(compute_ECC_contributions_alpha(sample))
            ecc[:, 1] = np.cumsum(ecc[:, 1])
            range_ind = ecc[:, 0] < self.max_range
            ecc = ecc[range_ind, :]
            ecc = np.vstack([ecc, [self.max_range, 1]])
            interpolator = spi.interp1d(ecc[:, 0], ecc[:, 1], kind="previous")
            representation = interpolator(self.xs)
            representations.append(representation)
            if self.norm == "l1":
                dist.append(np.trapz(np.abs(representation - self.representation), x=self.xs))
            elif self.norm == "l2":
                dist.append(np.trapz((representation - self.representation) ** 2, x=self.xs))
            else:  # sup
                dist.append(np.max(np.abs(representation - self.representation)))

        return dist, representations
