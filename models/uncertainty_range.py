"""uncertainty_range.py - turn quantiles into a whole range of possible demands (Monte Carlo)."""
import numpy as np

from .transforms import Log1p

Z90 = 1.2816      # a normal curve's 90th percentile sits 1.2816 standard deviations above its centre
_LOG = Log1p()


class Sampler:
    """Interface: draw(n_draws, seed) -> array (n_rows, n_draws) of possible demands."""

    def draw(self, n_draws, seed=42):
        raise NotImplementedError


class TwoPieceCurve(Sampler):
    """A bell curve on the log1p scale with its own spread left and right of the median, fitted from P10 / P50 / P90."""

    def __init__(self, p10, p50, p90):
        # Responsibility: fit the curve.
        # Formula:  mu = log(1+P50);  sigma_L = (log(1+P50) - log(1+P10)) / 1.2816;  sigma_R = (log(1+P90) - log(1+P50)) / 1.2816
        # Example:  P10=120, P50=200, P90=360  ->  mu = 5.3033, sigma_L = 0.396, sigma_R = 0.457
        l10, l50, l90 = (_LOG.forward(np.asarray(v, float)) for v in (p10, p50, p90))
        self.mu, self.sig_l, self.sig_r = l50, (l50 - l10) / Z90, (l90 - l50) / Z90

    @classmethod
    def from_forecast(cls, qf):
        return cls(qf.col(0.1), qf.col(0.5), qf.col(0.9))

    def from_z(self, z):
        # Responsibility: standard-normal draws z (n_rows, n_draws) -> demand.
        # Formula:  y = max(0, exp(mu + sigma_side * z) - 1),  sigma_side = sigma_L if z < 0 else sigma_R
        # Example:  z = 0 -> 200 (median);  z = +1.2816 -> 360 (P90);  z = -1.2816 -> 120 (P10)
        sig = np.where(z < 0, self.sig_l[:, None], self.sig_r[:, None])
        return _LOG.inverse(self.mu[:, None] + sig * z)

    def draw(self, n_draws, seed=42):
        rng = np.random.default_rng(seed)
        return self.from_z(rng.standard_normal((len(self.mu), n_draws)))


class QuantileInterpolator(Sampler):
    """Sample by reading demand off the quantile function between the given levels. No bell-curve assumption; zeros stay zeros."""

    def __init__(self, levels, values):
        self.levels, self.values = np.asarray(levels, float), np.asarray(values, float)   # values: (n_rows, n_levels)

    @classmethod
    def from_forecast(cls, qf):
        return cls(qf.levels, qf.values)

    def from_u(self, u):
        # Responsibility: uniform levels u (n_rows, n_draws) -> demand by straight-line interpolation between levels.
        # Formula:  y = v_j + (u - l_j) / (l_j+1 - l_j) * (v_j+1 - v_j)  for l_j < u <= l_j+1;  beyond the outer levels: nearest value
        # Example:  levels [.1,.25,.5,.75,.9], values [20,45,80,120,150], u = 0.62  ->  80 + 0.48 * 40 = 99.2
        L, V = self.levels, self.values
        out = np.where(u <= L[0], V[:, [0]], V[:, [-1]])
        for j in range(len(L) - 1):
            inside = (u > L[j]) & (u <= L[j + 1])
            line = V[:, [j]] + (u - L[j]) / (L[j + 1] - L[j]) * (V[:, [j + 1]] - V[:, [j]])
            out = np.where(inside, line, out)
        return out

    def draw(self, n_draws, seed=42):
        rng = np.random.default_rng(seed)
        return self.from_u(rng.random((len(self.values), n_draws)))


class MonteCarlo:
    """Holds the draws of one sampler and answers questions about them."""

    def __init__(self, sampler, n_draws=2000, seed=42):
        self.samples = sampler.draw(n_draws, seed)            # (n_rows, n_draws)

    def quantiles(self, levels):
        # Responsibility: any percentile of each row's draws.   Example (R1 curve): levels [.1,.5,.9] -> [120, 200, 360]
        return np.percentile(self.samples, [100 * l for l in levels], axis=1).T

    def mean(self):
        # Responsibility: expected demand = average of the draws.   Example (R1 curve): about 225, above the median 200 (right skew)
        return self.samples.mean(axis=1)

    def prob_above(self, x):
        # Responsibility: chance demand exceeds x.   Example (R1 curve): P(demand > 300) = 18.8%
        return (self.samples > x).mean(axis=1)

    def total(self, idx):
        # Responsibility: a group's total as draws - ADD THE DRAWS FIRST, then read quantiles (quantiles do not add).
        # Example: two independent series, each P90 = 360 -> P90 of the sum is about 645, not 720
        return self.samples[idx].sum(axis=0)