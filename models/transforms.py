"""transforms.py - reversible changes of scale applied to demand before a model sees it."""
import numpy as np


class Transform:
    """Interface: forward() before training / feeding a model, inverse() on its output."""

    def forward(self, y):
        raise NotImplementedError

    def inverse(self, z):
        raise NotImplementedError


class Identity(Transform):
    # Responsibility: change nothing (the model sees raw demand). inverse() only keeps demand >= 0.
    def forward(self, y):
        return np.asarray(y, float)

    def inverse(self, z):
        return np.clip(np.asarray(z, float), 0, None)


class Log1p(Transform):
    # Responsibility: shrink huge demand so big rows stop dominating the loss; fully reversible.
    # Formula:  z = log(1 + y)          y = exp(z) - 1
    # Example:  y = 200 -> z = 5.303 -> back to 200;   y = 0 -> z = 0 -> back to 0
    # Quantiles survive it: median of [2, 5, 9] = 5; median of log1p values = 1.792; expm1(1.792) = 5.
    def forward(self, y):
        return np.log1p(np.asarray(y, float))

    def inverse(self, z):
        return np.clip(np.expm1(np.asarray(z, float)), 0, None)