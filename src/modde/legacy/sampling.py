from scipy.stats import qmc


class UniformEngine(qmc.QMCEngine):
    def __init__(self, d, seed=None, **kwargs):
        super().__init__(d=d, seed=seed)

    def _random(self, n=1, *, workers=1):
        return self.rng.random((n, self.d))


class GaussianEngine(qmc.QMCEngine):
    def __init__(self, d, seed=None, **kwargs):
        super().__init__(d=d, seed=seed)

    def _random(self, n=1, *, workers=1):
        return self.rng.normal(0.5, 0.5 / 3, size=(n, self.d))
