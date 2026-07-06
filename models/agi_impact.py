import numpy as np
from scipy import interpolate


class AGIModel:
    def __init__(self, cdf_points, wage_multiplier, returns_multiplier, m, years):
        """
        Initialize AGI model with CDF points and impact parameters.

        Args:
            cdf_points: dict with years as keys and cumulative probabilities as values
            wage_multiplier: Factor by which wages change after AGI
            returns_multiplier: Factor by which investment returns change after AGI
            m: Number of simulations
            years: Number of years to simulate
        """
        self.cdf_points = cdf_points
        self.wage_multiplier = wage_multiplier
        self.returns_multiplier = returns_multiplier
        self.m = m
        self.years = years

        self._validate_cdf_points()

        # Generate AGI timing for each simulation
        self.agi_timing = self._generate_agi_timing()

        # Create multiplier arrays
        self.wage_multipliers = self._create_multiplier_array(self.wage_multiplier)
        self.returns_multipliers = self._create_multiplier_array(
            self.returns_multiplier
        )

    def _validate_cdf_points(self):
        """A CDF must have probabilities in [0, 1] that don't decrease as
        the horizon grows — e.g. P(AGI within 5y) can't exceed P(AGI
        within 10y). Inverse-CDF sampling on non-monotonic points would
        produce nonsensical arrival times, so fail loudly instead."""
        years = np.array(sorted(self.cdf_points.keys()))
        probs = np.array([self.cdf_points[y] for y in years])
        if np.any((probs < 0) | (probs > 1)):
            raise ValueError(f"AGI CDF probabilities must be in [0, 1]: {self.cdf_points}")
        if np.any(np.diff(probs) < 0):
            raise ValueError(
                "AGI CDF probabilities must not decrease as the year "
                f"horizon grows: {self.cdf_points}"
            )

    def _generate_agi_timing(self):
        """Generate AGI arrival times for each simulation."""
        # Generate uniform random numbers
        u = np.random.uniform(0, 1, self.m)

        # Create inverse CDF points (sorted by year; dict order is not
        # guaranteed to be)
        x = np.array(sorted(self.cdf_points.keys()))
        y = np.array([self.cdf_points[year] for year in x])

        # Add endpoints if not provided
        if 0 not in x:
            x = np.insert(x, 0, 0)
            y = np.insert(y, 0, 0)
        if 100 not in x:
            x = np.append(x, 100)
            y = np.append(y, 1)

        # Create inverse CDF interpolation
        inv_cdf = interpolate.interp1d(
            y, x, kind="linear", bounds_error=False, fill_value=(0, 100)
        )

        # Get AGI timing for each simulation
        agi_timing = inv_cdf(u)

        return np.floor(agi_timing).astype(int)

    def _create_multiplier_array(self, multiplier):
        """Create array of multipliers for each simulation and year."""
        multipliers = np.ones((self.m, self.years))
        for i in range(self.m):
            if self.agi_timing[i] < self.years:
                multipliers[i, self.agi_timing[i] :] = multiplier
        return multipliers

    def get_wage_multipliers(self):
        """Get array of wage multipliers for each simulation and year."""
        return self.wage_multipliers

    def get_returns_multipliers(self):
        """Get array of returns multipliers for each simulation and year."""
        return self.returns_multipliers

    def get_agi_timing(self):
        """Get array of AGI timing for each simulation."""
        return self.agi_timing
