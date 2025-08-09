import numpy as np
from copy import copy

class OrnsteinUhlenbeck:
    """
    @class OrnsteinUhlenbeck
    @brief Implements an Ornstein-Uhlenbeck (OU) process for temporally correlated exploration noise.

    This stochastic process is useful in reinforcement learning algorithms such as DDPG
    to generate noise with temporal correlation, encouraging smoother exploration in
    continuous action spaces. OU noise is particularly effective in tasks where momentum
    or sustained directional actions are beneficial.

    The OU process evolves according to:
    @f[
        dx_t = \theta (\mu - x_t) dt + \sigma \sqrt{dt} \mathcal{N}(0, 1)
    @f]
    where:
    - @f$ \mu @f$: long-term mean (drift target)
    - @f$ \theta @f$: rate of mean reversion
    - @f$ \sigma @f$: noise scale
    - @f$ dt @f$: time step

    Temporal correlation arises from the mean-reversion term @f$ \theta (\mu - x_t) @f$,
    making this process different from simple Gaussian noise.
    """

    def __init__(self,
                 mu=0.0,
                 theta=0.05,
                 sigma=0.3,
                 dt=0.4,
                 x0=None,
                 min_sigma=0.01,
                 sigma_decay=0.9995):
        """
        @brief Constructs an Ornstein-Uhlenbeck process.

        @param mu           The long-term mean value toward which the process reverts.
        @param theta        The speed of mean reversion (higher values pull back to mu faster).
        @param sigma        Initial standard deviation of the noise term.
        @param dt           Time step size for the process.
        @param x0           Optional initial state. Defaults to mu if None.
        @param min_sigma    Minimum allowed sigma value when decaying over time.
        @param sigma_decay  Multiplicative decay factor applied to sigma per call to decay_sigma().
        """
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x_prev = x0 if x0 is not None else mu
        self.min_sigma = min_sigma
        self.sigma_decay = sigma_decay

    def step(self):
        """
        @brief Advances the OU process by one time step.

        @return The next value of the process after adding temporally correlated noise.
        """
        dx = self.theta * (self.mu - self.x_prev) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.x_prev += dx
        return self.x_prev

    def reset(self):
        """
        @brief Resets the process to its initial state (mu or given x0).
        """
        self.x_prev = copy(self.mu)

    def decay_sigma(self):
        """
        @brief Decays the noise scale (sigma) multiplicatively, down to a specified minimum.

        This is useful for reducing exploration over time in reinforcement learning.
        """
        self.sigma = max(self.min_sigma, self.sigma * self.sigma_decay)
