import numpy as np
from copy import copy

class OrnsteinUhlenbeck:
    """
    OrnsteinUhlenbeck implements the Ornstein-Uhlenbeck process for generating temporally correlated noise.

    This process is commonly used in reinforcement learning to add noise to actions for exploration, especially in continuous action spaces.

    Attributes:
        mu (float): The long-term mean of the process.
        theta (float): The rate of mean reversion.
        sigma (float): The volatility parameter.
        dt (float): The time step size.
        x0 (float): The previous value of the process.

    Methods:
        step():
            Advances the process by one time step and returns the new value.

        reset():
            Resets the process to the initial state.

    @brief Ornstein-Uhlenbeck process for temporally correlated noise generation.
    """
    def __init__(self, mu=0.0, theta=0.05, sigma=0.25, dt=1.0, x0=None):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x_prev = x0 if x0 is not None else mu

    def step(self):
        dx = self.theta * (self.mu - self.x_prev) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.x_prev += dx
        return self.x_prev

    def reset(self):
        self.x_prev = copy(self.mu)

