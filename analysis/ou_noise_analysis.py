import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from common.loggerConfig import logger
from common.ornsteinUhlenbeck import OrnsteinUhlenbeck

def run_ou_noise_analysis(mu=0.0,
                          theta=0.05,
                          sigma=0.3,
                          dt=0.4,
                          x0=None,
                          min_sigma=0.01,
                          sigma_decay=0.9995,
                          steps=5000):

    ou = OrnsteinUhlenbeck(mu=mu, theta=theta, sigma=sigma, dt=dt, x0=x0, min_sigma=min_sigma, sigma_decay=sigma_decay)
    values = []

    # -------------------------
    #  Simulate OU noise
    # -------------------------
    for t in range(steps):
        val = ou.step()
        ou.decay_sigma()
        values.append(val)

    values = np.array(values)

    # -------------------------
    #  Compute stats
    # -------------------------
    v_min, v_max = values.min(), values.max()
    v_mean, v_std = values.mean(), values.std()

    logger.info("==== Ornstein-Uhlenbeck Noise Stats ====")
    logger.info(f"Min:   {v_min:.3f}")
    logger.info(f"Max:   {v_max:.3f}")
    logger.info(f"Mean:  {v_mean:.3f}")
    logger.info(f"Std:   {v_std:.3f}")
    logger.info(f"Final sigma: {ou.sigma:.3f}")

    # -------------------------
    #  Compute autocorrelation
    # -------------------------
    def autocorr(x):
        x = x - np.mean(x)
        result = correlate(x, x, mode='full')
        result = result[result.size // 2:]  # take positive lags
        result /= result[0]  # normalize
        return result

    lags = 200
    acorr = autocorr(values)[:lags]

    # -------------------------
    #  Plotting
    # -------------------------
    plt.figure(figsize=(16, 10))

    # (a) Time series
    plt.subplot(2, 2, 1)
    plt.plot(values, label="OU noise")
    plt.xlabel("Step")
    plt.ylabel("Noise value")
    plt.title("Ornstein-Uhlenbeck Noise Over Time")
    plt.grid(True)

    # (b) Histogram
    plt.subplot(2, 2, 2)
    plt.hist(values, bins=50, color='gray', edgecolor='black', density=True)
    plt.xlabel("Noise value")
    plt.ylabel("Density")
    plt.title("Noise Value Distribution")

    # (c) Autocorrelation
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(lags), acorr)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Noise Autocorrelation (Temporal Correlation)")
    plt.grid(True)

    # (d) Sigma decay over time
    plt.subplot(2, 2, 4)
    sigma_values = [0.3 * (0.9995**t) for t in range(steps)]
    sigma_values = np.maximum(sigma_values, ou.min_sigma)
    plt.plot(sigma_values)
    plt.xlabel("Step")
    plt.ylabel("Sigma")
    plt.title("Sigma Decay Over Time")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
