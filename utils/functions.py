import numpy as np


def std_velocity_increments(trajectories, tau):
    """
    Calculate all standardized velocity increments at a specified tau across all trajectory dimensions.
    Inputs:
        trajectories[numpy.array] : turbulent velocity trajectories with shape (n_samples, n_timesteps, n_dimensions)
        tau[Int64] : time lag for computing the increments
    Output:
        std_increments[numpy.array] : standardized velocity increments at tau with shape (n_samples * (n_timesteps - tau), n_dimensions)
    """
    n_dimensions = trajectories.shape[2]
    increments = (trajectories[:,tau:,:] - trajectories[:,:-tau,:]).reshape(-1,n_dimensions)
    std_increments = increments / increments.std(axis=0)
    print(f"For {std_increments.shape[1]} dimension(s), each dimension has {std_increments.shape[0]} samples of velocity increments at τ = {tau}.")
    return std_increments
