import math
import numpy as np


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of (1-beta) over time from t = [0,1].

    Inputs :
        num_diffusion_timesteps[Int64]: the number of betas to produce.
        alpha_bar[function]: a lambda that takes an argument t from 0 to 1 and produces the cumulative product of (1-beta) up to that part of the diffusion process.
        max_beta[float]: the maximum beta to use; use values lower than 1 to prevent singularities.
    Output :
        betas[np.array] : sequence of betas
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name. Beta schedules may be added, but should not be removed or changed once they are committed to maintain backwards compatibility.
    
    Inputs :
        schedule_name[String] : the name referring to the schedule function
        num_diffusion_timesteps[Int64]: the number of betas to produce.
    Output :
        betas[np.array] : the sequence of betas schedule for the given name
    """
    # Linear schedule from Ho et al, extended to work for any number of diffusion steps.
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    
    elif schedule_name.startswith("power"):
        power = int(schedule_name[5:])
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1 - t**power,
        )
    
    elif schedule_name.startswith("exp"):
        t0 = float(schedule_name[3:])
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 2 - math.exp((t0 + math.log(2)) * t - t0),
        )
    
    elif schedule_name.startswith("tanh"):
        t0, t1 = schedule_name.split(",")
        t0, t1 = float(t0[4:]), float(t1)
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: -math.tanh((t0 + t1) * t - t0) + math.tanh(t1),
        )
    
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    

def alpha_bar_from_betas(betas):
    """
    Calculate the alpha bar values from the sequence of betas

    Input:
        betas[numpy.array] : sequence of betas
    Output:
        alpha_bar[numpy.array] : sequence of alpha bar values
    """
    return np.cumprod(1-betas)