import h5py
import os
import numpy as np


def file_loader(file_name):
    """
    Load the file .h5 to retrieve the turbulent velocity trajectories

    Input:
        file_name[String] : name of the .h5 dataset file
    Output:
        trajectory_data[numpy.array] : array containing turbulent velocity data (n_samples, n_timesteps, n_dimensions)

    """
    # Create the file path
    datasets_path = r'C:\Users\johnn\Turbulence-modeling---U-Net-neural-network\datasets'
    file_path = os.path.join(datasets_path, file_name)

    # Fetch the dataset information
    with h5py.File(file_path, 'r') as f:
        print(f"Keys of the file {file_name} : {list(f.keys())}")
        max_value = np.array(f.get('max'))
        min_value = np.array(f.get('min'))
        train = np.array(f.get('train'))
        print(f"    min value : {min_value}")
        print(f"    max value : {max_value}")
        print(f"    data shape : (n_samples, n_timesteps, n_dimensions) = {train.shape}")

    # Rescale training data to retrieve initial velocities before min-max method
    velocities = (train+1) * (max_value-min_value)/2 + min_value

    return velocities