import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import gaussian_kde
from utils.functions import velocity_increments



def plot_time_series(sample):
    """
    Plot graph of a sample of 1D / 3D turbulent velocity trajectory

    Input:
        sample[numpy.array] : array containing turbulent velocity data (n_timesteps, n_dimensions)
    Output:
        fig
    """
    time = np.arange(sample.shape[0])
    fig = go.Figure()
    for i in range(sample.shape[1]):
        fig.add_traces(go.Scatter(x = time,
                                  y = sample[:,i],
                                  mode = 'lines',
                                  name = f'V{i+1}'))
    fig.update_layout(title=dict(text=f'A sample of {i+1}D turbulent velocity trajectory', font=dict(size=25)),
                      xaxis=dict(title=dict(text='Timestep', font=dict(size=20))),
                      yaxis=dict(title=dict(text='Velocity', font=dict(size=20)))
                      )
    fig.show()


def plot_pdf_increments(velocities, tau_values):
    """
    Plot the probability density function (PDF) graph of velocity increments for one component

    Inputs:
        velocities[numpy.array] : turbulent velocities with shape (n_samples, n_timesteps, n_dimensions)
        tau_values[List[Int64]] : list of time lags for computing the increments
    """
    fig = go.Figure()
    # Choose one component of velocities 
    if velocities.shape[2] == 1:
        dim = 0
    else:
        dim = np.random.randint(0,velocities.shape[2])
    # Loop over the tau values
    for tau in tau_values:
        # Calculate the velocity increments at tau
        increments = velocity_increments(velocities, tau)[:,dim]
        # Calculate pdf of velocity increments
        kde = gaussian_kde(increments)
        # Standardized the increments
        std_increments = increments / increments.std()
        # Calculate pdf for standardized velocity increments
        x_values = np.linspace(std_increments.min(), std_increments.max(), 1000)
        y_values = kde(x_values)
        # Add the pdf to the graph
        fig.add_traces(go.Scatter(x = x_values,
                       y = y_values,
                       mode ='lines',
                       name = f'τ = {tau}'))
    fig.update_layout(title=dict(text=f'Standardized PDFs of one generic component of the velocity increment for different τ', font=dict(size=25)),
                      xaxis=dict(title=dict(text='δτVi/σ(δτVi)', font=dict(size=20))),
                      yaxis=dict(title=dict(text='PDF(δτVi)', font=dict(size=20)))
                      )
    fig.show()

