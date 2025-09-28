import numpy as np
import plotly.express as px
import plotly.graph_objects as go


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

