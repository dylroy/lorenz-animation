"""
Lorenz System Integration with RK4

This module provides functions to integrate through the Lorenz system of ODEs
using a fourth-order Runge-Kutta (RK4) integrator, iterate over different
parameter values, and visualize the results interactively.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def lorenz_derivatives(t, state, sigma, rho, beta):
    """
    Compute the derivatives of the Lorenz system.
    
    Parameters:
    -----------
    t : float
        Time (not used in Lorenz system, but required for ODE integration)
    state : array-like
        Current state vector [x, y, z]
    sigma : float
        Lorenz constant (Prandtl number)
    rho : float
        Lorenz constant (Rayleigh number)
    beta : float
        Lorenz constant (geometric factor)
    
    Returns:
    --------
    dstate_dt : np.ndarray
        Derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])


def rk4_step(func, t, y, dt, *args):
    """
    Perform a single RK4 integration step.
    
    Parameters:
    -----------
    func : callable
        Derivative function f(t, y, *args)
    t : float
        Current time
    y : np.ndarray
        Current state
    dt : float
        Time step
    *args : additional arguments
        Arguments to pass to func
    
    Returns:
    --------
    y_new : np.ndarray
        New state after one RK4 step
    """
    k1 = func(t, y, *args)
    k2 = func(t + dt/2, y + dt*k1/2, *args)
    k3 = func(t + dt/2, y + dt*k2/2, *args)
    k4 = func(t + dt, y + dt*k3, *args)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def integrate_lorenz(config):
    """
    Integrate the Lorenz system using RK4.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with keys:
        - 'sigma': float, Lorenz constant
        - 'rho': float, Lorenz constant  
        - 'beta': float, Lorenz constant
        - 'dt': float, time step
        - 'n_steps': int, number of time steps (length = n_steps * dt)
        - 'initial_state': array-like, starting position [x, y, z]
    
    Returns:
    --------
    trajectory : np.ndarray
        Array of shape (n_steps, 3) containing the position vectors
        at each time step
    times : np.ndarray
        Array of shape (n_steps,) containing the time values
    """
    # Extract configuration
    sigma = config['sigma']
    rho = config['rho']
    beta = config['beta']
    dt = config['dt']
    n_steps = config['n_steps']
    initial_state = np.array(config['initial_state'])
    
    # Initialize arrays
    trajectory = np.zeros((n_steps, 3))
    times = np.zeros(n_steps)
    
    # Set initial conditions
    trajectory[0] = initial_state
    times[0] = 0.0
    
    # Integrate
    current_state = initial_state.copy()
    for i in range(1, n_steps):
        current_state = rk4_step(
            lorenz_derivatives, 
            times[i-1], 
            current_state, 
            dt, 
            sigma, 
            rho, 
            beta
        )
        trajectory[i] = current_state
        times[i] = times[i-1] + dt
    
    return trajectory, times


def iterator_function(constant_values, constant_name, base_config):
    """
    Iteratively run the integrator for multiple values of one Lorenz constant.
    
    Parameters:
    -----------
    constant_values : iterable
        Values to iterate over for the specified constant
    constant_name : str
        Which constant to vary: "s" (sigma), "r" (rho), or "b" (beta)
    base_config : dict
        Base configuration dictionary. The specified constant will be
        overridden by values from constant_values.
    
    Returns:
    --------
    df : pd.DataFrame
        MultiIndex DataFrame with columns ['x', 'y', 'z', 'time'].
        Index levels: (constant_name, constant_value, time_step)
        Optimized for fast indexing and visualization.
    """
    # Validate constant_name
    constant_map = {'s': 'sigma', 'r': 'rho', 'b': 'beta'}
    if constant_name not in constant_map:
        raise ValueError(f"constant_name must be 's', 'r', or 'b', got '{constant_name}'")
    
    config_key = constant_map[constant_name]
    
    # Storage for all results
    all_data = []
    
    # Iterate over constant values
    for const_val in constant_values:
        # Create config for this iteration
        config = base_config.copy()
        config[config_key] = const_val
        
        # Integrate
        trajectory, times = integrate_lorenz(config)
        
        # Store results
        n_steps = trajectory.shape[0]
        for i in range(n_steps):
            all_data.append({
                'constant_name': constant_name,
                'constant_value': const_val,
                'time_step': i,
                'time': times[i],
                'x': trajectory[i, 0],
                'y': trajectory[i, 1],
                'z': trajectory[i, 2]
            })
    
    # Create DataFrame with MultiIndex for fast indexing
    df = pd.DataFrame(all_data)
    df = df.set_index(['constant_name', 'constant_value', 'time_step'])
    
    return df


def plot_interactive(df, constant_name):
    """
    Create an interactive 3D plot of Lorenz trajectories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame from iterator_function
    constant_name : str
        The constant name ('s', 'r', or 'b') to use for filtering
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive 3D plotly figure
    """
    # Get unique constant values
    constant_values = sorted(df.index.get_level_values('constant_value').unique())
    
    # Create figure
    fig = go.Figure()
    
    # Color palette
    try:
        import plotly.express as px
        colors = px.colors.qualitative.Set3
    except ImportError:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Add traces for each constant value
    for i, const_val in enumerate(constant_values):
        # Extract trajectory for this constant value
        trajectory_df = df.xs((constant_name, const_val), level=['constant_name', 'constant_value'])
        
        fig.add_trace(go.Scatter3d(
            x=trajectory_df['x'],
            y=trajectory_df['y'],
            z=trajectory_df['z'],
            mode='lines',
            name=f'{constant_name} = {const_val:.3f}',
            line=dict(color=colors[i % len(colors)], width=2),
            visible=True
        ))
    
    # Create dropdown buttons for interactivity
    buttons = []
    
    # Button to show all
    buttons.append({
        'label': 'Show All',
        'method': 'update',
        'args': [{'visible': [True] * len(constant_values)}]
    })
    
    # Buttons to show individual trajectories
    for i, val in enumerate(constant_values):
        visibility = [False] * len(constant_values)
        visibility[i] = True
        buttons.append({
            'label': f'{constant_name} = {val:.3f}',
            'method': 'update',
            'args': [{'visible': visibility}]
        })
    
    # Update layout
    fig.update_layout(
        title=f'Lorenz Attractor Trajectories (varying {constant_name})',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0)
            )
        ),
        width=900,
        height=700,
        updatemenus=[{
            'type': 'dropdown',
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'y': 1.15,
            'buttons': buttons
        }]
    )
    
    return fig


if __name__ == "__main__":
    # Example usage
    # Base configuration
    base_config = {
        'sigma': 10.0,
        'rho': 28.0,
        'beta': 8.0/3.0,
        'dt': 0.01,
        'n_steps': 5000,
        'initial_state': [1.0, 1.0, 1.0]
    }
    
    # Iterate over rho values
    rho_values = [20.0, 24.0, 28.0, 32.0]
    
    print("Running iterator function...")
    df = iterator_function(rho_values, 'r', base_config)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame index levels: {df.index.names}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Create interactive plot
    print("\nCreating interactive plot...")
    fig = plot_interactive(df, 'r')
    fig.show()