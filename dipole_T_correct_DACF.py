import numpy as np
n_atoms = 1728
n_steps = 50001
dt_fs = 0.5

def time_derivative(data, dt_fs):
    """
    Compute the time derivative using central finite differences.

    Parameters:
    - data : 1D or 2D numpy array (time series or per-atom time series)
    - dt_fs : timestep in femtoseconds

    Returns:
    - derivative : array of same shape as input, with end points handled by forward/backward difference
    """
    
    derivative = np.zeros_like(data)

    # Central difference
    derivative[..., 1:-1] = (data[..., 2:] - data[..., :-2]) / (2 * dt_fs)

    # Forward and backward difference at ends
    derivative[..., 0] = (data[..., 1] - data[..., 0]) / dt_fs
    derivative[..., -1] = (data[..., -1] - data[..., -2]) / dt_fs

    return derivative


def compute_acf(vx, vy, vz):
    """
    Compute the total vector autocorrelation function (ACF) from per-atom x/y/z components.

    Parameters:
    - vx, vy, vz: arrays of shape (n_atoms, n_timesteps)

    Returns:
    - total_acf: 1D array of shape (n_timesteps,)
    """

    # Ensure all arrays are the same shape
    assert vx.shape == vy.shape == vz.shape
    n_atoms, n_steps = vx.shape

    total_acf = np.zeros((n_atoms, n_steps))

    for t in range(n_steps):
        # Use full length if t == 0
        v_dot = (
            vx[:, :-t or None] * vx[:, t:] +
            vy[:, :-t or None] * vy[:, t:] +
            vz[:, :-t or None] * vz[:, t:]
        )  # shape: (n_atoms, n_steps - t)

        # Average over all time windows for each time t
        total_acf[:, t] = np.mean(v_dot, axis=1)

    # Average over atoms, final shape: (n_timesteps,)
    return np.mean(total_acf, axis=0)

# ACF

# Shape: (n_atoms, n_timesteps)
d_x = np.loadtxt("dipole_n_x.txt")
d_y = np.loadtxt("dipole_n_y.txt")
d_z = np.loadtxt("dipole_n_z.txt")

# Convert to arrays and transpose to (atoms, timesteps)
d_x = np.array(d_x).T
d_y = np.array(d_y).T
d_z = np.array(d_z).T

ddt_x = time_derivative(d_x, dt_fs)
ddt_y = time_derivative(d_y, dt_fs)
ddt_z = time_derivative(d_z, dt_fs)

# Shape: (n_timesteps,)
dacf_dt = compute_acf(ddt_x, ddt_y, ddt_z)
np.savetxt("dacf_n_dt300.txt", dacf_dt)