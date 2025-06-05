# ---- dipole_extraction_nearest.py ----

import numpy as np
from periodic_kdtree import PeriodicCKDTree
n_atoms = 1728
n_steps = 50001
n_neighbors = 1
dump_file = "dump300.equilibrium.lammpstrj"
dt_fs = 0.5


with open(dump_file, 'r') as file:

    def read_next_frame(file):
        line = file.readline()
        if not line:
            return None, None, None, None
        assert "ITEM: TIMESTEP" in line
        timestep = int(file.readline().strip())
    
        for _ in range(2): line = file.readline()
    
        dims = file.readline()
        assert "ITEM: BOX BOUNDS" in dims
        box_lengths = []
        for _ in range(3):
            lo, hi = map(float, file.readline().strip().split())
            box_lengths.append(hi - lo)
        box = np.array(box_lengths)  # box size in x, y, z
        L = max(box_lengths)
        
        header = file.readline()
        assert "ITEM: ATOMS" in header
        data = []
        for _ in range(n_atoms):
            parts = file.readline().split()
            atom_id = int(parts[0])
            atom_type = int(parts[1]) # <--- capture type... for charge identification
            x, y, z = map(float, parts[2:5])
            charge = 1 if atom_type == 1 else -1
            data.append((atom_id, charge, x, y, z))
        data.sort(key=lambda x: x[0])
        ids = [d[0] for d in data]
        charges = np.array([d[1] for d in data])
        positions = np.array([d[2:] for d in data])
        return timestep, positions, ids, charges, box, L
    
    # First frame
    timestep_first = None
    timestep_last = None

    # Preallocate arrays
    dipole_x = np.zeros((n_atoms, n_steps), dtype=float)
    dipole_y = np.zeros((n_atoms, n_steps), dtype=float)
    dipole_z = np.zeros((n_atoms, n_steps), dtype=float)
    
    t_index = 0

    # Goes line by line through the file and extracts values for every time step
    while True:
        result = read_next_frame(file)
        if result[0] is None:
            break
        timestep, positions, _, charges, box, L = result

        if timestep_first is None:
            timestep_first = timestep
        timestep_last = timestep

        mu = np.zeros((n_atoms, 3))

        tree = PeriodicCKDTree(box, positions)
        
        for i, ri in enumerate(positions):
            distances, indices = tree.query(ri, k=n_neighbors + 1)  # includes self

            mu_sum = np.zeros(3)
            count = 0
        
            for j in indices:
                if i == j:
                    continue  # skip self
                r_vec = positions[j] - ri

                # Apply Minimum Image Convention
                r_vec[r_vec >  L / 2] -= L
                r_vec[r_vec < -L / 2] += L
                
                mu_sum += charges[j] * r_vec
                count += 1
        
            if count > 0:
                mu[i] = mu_sum / count # I want the average dipole moment contribution of all neighbors, for x y z, for each atom, at each time step

        dipole_x[:, t_index] = mu[:, 0]
        dipole_y[:, t_index] = mu[:, 1]
        dipole_z[:, t_index] = mu[:, 2]
        
        t_index += 1

# Convert to arrays and transpose to (atoms, timesteps)
dipole_x = np.array(dipole_x).T
dipole_y = np.array(dipole_y).T
dipole_z = np.array(dipole_z).T

# Save to files
np.savetxt("dipole_n_x.txt", dipole_x, fmt="%.6f")
np.savetxt("dipole_n_y.txt", dipole_y, fmt="%.6f")
np.savetxt("dipole_n_z.txt", dipole_z, fmt="%.6f")

elapsed_time_fs = (timestep_last - timestep_first) * dt_fs
print(f"Saved dipole components into 3 files (x, y, z)")
print(f"Time range: timestep {timestep_first} to {timestep_last}")
print(f"Total time elapsed: {elapsed_time_fs:.2f} fs")


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

ddt_x = time_derivative(d_x, dt_fs)
ddt_y = time_derivative(d_y, dt_fs)
ddt_z = time_derivative(d_z, dt_fs)

# Shape: (n_timesteps,)
dacf_dt = compute_acf(ddt_x, ddt_y, ddt_z)
np.savetxt("dacf_n_dt300.txt", dacf_dt)