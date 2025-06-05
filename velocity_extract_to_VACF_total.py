# ---- velocity_extraction.py ----

import numpy as np

# Parameters
n_atoms = 1728
known_step = 50001
dump_file = "dump300.equilibrium.lammpstrj"
dt_fs = 0.5


n_vel = int(known_step) - 1

with open(dump_file, 'r') as file:

    def read_next_frame(file):
        line = file.readline()
        if not line:
            return None, None, None, None, None
        assert "ITEM: TIMESTEP" in line
        timestep = int(file.readline().strip())

        for _ in range(2): line = file.readline()
            
        dims = file.readline()
        assert "ITEM: BOX BOUNDS" in dims
        box_lengths = []
        lo, hi = map(float, file.readline().strip().split())
        box_lengths.append(hi - lo)
        L = max(box_lengths)

        for _ in range(2): line = file.readline()

        header = file.readline().strip()
        assert "ITEM: ATOMS" in header
        data = []
        for _ in range(n_atoms):
            parts = file.readline().split()
            atom_id = int(parts[0])
            atom_type = int(parts[1])  # <--- capture type
            x, y, z = map(float, parts[2:5])
            data.append((atom_id, atom_type, x, y, z))
        data.sort(key=lambda x: x[0])
        ids = [d[0] for d in data]
        types = [d[1] for d in data]
        positions = np.array([d[2:] for d in data])
        return timestep, positions, ids, types, L

    # First frame
    timestep_prev, pos_prev, atom_ids, atom_types, L = read_next_frame(file)
    timestep_first = timestep_prev
    timestep_last = timestep_prev
    
    # Preallocate full velocity array: (n_atoms, n_vel, 3)
    vel_array_xyz = np.empty((n_atoms, n_vel, 3), dtype=np.float32)
    t_index = 0 # Start with initial time = 0
    
    # Goes line by line through the file and extracts values for every time step
    while True:
        result = read_next_frame(file)
        if result[0] is None or t_index >= n_vel:
            break
        timestep_curr, pos_curr, _, _, _  = result
        timestep_last = timestep_curr

        displacement = pos_curr - pos_prev

        # Minimum image convention PBC correction using box length L
        displacement[displacement >  L/2] -= L
        displacement[displacement < -L/2] += L
    
        vel_array_xyz[:, t_index, :] = displacement / dt_fs
        pos_prev = pos_curr
        t_index += 1

# Final trimmed array
vel_array_xyz = vel_array_xyz[:, :t_index, :]  # (n_atoms, t_index, 3)

# Split by component
for comp_index, comp_label in zip(range(3), ["x", "y", "z"]):
    comp_data = vel_array_xyz[:, :, comp_index]
    np.savetxt(f"velocities_total_{comp_label}.txt", comp_data, fmt="%.6f")

# Time summary
elapsed_time_fs = (timestep_last - timestep_first) * dt_fs
print(f"Saved velocity components for all atoms into 3 files (x, y, z)")
print(f"Time range: timestep {timestep_first} to {timestep_last}")
print(f"Total time elapsed: {elapsed_time_fs:.2f} fs")


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
v_all_x = np.loadtxt("velocities_total_x.txt")
v_all_y = np.loadtxt("velocities_total_y.txt")
v_all_z = np.loadtxt("velocities_total_z.txt")

vacf_all = compute_acf(v_all_x, v_all_y, v_all_z)

# Shape: (n_timesteps,)
np.savetxt("vacf_all300.txt", vacf_all)