# A-Method-to-Extract-Dielectric-and-Diffusive-Properties-from-Molecular-Dynamics-Trajectories-in-NaCl

Hello!

Here is where I will guide you, the user, through the different functions and files that correspond to the different steps and outputs through my process. This will conclude with the results presented in my thesis. 

This folder has a few components:

* Base input and run files for a simulation
* JUPYTER codes to manipulate a standard dump file
* JUPYTER codes that use the preceeding outputs to populate the relevant figures


Step 1:

Find file that starts with "in.", this is the input file that may be run in the RDFMG cluster (for NCSU users). Use the corresponding "run_-.sh" file to initialize the simulation.

This input file may be modified for each temperature. 


Step 2:

The resulting "dumpfile" will have data for every atom at every selected timestep in the form of <id type x y z>.


Step 3:

Process each dump file with the following JUPYTER codes. The outputs of those files will later be used to generate figures.

"velocity_extract_to_VACF.ipynb"
"velocity_extract_to_VACF_total.ipynb"
"dipole_extract_nearest_to_DACF.ipynb"

NOTE: must convert the file tag from ".ipynb" to ".py" to run the functions on the RDFMG cluster.

These functions do as they are titled. They extract the data from the dumpfile and then output files for the velocities, VACFs, dipoles, and DACFs.

The "...slurm" files are used to initialize and run the ".py" files. 

The associated output files are:

"vacf_all-.txt"
"vacf_Na-.txt"
"vacf_Cl-.txt"
"dipole_n_dt-.txt"

WARNING: See below for notes on modifications implemented to the data processing. 

ex) "dipole_T_correct_DACF" is a function specifically to modify the transposing of dipole files.

WARNING: The above file should be modified for ease of use.


Step 4:

The previous vacf and dacf output files now become inputs for plotting functions that were performed off the cluster within JUPYTER notebook. 

The following plotting functions are:

"RDF_plot.ipynb"
"many_VACF_VDOS_plot.ipynb"
"many_DACF_dielectric_refractive_plot.ipynb"
"Previous_Attempt_Kramers_Kronig_Calculation.ipynb"

Simply run these files and ensure the input file locations are correct.

Plotting and analysis may now commence.

Congratulations, you've made it through the process to obtain the results presented in my thesis.

Thanks,
Alina Jugan
asjugan@gmail.com


NOTES:

This set of simulations ran with a timestep of 0.5 fs and 50000 timesteps. Equivalent 25 picoseconds. 
This only produced the dump file of id type x y z.

GOAL: 

Apply functions for velocity/dipole moment & VACF/DACF extraction in the RDFMG cluster.
Use JUPYTER to plot the VACF/DACF & then VDOS/dielectric function/refractive index.

OUTCOMES:

I compiled RDFs straight from LAMMPS functions and plotted them with respect to time. It appears that 1500K is the only melted NaCl simulation.
The dipole moments have some noise at the end so I cropped them before calculating the Fourier transform: acf[:int(0.9 * len(acf))]
I attempted to perform a double integral Kramer's Kronig calculation. Since then I have ended up utilizing a predefined python function that is cited in the JUPYTER run file. 
Since implementing the python function for the Kramer's Kronig integral, the results for real dielectric function are less smooth and match up with the imaginary dielectric function more clearly.
Diffusion results for non-liquid fluctuate around zero and occassionally go negative, this is likely a noise issue.
Magnitudes for dielectric function and refractive index for the liquid state are slightly larger than expected from literature. 

        WARNING: the functions for dipole extraction that I ran on the cluster and that are currently saved in JUPYTER have a minor error 
        (to be corrected for further runs). In all dipole_extract files, the following lines are incorrect:

                    # Convert to arrays and transpose to (atoms, timesteps)
                    dipole_x = np.array(dipole_x).T
                    dipole_y = np.array(dipole_y).T
                    dipole_z = np.array(dipole_z).T

            - They are already in the correct shape and no transposing is necessary. 
            - I had to make another function to re-transpose the data and recalculate the time derivative and DACF

        WARNING: the functions for dipole extraction that I ran on the cluster and that are currently saved in jupyter have another minor error 
        (to be corrected for further runs). In all dipole_extract files, the following line is incorrect:
        
                    mu[i] /= (len(neighbors) - 1) # I want the average dipole moment contribution of all neighbors, for x y z, for each atom, at each time step
        
        Instead, I should not be dividing the dipole moment by the total number of atoms. I need the TOTAL dipole moment at each atom, not the average. 
            - If I know the number of atoms (2)(3): 
                - I can correct this post processing and simply multiply each dipole moment contribution by n_atoms.
                - Or I can multiply the DACF output by (n_atoms)^2
            - For the _cutoff (1) version of the dipole_extraction, since n_atoms can fluctuate, I cannot perform this correction. 
            - For fixed_nearest (2) where only one atom is used, no correction is needed.
