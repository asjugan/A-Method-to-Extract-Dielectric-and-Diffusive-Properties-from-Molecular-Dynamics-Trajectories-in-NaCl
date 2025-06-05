#!/bin/bash
#BATCH -J "NaCl_T"
#SBATCH -t 300:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p newq
#SBATCH -A ned

mpirun /home/asjugan/asjugan/lammps/build/lmp -in in.NaCl_dipole

