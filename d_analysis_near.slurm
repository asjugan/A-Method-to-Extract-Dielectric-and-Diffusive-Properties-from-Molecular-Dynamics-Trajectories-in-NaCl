#!/bin/bash
#
#SBATCH -J analysis
#SBATCH -t 24:00:00
#SBATCH -N 2
#SBATCH -n 20
#SBATCH -o log%j.o
#SBATCH -e log%j.e
#SBATCH -p newq

conda activate neup
date
python dipole_extract_nearest_to_DACF.py
date


