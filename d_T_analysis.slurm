#!/bin/bash
#
#SBATCH -J analysis
#SBATCH -t 100:00:00
#SBATCH -N 2
#SBATCH -n 20
#SBATCH -o log%j.o
#SBATCH -e log%j.e
#SBATCH -p newq

conda activate neup
date
python dipole_T_correct_DACF.py
date


