#!/bin/bash
#BSUB -J NACA_SEM_HPC
#BSUB -o NACA_SEM_HPC.out
#BSUB -e NACA_SEM_HPC.err

#BSUB -q hpc
#BSUB -R "rusage[mem=10GB]" # Multplied by number of cores (n=3 -> 30GB)
#BSUB -N
#BSUB -W 12:00 

#BSUB -n 3
#BSUB -R "span[hosts=1]"


module purge
source ~/venv-firedrake/bin/activate # Activate is modified to include gcc and openmpi paths

python3 -u HPC_Class_FS.py 