#!/bin/bash
#BSUB -J NACA_SEM_HPC
#BSUB -o NACA_SEM_HPC.out
#BSUB -e NACA_SEM_HPC.err

#BSUB -q hpc
#BSUB -R "rusage[mem=4GB]"
#BSUB -N
#BSUB -W 12:00 

#BSUB -n 3
#BSUB -R "span[hosts=1]"


module purge
source ~/venv-firedrake/bin/activate

python3 -u HPC_Class_FS.py