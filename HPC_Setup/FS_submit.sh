#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J FS_wide3
### -- ask for number of cores (default: 1) --
#BSUB -n 8 
### -- specify that the cores MUST BE on a single host --
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=10GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o FS_wide3_%J.out
#BSUB -e FS_wide3_%J.err
# -- end of LSF options --
#here load the modules, and activate the environment if needed


export CC="~/petsc/arch-firedrake-default/bin/mpicc"
export MPICC="~/petsc/arch-firedrake-default/bin/mpicc"
export CXX="~/petsc/arch-firedrake-default/bin/mpicxx"
export FC="~/petsc/arch-firedrake-default/bin/mpifort"
module load gcc/12.3.0-binutils-2.40
module load mpi/5.0.3-gcc-12.3.0-binutils-2.40
source ~/firedrake-venv/bin/activate
python --version

# here call torchrun
python3 -u HPC.py
