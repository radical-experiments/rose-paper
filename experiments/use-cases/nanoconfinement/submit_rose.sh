#!/bin/bash
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -N rose
#PBS -o pbs.rose.out
#PBS -e pbs.rose.err
#PBS -q preemptable
#PBS -l walltime=10:00:00
#PBS -A RECUP
#PBS -l filesystems=home:grand:eagle

# move into the directory where qsub was invoked
cd $PBS_O_WORKDIR

# no environment variables are automatically exported (Slurm’s --export=NONE)
# so we simply activate our virtualenv
source /home/twang3/useful_script/conda_rose.sh

export RADICAL_PROFILE="TRUE"
export RADICAL_REPORT="TRUE"
export RADICAL_LOG_LVL="DEBUG"

python /eagle/RECUP/twang/rose/nanoconfinement/nanoconfinement-md/python/surrogate_samplesize/run_rose.py
