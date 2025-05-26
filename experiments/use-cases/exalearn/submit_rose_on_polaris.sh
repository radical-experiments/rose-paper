#!/bin/bash
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -N rose
#PBS -o pbs.rose.out
#PBS -e pbs.rose.err
#PBS -q debug
#PBS -l walltime=1:00:00
#PBS -A RECUP
#PBS -l filesystems=home:grand:eagle

# move into the directory where qsub was invoked
cd $PBS_O_WORKDIR

# no environment variables are automatically exported (Slurm’s --export=NONE)
# so we simply activate our virtualenv
conda activate conda_rose

export RADICAL_PROFILE="TRUE"
export RADICAL_REPORT="TRUE"
export RADICAL_LOG_LVL="DEBUG"

python $USER/rose-paper/experiments/use-cases/exalearn/run_me.py 
