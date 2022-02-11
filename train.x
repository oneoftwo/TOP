#!/bin/bash
#PBS -N LJW_
#PBS -l nodes=gnode2:ppn=4
#PBS -l walltime=1000:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

source activate shwan
python -u parse.py

