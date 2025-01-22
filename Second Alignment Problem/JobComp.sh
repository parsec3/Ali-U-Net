#!/bin/bash     #Shebang
#
#$ -cwd         #Execute job from current directory
#$ -S /bin/bash #Interpeting shell for the job
#$ -j n         #Do not join stdout and stderr
#$ -N make_fasta   #Name of the job
#$ -m n
#$ -q fast.q


conda activate tf


python3 NPZtoUnaligned.py
