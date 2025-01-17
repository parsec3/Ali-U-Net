#!/bin/bash     #Shebang
#
#$ -cwd         #Execute job from current directory
#$ -S /bin/bash #Interpeting shell for the job
#$ -j n         #Do not join stdout and stderr
#$ -N tfr_to_fasta #Name of the job
#$ -m n        #Send mail at beginning and end of job.
#$ -q fast.q


conda activate tf


python3 tfr2fasta.py filename rows columns N
