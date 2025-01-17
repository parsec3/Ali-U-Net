#!/bin/bash     #Shebang
#
#$ -cwd         #Execute job from current directory
#$ -S /bin/bash #Interpeting shell for the job
#$ -j n         #Do not join stdout and stderr
#$ -N jobname #Name of the job
#$ -m n        #Send mail at beginning and end of job.
#$ -q fast.q
#$ -pe smp 5


conda activate tf


#python3 AliSim_RSE.py 48 48 5 3 alignments filename.tfr2
#python3 AliSim_RSE.py 96 96 7 5 alignments filename.tfr2

#python3 AliSim_RSE+IG.py 48 48 5 0 3 alignments filename.tfr2
#python3 AliSim_RSE+IG.py 96 96 7 1 5 alignments filename.tfr2

#python3 AliSim_RSE_mixed.py 48 48 5 0 3 alignments filename.tfr2
#python3 AliSim_RSE_mixed.py 96 96 7 1 5 alignments filename.tfr2
