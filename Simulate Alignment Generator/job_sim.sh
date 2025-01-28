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


#python3 AliSim_RSE.py 48 48 5 3 10000000 48RSE_train.tfr2 #Training data
#python3 AliSim_RSE.py 48 48 5 3 1000 48RSE_val.tfr2 #Validation data

#python3 AliSim_RSE.py 96 96 7 5 10000000 96RSE_train.tfr2
#python3 AliSim_RSE.py 96 96 7 5 1000 96RSE_val.tfr2

#python3 AliSim_RSE+IG.py 48 48 5 0 3 10000000 48RSE-IG_train.tfr2
#python3 AliSim_RSE+IG.py 48 48 5 0 3 1000 48RSE-IG_val.tfr2

#python3 AliSim_RSE+IG.py 96 96 7 1 5 10000000 96RSE-IG_train.tfr2
#python3 AliSim_RSE+IG.py 96 96 7 1 5 1000 96RSE-IG_val.tfr2

#python3 AliSim_RSE_mixed.py 48 48 5 0 3 10000000 48RSE-mixed_train.tfr2
#python3 AliSim_RSE_mixed.py 48 48 5 0 3 1000 48RSE-mixed_val.tfr2

#python3 AliSim_RSE_mixed.py 96 96 7 1 5 10000000 96RSE-mixed_train.tfr2
#python3 AliSim_RSE_mixed.py 96 96 7 1 5 1000 96RSE-mixed_val.tfr2
