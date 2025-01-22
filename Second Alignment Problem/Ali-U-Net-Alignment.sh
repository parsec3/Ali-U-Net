#!/bin/bash     #Shebang
#
#$ -cwd         #Execute job from current directory
#$ -S /bin/bash #Interpeting shell for the job
#$ -j n         #Do not join stdout and stderr
#$ -N AliU_gap96_1k
#$ -m n        #Do no send mail at beginning and end of job.
#$ -q fast.q
#$ -pe smp 5

conda activate tf

### An alignment file exists for every architecture, as the architecture must match that of the .hdf5

#python3 Ali-U-Net-Alignment-1B.py rows colums activation_function initialization_function
#python3 Ali-U-Net-Alignment-2B.py rows colums activation_function initialization_function
#python3 Ali-U-Net-Alignment-3B.py rows colums activation_function initialization_function
