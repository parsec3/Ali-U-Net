#!/bin/bash     #Shebang
#
#$ -cwd         #Execute job from current directory
#$ -S /bin/bash #Interpeting shell for the job
#$ -j n         #Do not join stdout and stderr
#$ -N jobname   #Name of the job
#$ -m n
#$ -q fast.q
#$ -pe smp 5

conda activate tf

python3 Ali-U-Net-Score.py
