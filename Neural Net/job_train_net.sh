#!/bin/bash     #Shebang
#
#$ -cwd         #Execute job from current directory
#$ -S /bin/bash #Interpeting shell for the job
#$ -j n         #Do not join stdout and stderr
#$ -N train_net #Name of the job
#$ -m n        #Do no send mail at beginning and end of job.
#$ -q fast.q
#$ -pe smp 5

conda activate tf


#python Ali-U-Net-1B.py rows colums activation_function initialization_function training_file.tfr2 test_file.tfr2 Checkpoint-file-path
#python Ali-U-Net-2B.py rows colums activation_function initialization_function training_file.tfr2 test_file.tfr2 Checkpoint-file-path
#python Ali-U-Net-3B.py rows colums activation_function initialization_function training_file.tfr2 test_file.tfr2 Checkpoint-file-path
