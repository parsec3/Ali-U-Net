##Job Settings

#Important: The input and output files have a name format of filename + number + ".fas", e.g. 48_RSE_val_y_3.fas
#When giving the arguments to the model, it is important to only give the base name, as the script will add the number and the
#file name extension (e.g. 48_RSE_val_y rather than 48_RSE_val_y_3.fas).
#The same rule applies to "Ali-U-Net-Hallucination-Score.sh and Ali-U-Net-Score.sh"

conda activate tf

#python3 Ali-U-Net-Alignment.py rows columns filepath/to/model filepath/to/input_file output_file
