##Job settings

#See "Ali-U-Net-Alignment.sh" for how to name the aligned_file."

conda activate tf

python3 Ali-U-Net-Score.py rows columns filepath/to/tfrecord_file filepath/to/aligned_file
