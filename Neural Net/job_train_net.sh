##Job Settings

conda activate tf

##B1 nets:

#python Ali-U-Net-B1.py 48 48 relu he_normal 48RSE_train.tfr2 48RSE_val.tfr2 48_RSE-relu-B1
#python Ali-U-Net-B1.py 48 48 sigmoid glorot_normal 48RSE_train.tfr2 48RSE_val.tfr2 48_RSE-sigmoid-B1
#python Ali-U-Net-B1.py 96 96 relu he_normal 96RSE_train.tfr2 96RSE_val.tfr2 96_RSE-relu-B1
#python Ali-U-Net-B1.py 96 96 sigmoid glorot_normal 96RSE_train.tfr2 96RSE_val.tfr2 96_RSE-sigmoid-B1

#python Ali-U-Net-B1.py 48 48 relu he_normal 48RSE+IG_train.tfr2 48RSE+IG_val.tfr2 48_RSE+IG-relu-B1
#python Ali-U-Net-B1.py 48 48 sigmoid glorot_normal 48RSE+IG_train.tfr2 48RSE+IG_val.tfr2 48_RSE+IG-sigmoid-B1
#python Ali-U-Net-B1.py 96 96 relu he_normal 96RSE+IG_train.tfr2 96RSE+IG_val.tfr2 96_RSE+IG-relu-B1
#python Ali-U-Net-B1.py 96 96 sigmoid glorot_normal 96RSE+IG_train.tfr2 96RSE+IG_val.tfr2 96_RSE+IG-sigmoid-B1

#python Ali-U-Net-B1.py 48 48 relu he_normal 48RSE-mixed_train.tfr2 48RSE-mixed_val.tfr2 48_RSE_mixed-relu-B1
#python Ali-U-Net-B1.py 48 48 sigmoid glorot_normal 48RSE-mixed_train.tfr2 48RSE-mixed_val.tfr2 48_RSE_mixed-sigmoid-B1
#python Ali-U-Net-B1.py 96 96 relu he_normal 96RSE-mixed_train.tfr2 96RSE-mixed_val.tfr2 96_RSE_mixed-relu-B1
#python Ali-U-Net-B1.py 96 96 sigmoid glorot_normal 96RSE-mixed_train.tfr2 96RSE-mixed_val.tfr2 96_RSE_mixed-sigmoid-B1

##B2 nets:

#python Ali-U-Net-B2.py rows colums activation_function initialization_function training_file.tfr2 test_file.tfr2 Checkpoint-file-path

#[...]

##B3 nets:

#python Ali-U-Net-B2.py rows colums activation_function initialization_function training_file.tfr2 test_file.tfr2 Checkpoint-file-path

#[...]
