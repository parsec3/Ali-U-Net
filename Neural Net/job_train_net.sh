##Job Settings

conda activate tf

##B1 nets:

#python Ali-U-Net-B1.py rows colums activation_function initialization_function training_file.tfr2 test_file.tfr2 Checkpoint-file-path file-name

python Ali-U-Net-B1.py 48 48 relu he_normal 48RSE_train.tfr2 48RSE_val.tfr2 48_RSE-relu-B1 Ali-U-Net_48x48_RSE-relu_B1_10M.h5
python Ali-U-Net-B1.py 48 48 sigmoid glorot_normal 48RSE_train.tfr2 48RSE_val.tfr2 48_RSE-sigmoid-B1 Ali-U-Net_48x48_RSE-sigmoid_B1_10M.h5
python Ali-U-Net-B1.py 96 96 relu he_normal 96RSE_train.tfr2 96RSE_val.tfr2 96_RSE-relu-B1 Ali-U-Net_96x96_RSE-relu_B1_10M.h5
python Ali-U-Net-B1.py 96 96 sigmoid glorot_normal 96RSE_train.tfr2 96RSE_val.tfr2 96_RSE-sigmoid-B1 Ali-U-Net_96x96_RSE-sigmoid_B1_10M.h5

python Ali-U-Net-B1.py 48 48 relu he_normal 48RSE+IG_train.tfr2 48RSE+IG_val.tfr2 48_RSE+IG-relu-B1 Ali-U-Net_48x48_RSE-IG-relu_B1_10M.h5
python Ali-U-Net-B1.py 48 48 sigmoid glorot_normal 48RSE+IG_train.tfr2 48RSE+IG_val.tfr2 48_RSE+IG-sigmoid-B1 Ali-U-Net_48x48_RSE-IG-sigmoid_B1_10M.h5
python Ali-U-Net-B1.py 96 96 relu he_normal 96RSE+IG_train.tfr2 96RSE+IG_val.tfr2 96_RSE+IG-relu-B1 Ali-U-Net_96x96_RSE-IG-relu_B1_10M.h5
python Ali-U-Net-B1.py 96 96 sigmoid glorot_normal 96RSE+IG_train.tfr2 96RSE+IG_val.tfr2 96_RSE+IG-sigmoid-B1 Ali-U-Net_96x96_RSE-IG-sigmoid_B1_10M.h5

python Ali-U-Net-B1.py 48 48 relu he_normal 48RSE-mixed_train.tfr2 48RSE-mixed_val.tfr2 48_RSE_mixed-relu-B1 Ali-U-Net_48x48_RSE_IG_mix-relu_B1_10M.h5
python Ali-U-Net-B1.py 48 48 sigmoid glorot_normal 48RSE-mixed_train.tfr2 48RSE-mixed_val.tfr2 48_RSE_mixed-sigmoid-B1 Ali-U-Net_48x48_RSE_IG_mix-sigmoid_B1_10M.h5
python Ali-U-Net-B1.py 96 96 relu he_normal 96RSE-mixed_train.tfr2 96RSE-mixed_val.tfr2 96_RSE_mixed-relu-B1 Ali-U-Net_96x96_RSE_IG_mix-relu_B1_10M.h5
python Ali-U-Net-B1.py 96 96 sigmoid glorot_normal 96RSE-mixed_train.tfr2 96RSE-mixed_val.tfr2 96_RSE_mixed-sigmoid-B1 Ali-U-Net_96x96_RSE_IG_mix-sigmoid_B2_10M.h5

##B2 nets:

#python Ali-U-Net-B2.py rows colums activation_function initialization_function training_file.tfr2 test_file.tfr2 Checkpoint-file-path file-name

python Ali-U-Net-B2.py 48 48 relu he_normal 48RSE_train.tfr2 48RSE_val.tfr2 48_RSE-relu-B2 Ali-U-Net_48x48_RSE-relu_B2_10M.h5
python Ali-U-Net-B2.py 48 48 sigmoid glorot_normal 48RSE_train.tfr2 48RSE_val.tfr2 48_RSE-sigmoid-B2 Ali-U-Net_48x48_RSE-sigmoid_B2_10M.h5
python Ali-U-Net-B2.py 96 96 relu he_normal 96RSE_train.tfr2 96RSE_val.tfr2 96_RSE-relu-B2 Ali-U-Net_96x96_RSE-relu_B2_10M.h5
python Ali-U-Net-B2.py 96 96 sigmoid glorot_normal 96RSE_train.tfr2 96RSE_val.tfr2 96_RSE-sigmoid-B2 Ali-U-Net_96x96_RSE-sigmoid_B2_10M.h5

python Ali-U-Net-B2.py 48 48 relu he_normal 48RSE+IG_train.tfr2 48RSE+IG_val.tfr2 48_RSE+IG-relu-B2 Ali-U-Net_48x48_RSE-IG-relu_B2_10M.h5
python Ali-U-Net-B2.py 48 48 sigmoid glorot_normal 48RSE+IG_train.tfr2 48RSE+IG_val.tfr2 48_RSE+IG-sigmoid-B2 Ali-U-Net_48x48_RSE-IG-sigmoid_B2_10M.h5
python Ali-U-Net-B2.py 96 96 relu he_normal 96RSE+IG_train.tfr2 96RSE+IG_val.tfr2 96_RSE+IG-relu-B2 Ali-U-Net_96x96_RSE-IG-relu_B2_10M.h5
python Ali-U-Net-B2.py 96 96 sigmoid glorot_normal 96RSE+IG_train.tfr2 96RSE+IG_val.tfr2 96_RSE+IG-sigmoid-B2 Ali-U-Net_96x96_RSE-IG-sigmoid_B2_10M.h5

python Ali-U-Net-B2.py 48 48 relu he_normal 48RSE-mixed_train.tfr2 48RSE-mixed_val.tfr2 48_RSE_mixed-relu-B2 Ali-U-Net_48x48_RSE_IG_mix-relu_B2_10M.h5
python Ali-U-Net-B2.py 48 48 sigmoid glorot_normal 48RSE-mixed_train.tfr2 48RSE-mixed_val.tfr2 48_RSE_mixed-sigmoid-B2 Ali-U-Net_48x48_RSE_IG_mix-sigmoid_B2_10M.h5
python Ali-U-Net-B2.py 96 96 relu he_normal 96RSE-mixed_train.tfr2 96RSE-mixed_val.tfr2 96_RSE_mixed-relu-B2 Ali-U-Net_96x96_RSE_IG_mix-relu_B2_10M.h5
python Ali-U-Net-B2.py 96 96 sigmoid glorot_normal 96RSE-mixed_train.tfr2 96RSE-mixed_val.tfr2 96_RSE_mixed-sigmoid-B2 Ali-U-Net_96x96_RSE_IG_mix-sigmoid_B2_10M.h5


##B3 nets:

#python Ali-U-Net-B3.py rows colums activation_function initialization_function training_file.tfr2 test_file.tfr2 Checkpoint-file-path file-name

python Ali-U-Net-B2.py 48 48 relu he_normal 48RSE_train.tfr2 48RSE_val.tfr2 48_RSE-relu-B3 Ali-U-Net_48x48_RSE-relu_B3_10M.h5
python Ali-U-Net-B2.py 48 48 sigmoid glorot_normal 48RSE_train.tfr2 48RSE_val.tfr2 48_RSE-sigmoid-B3 Ali-U-Net_48x48_RSE-sigmoid_B3_10M.h5
python Ali-U-Net-B2.py 96 96 relu he_normal 96RSE_train.tfr2 96RSE_val.tfr2 96_RSE-relu-B3 Ali-U-Net_96x96_RSE-relu_B3_10M.h5
python Ali-U-Net-B2.py 96 96 sigmoid glorot_normal 96RSE_train.tfr2 96RSE_val.tfr2 96_RSE-sigmoid-B3 Ali-U-Net_96x96_RSE-sigmoid_B3_10M.h5

python Ali-U-Net-B2.py 48 48 relu he_normal 48RSE+IG_train.tfr2 48RSE+IG_val.tfr2 48_RSE+IG-relu-B3 Ali-U-Net_48x48_RSE-IG-relu_B3_10M.h5
python Ali-U-Net-B2.py 48 48 sigmoid glorot_normal 48RSE+IG_train.tfr2 48RSE+IG_val.tfr2 48_RSE+IG-sigmoid-B3 Ali-U-Net_48x48_RSE-IG-sigmoid_B3_10M.h5
python Ali-U-Net-B2.py 96 96 relu he_normal 96RSE+IG_train.tfr2 96RSE+IG_val.tfr2 96_RSE+IG-relu-B3 Ali-U-Net_96x96_RSE-IG-relu_B3_10M.h5
python Ali-U-Net-B2.py 96 96 sigmoid glorot_normal 96RSE+IG_train.tfr2 96RSE+IG_val.tfr2 96_RSE+IG-sigmoid-B3 Ali-U-Net_96x96_RSE-IG-sigmoid_B3_10M.h5

python Ali-U-Net-B2.py 48 48 relu he_normal 48RSE-mixed_train.tfr2 48RSE-mixed_val.tfr2 48_RSE_mixed-relu-B3 Ali-U-Net_48x48_RSE_IG_mix-relu_B3_10M.h5
python Ali-U-Net-B2.py 48 48 sigmoid glorot_normal 48RSE-mixed_train.tfr2 48RSE-mixed_val.tfr2 48_RSE_mixed-sigmoid-B3 Ali-U-Net_48x48_RSE_IG_mix-sigmoid_B3_10M.h5
python Ali-U-Net-B2.py 96 96 relu he_normal 96RSE-mixed_train.tfr2 96RSE-mixed_val.tfr2 96_RSE_mixed-relu-B3 Ali-U-Net_96x96_RSE_IG_mix-relu_B3_10M.h5
python Ali-U-Net-B2.py 96 96 sigmoid glorot_normal 96RSE-mixed_train.tfr2 96RSE-mixed_val.tfr2 96_RSE_mixed-sigmoid-B3 Ali-U-Net_96x96_RSE_IG_mix-sigmoid_B3_10M.h5

