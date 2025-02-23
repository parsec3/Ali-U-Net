# Ali-U-Net
Aligning nucleotide sequences with a convolutional transformer neural network.

## The multiple sequence alignment problem
Determining the best multiple or pairwise sequence alignment lies at the basis of most sequence comparisons and therefore is a key problem in bioinformatics. The aim of aligning multiple homologous sequences of e.g. nucleotides or amino acids is to maximise the number of homologous residues that are found in the same alignment column of a multiple sequence alignment by adding gaps. See the [wikipedia article for more details about multiple sequence alignments](https://en.wikipedia.org/wiki/Multiple_sequence_alignment).

## The neural network
We present Ali-U-Net, a novel neural network architecture, inspired by the U-Net [Ronneberger et al. 2015](http://arxiv.org/abs/1505.04597), that is capable of transforming unaligned nucleotide sequences using a slightly modified U-Net architecture. The U-Net is built using a series of convolutional layers that encode the alignment (encoder branch), followed by a series of upsampling/transpose convolutional layers (decoder branch) in which the resolution of the original alignment is reintroduced by skip connections from the decoder branch. 

## Implementation
This repository contains the code of 
- the Ali-U-Net neural networks implemented using Python and Tensorflow
- Python scripts to create the training data sets
- Python scripts to train the Ali-U-Net
- Python scripts to predict alignments for unaligned sequences

## Installing Tensorflow Library
The only library that needs to be installed for this project is the Tensorflow Library.
We recommend to install Tensorflow using the [miniforge package manager](https://github.com/conda-forge/miniforge)
and the following commands:
```
conda create --name name_of_the_environment tensorflow
```

## Creating training, validation, and test datasets for the different study cases
Creating train, validation, and test datasets for the RSE (Ragged sequence ends) study case.
The Python scripts for the simulations can be found in the "Simulate Alignment Generator" folder. Each script takes the following arguments in this order:
columns rows margin_size skipped_rows number_of_alignments file_name
```
python3 AliSim_RSE.py 48 48 5 3 10000000 48RSE_train.tfr2 #Training data
python3 AliSim_RSE.py 48 48 5 3 1000     48RSE_val.tfr2   #Validation data
python3 AliSim_RSE.py 48 48 5 3 1000     48RSE_test.tfr2  #Test data

python3 AliSim_RSE.py 96 96 7 5 10000000 96RSE_train.tfr2
python3 AliSim_RSE.py 96 96 7 5 1000     96RSE_val.tfr2
python3 AliSim_RSE.py 96 96 7 5 1000     96RSE_test.tfr2

python3 AliSim_RSE+IG.py 48 48 5 0 3 10000000 48RSE-IG_train.tfr2
python3 AliSim_RSE+IG.py 48 48 5 0 3 1000     48RSE-IG_val.tfr2
python3 AliSim_RSE+IG.py 48 48 5 0 3 1000     48RSE-IG_test.tfr2

python3 AliSim_RSE+IG.py 96 96 7 1 5 10000000 96RSE-IG_train.tfr2
python3 AliSim_RSE+IG.py 96 96 7 1 5 1000     96RSE-IG_val.tfr2
python3 AliSim_RSE+IG.py 96 96 7 1 5 1000     96RSE-IG_test.tfr2

python3 AliSim_RSE_mixed.py 48 48 5 0 3 10000000 48RSE-mixed_train.tfr2
python3 AliSim_RSE_mixed.py 48 48 5 0 3 1000     48RSE-mixed_val.tfr2
python3 AliSim_RSE_mixed.py 48 48 5 0 3 1000     48RSE-mixed_test.tfr2

python3 AliSim_RSE_mixed.py 96 96 7 1 5 10000000 96RSE-mixed_train.tfr2
python3 AliSim_RSE_mixed.py 96 96 7 1 5 1000     96RSE-mixed_val.tfr2
python3 AliSim_RSE_mixed.py 96 96 7 1 5 1000     96RSE-mixed_val.tfr2
```

## Training the neural network
The neural network-scripts can be found in the "Neural Net" folder and they take the following arguments:
rows columns activation_function initializer training_file validation_file checkpoint_file_path file_name
Here is how this looks for the 48x48 RSE study case (see the "job_train_net.sh" file in the "Neural Net" folder for the other examples:
```
python Ali-U-Net-B1.py 48 48 relu he_normal 48RSE_train.tfr2 48RSE_val.tfr2 48_RSE-relu-B1 Ali-U-Net_48x48_RSE-relu_B1_10M.h5
python Ali-U-Net-B1.py 48 48 sigmoid glorot_normal 48RSE_train.tfr2 48RSE_val.tfr2 48_RSE-sigmoid-B1 Ali-U-Net_48x48_RSE-sigmoid_B1_10M.h5

python Ali-U-Net-B2.py 48 48 relu he_normal 48RSE_train.tfr2 48RSE_val.tfr2 48_RSE-relu-B2 Ali-U-Net_48x48_RSE-relu_B2_10M.h5
python Ali-U-Net-B2.py 48 48 sigmoid glorot_normal 48RSE_train.tfr2 48RSE_val.tfr2 48_RSE-sigmoid-B2 Ali-U-Net_48x48_RSE-sigmoid_B2_10M.h5

python Ali-U-Net-B3.py 48 48 relu he_normal 48RSE_train.tfr2 48RSE_val.tfr2 48_RSE-relu-B3 Ali-U-Net_48x48_RSE-relu_B3_10M.h5
python Ali-U-Net-B3.py 48 48 sigmoid glorot_normal 48RSE_train.tfr2 48RSE_val.tfr2 48_RSE-sigmoid-B3 Ali-U-Net_48x48_RSE-sigmoid_B3_10M.h5
```

## Using trained neural networks for predicting alignments
Trained neural networks can be downloaded from the Zenodo repository xxx.
The script can be found in the "Testing the Models" folder and it takes the following arguments:
rows columns filepath_to_the_model
Here an example for loading the Ali-U-Net_48x48_RSE-IG_B3_10M.h5 model:
```
python3 Ali-U-Net-Alignment.py 48 48 Ali-U-Net_48x48_RSE-IG_B3_10M.h5
```

## Authors:
Petar Arsic [(Leibniz Institute for the Analysis of Biodiversity Change, Bonn)](https://bonn.leibniz-lib.de/de/forschung)\
Christoph Mayer [(Leibniz Institute for the Analysis of Biodiversity Change, Bonn)](https://bonn.leibniz-lib.de/de/forschung)

## Reference: When using the Ali-U-Net, please cite:
REF


