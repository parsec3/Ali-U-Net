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

## Authors:
Petar Arsic [(Leibniz Institute for the Analysis of Biodiversity Change, Bonn)](https://bonn.leibniz-lib.de/de/forschung)\
Christoph Mayer [(Leibniz Institute for the Analysis of Biodiversity Change, Bonn)](https://bonn.leibniz-lib.de/de/forschung)

## Reference: When using the Ali-U-Net, please cite:
REF

## Trained neural networks can be downloaded from:
REF

