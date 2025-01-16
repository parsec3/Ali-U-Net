# Ali-U-Net
Aligning nucleotide sequences with a convolutional transformer neural network.

## The multiple sequence alignment problem
Determining the best multiple or pairwise sequence alignment lies at the basis of most sequence comparisons and therefore is a key problem in bioinformatics. The aim of aligning multiple homologous sequences of e.g. nucleotides or amino acids is to maximise the number of homologous residues that are found in the same alignment column of a multiple sequence alignment by adding gaps. See the [wikipedia article for more details about multiple sequence alignments](https://en.wikipedia.org/wiki/Multiple_sequence_alignment).

## The neural network
We present Ali-U-Net, a novel supervised machine learning strategy for the multiple sequence alignment problem using a slightly modified U-Net [Ronneberger et al. 2015](http://arxiv.org/abs/1505.04597) to transform unaligned sequences to a multiple sequence alignment. The U-Net is built using a series of convolutional layers that encode the image (encoder branch), followed by a series of upsampling/transpose convolutional layers (decoder branch) in which the resolution of the original image is reintroduced by skip connections from the decoder branch. This architecture will be used here to transform the matrix of unaligned nucleotides to a matrix of aligned nucleotides.

## Implementation
This repository contains the code of 
- the Ali-U-Net neural networks implemented using Python and Tensorflow
- Python scripts to create the training data sets
- Python scripts to train the Ali-U-Net
- Python scripts to predict alignments for unaligned sequences

## Installing prerequisite: Tensorflow Library
We recommend to install Tensorflow using the [miniforge package manager](https://github.com/conda-forge/miniforge).

## Authors:
Petar Arsic [(Leibniz Institute for the Analysis of Biodiversity Change, Bonn)](https://bonn.leibniz-lib.de/de/forschung)\
Christoph Mayer [(Leibniz Institute for the Analysis of Biodiversity Change, Bonn)](https://bonn.leibniz-lib.de/de/forschung)

## Reference: When using the Ali-U-Net, please cite:
xxx

