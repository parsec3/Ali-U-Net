#!/usr/bin/env python3


#This code converts our simulated data into FASTA files (which are the standard in bioinformatics).


import numpy as np


nucleotide = ["A", "C", "G", "T", "-"]

def make_predict_sequences(pred_array):
  sequences = []
  rows = pred_array.shape[0]
  columns = pred_array.shape[1]
  for i in range(rows):
    weight_profile = pred_array[i]
    sequence = []
    for j in range(columns):
     probs = list(weight_profile[j])
     position =  probs.index(max(probs))
     sequence += nucleotide[position]
    sequence = ''.join(sequence)
    sequences += [sequence]
  sequences = np.array(sequences)
  return sequences

#This is familiar from before: We import our training data and convert the first entry into a FASTA file.

loaded = np.load('path/to/simulated-npz-file.npz')


X = loaded['x']
Y = loaded['y']


first_sequence = []
first_shift_sequence = []

for i in range(1000):
    first_unit_shift_sequence = Y[i]
    first_shift_sequence = [make_predict_sequences(first_unit_shift_sequence)]

    first_unit_sequence = X[i]
    first_sequence += [make_predict_sequences(first_unit_sequence)]



list_seq = []

for s in first_sequence:
    list_s = []
    for t in s:
        list_s += [t]
    list_seq += [list_s]


seq_num = 0


for s in list_seq:
    list_name = []
    seq_num += 1
    ofile = open(r"unaligned_sequences_file_name_{}.fasta".format(seq_num), "w")
    for t in range(len(s)):
            number = t +1
            element = "Seq" + str(number)
            list_name += [element]
            ofile.write(">" + list_name[t] + "\n" +s[t] + "\n") #General format of a FASTA file.
    ofile.close()
