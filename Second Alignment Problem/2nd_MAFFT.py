#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:07:28 2023

@author: Petar
"""

#Here, we will test how well the bioinformatics software MAFFT can align our training data:
#MAFFT has been installed according to the instructions on this website.
#https://mafft.cbrc.jp/alignment/software/ubuntu_on_windows.html
#Then, the file "unaligned.fasta" created in the previous file has been given as an input file.
#The output file was called "aligned.fasta" (see below)
#G-INS-i (accurate mode) was used. The default mode (FFT-NS-2) was very inaccurate and produced alignments that were too long.


import numpy as np
from Bio import SeqIO


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
    sequence = ''.join(sequence)  ##I'll keep this as an on/off-switch in case we need the raw letters.
    sequences += [sequence]
  sequences = np.array(sequences)
  return sequences

loaded = np.load('flatterrand_shift_sequences.npz')

y = loaded['y']

first_unit_shift_sequence = y[0]
first_shift_sequence = make_predict_sequences(first_unit_shift_sequence)

aligned_seq = []

handle = open("aligned.fasta")
for seq_record in SeqIO.parse(handle, "fasta"):
    seq = str(seq_record.seq)
    aligned_seq += [seq.upper()]
handle.close()

print("Here, we will see how MAFFT aligned the sequence vs how it should actually look. Differences are highlighted.")

accurate_hits = 0
inaccurate_hits = 0

for n, u in enumerate(aligned_seq): #This is all familiar from the predictions script.
  pred_seq = ''
  nucl1 = list(u)
  nucl2 = list(first_shift_sequence[n])
  for m, v in enumerate(nucl1):
    if v == nucl2[m]:
      pred_seq += v
      accurate_hits += 1 #To determine accuracy, we will count all the accurate and inaccurate hits.
    else:
      inaccurate_hits += 1
      if v == 'A':
        pred_seq += 'a'
      elif v == 'C':
        pred_seq += 'c'
      elif v == 'G':
        pred_seq += 'g'
      elif v == 'T':
        pred_seq += 't'
      elif v == '-':
        pred_seq += '~'
  print(pred_seq)

total_hits = accurate_hits + inaccurate_hits

print("MAFFT achieves a total accuracy of ",accurate_hits/total_hits,"%.")

#I got the result "MAFFT achieves a total accuracy of 98.741%."
#This is the accuracy that the Alig-Net has to defeat.
