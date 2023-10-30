#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:01:09 2023

@author: Petar
"""

#Here, we will test how well the bioinformatics software T-Coffee can align our training data:
#T-Coffee has been installed in ubuntu similar to how we downloaded MAFFT.
#Then, the file "unaligned.fasta" used in all the others has been given as an input file.
#The output file was called "aligned.aln"

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

handle = open("unaligned.aln")
for seq_record in SeqIO.parse(handle, "clustal"):
    seq = str(seq_record.seq)
    aligned_seq += [seq.upper()]
handle.close()

print("Here, we will see how T-Coffee aligned the sequence vs how it should actually look. Differences are highlighted.")

accurate_hits = 0
inaccurate_hits = 0

for n, u in enumerate(aligned_seq): #This is all familiar from the predictions script.
  pred_seq = ''
  nucl1 = list(u)[0:95] #As with ClustalW, I fear too long sequences. 
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

print("T-Coffee achieves a total accuracy of ",accurate_hits/total_hits,"%.")

#I got the result "T-Coffee achieves a total accuracy of 94.418%."