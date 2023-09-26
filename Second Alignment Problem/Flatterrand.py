#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:08:49 2023

@author: Petar
"""

import random
import numpy as np
import tensorflow as tf


def random_numbers_adding_up_100():
  while True:
      r1 = random.randint(10,100)
      r2 = random.randint(10,100)
      r3 = random.randint(10,100)
      r4 = random.randint(10,100)

      s = (r1+r2+r3+r4)

      r1 = r1/s
      r2 = r2/s
      r3 = r3/s
      r4 = r4/s

      yield (r1, r2, r3, r4)

gen = random_numbers_adding_up_100()

def DNA_profile(columns):
  probabilities = [0.90,0.04,0.03,0.03]
  profile=[]
  for i in range(columns):
    prob = random.sample(probabilities, len(probabilities)) #Shuffle around so that the spotlight is always on a different nucleotide.
    profile+=[prob]
  return profile

nucleotides = ["A", "C", "G", "T"]
probabilities = [0.90,0.04,0.03,0.03]


def make_sequences(rows,columns):
  profile = DNA_profile(columns)
  sequences = []
  for i in range(rows):
    sequence = []
    for i in range(columns):
      sequence += random.choices(nucleotides, weights=profile[i], k=1)
  #  sequence = ''.join(sequence)  ##I'll keep this as an on/off-switch in case we need the raw letters.
    sequences += [sequence]
  sequences = np.array(sequences)
  return sequences

sequences = make_sequences(96,96)

def recode_seq(seq): #This one is for recoding everything.
  seq_len = len(seq)
  a = np.empty(shape=(seq_len), dtype=np.uint8)
  for i, c in enumerate(seq):
    if (c == 'A'):
      a[i] = 0
    elif (c == 'C'):
      a[i] = 1
    elif (c == 'G'):
      a[i] = 2
    elif (c == 'T'):
      a[i] = 3
    elif (c == '-'):
      a[i] = 4
  return a

recoded_seq = []
for i in range(sequences.shape[0]):
  recoded_seq += [recode_seq(sequences[i])]

recoded_seq = np.array(recoded_seq)

growth_options = [-3,-2,-1,0,1,2,3]
growth_probabilities = [0.06,0.06,0.06,0.64,0.06,0.06,0.06]


def add_gaps(rows,columns):
  while True:
          sequence_array = make_sequences(rows,columns) #We generate a fully aligned matrix of 96 almost-identical sequences. It becomes a blueprint for both the aligned and unaligned seq.
          gap_size = 6 #Ground pattern for all the gaps
          gap_site = int(columns/4) #Gap position to the left of the middle
          gap_site2 = int(3*columns/4) #Gap position to the right of the middle.
          sequences = [] #This is all familiar
          shift_sequences = [] #We create two sequences this time!
          for i in range(rows):
              sequence = list(sequence_array[i])
              shift_sequence = list(sequence_array[i]) #Again, we need everything twice, this time.
              skip = random.choices(range(rows),k=5)
              if i in skip:
                sequence = ''.join(sequence)
                shift_sequence = ''.join(shift_sequence)
                sequences += [sequence]
                shift_sequences += [shift_sequence]
                continue
              right_margin = random.randint(0,gap_size) #We create a margin to the left and right where nucleotides are replaced with gaps. It's meant to give the sequences their "ragged" look
              left_margin = random.randint(0,gap_size) #The margin's size is always random, but it varies from between 0 to 6 in each run.
              #List 'a' stores values (e.g. [-1, 1, 2, 0]) that can grow or shrink a given gap sequence, depending on if they are positive or negative. They are added or subtracted from the indices below that decide which nucleotides are overwritten and which not.
              a = random.choices(growth_options, weights=growth_probabilities, k=4)
              total_margin = left_margin + right_margin + gap_size*2 + sum(a)
              sequence = ['-'] * left_margin + sequence[left_margin:] #This is the aligned sequence where the left margin is created.
              sequence = sequence[:columns-right_margin] + ['-'] * right_margin #Now, for the right margin of the aligned sequence.
              #What happens here is that the list is being sliced. One slice consists of all the nucleotides to the left of the gap site. Then, we get gaps equal to the gap size plus all the nucleotides to the right of the gap.
              sequence = sequence[:gap_site-a[0]] + ['-'] * (gap_size+a[0]+a[1]) + sequence[gap_site+a[1]+gap_size:]
              sequence = sequence[:gap_site2-a[2]] + ['-'] * (gap_size+a[2]+a[3]) + sequence[gap_site2+a[3]+gap_size:]
              del shift_sequence[0:left_margin] #The unaligned sequence has no left margin. Here, all nucleotides start in the same place. Where the aligned sequence has gaps, it has nothing.
              del shift_sequence[(gap_site-a[0]-left_margin):(gap_site+a[1]+gap_size-left_margin)] #Whenever we delete nucleotides, we move the index to the left, so, when we need to delete more, we need to move the index according to the preceding rows. Only that way, we make sure we are deleting the right nucleotides so that both alignments have the same letters.
              del shift_sequence[(gap_site2-a[2]-a[0]-a[1]-left_margin-gap_size):(gap_site2+a[3]-a[0]-a[1]-left_margin)]
              shift_sequence = shift_sequence[:columns-total_margin] + ['-'] * total_margin #The unaligned sequence has a right margin to make sure it's as wide as the aligned one.
              sequence = ''.join(sequence)
              shift_sequence = ''.join(shift_sequence)
              sequences += [sequence]
              shift_sequences += [shift_sequence]
          sequences = np.array(sequences)
          shift_sequences = np.array(shift_sequences)
          yield sequences, shift_sequences

gap_sequence = add_gaps(96,96)

def add_unit_gaps(sequence_array): ##Now for the unit vector.
        sequences = [] #Build the array new.
        for i in range(sequence_array.shape[0]): #Loops through all the rows of the array.
            sequence = list(sequence_array[i]) #Access the rows individually.
            seq = recode_seq(sequence)
            seq = tf.keras.utils.to_categorical(seq, num_classes=5, dtype='uint8') #The extra class is necessary; we have four nucleotides + a gap.
            sequences += [seq]
        sequences = np.array(sequences)
        return sequences

gap_sequences, shift_gap_sequences = next(gap_sequence)

train_no_shift = np.empty((50000,96,96,5),dtype='uint8') #For the unshifted gaps.
train_shift = np.empty((50000,96,96,5),dtype='uint8') #For the shifted gaps.

for i in range(50000):
  gap_sequences, shift_gap_sequences = next(gap_sequence)
  unit_gap_seq = add_unit_gaps(gap_sequences)
  unit_shift_gap_seq = add_unit_gaps(shift_gap_sequences)
  train_no_shift[i] = unit_gap_seq
  train_shift[i] = unit_shift_gap_seq

np.savez_compressed('flatterrand_shift_sequences.npz',x=train_shift, y=train_no_shift)
