#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:41:18 2023

@author: Petar
"""

import random
import numpy as np

import tensorflow as tf
from tensorflow import keras



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

sequences = make_sequences(96,83)

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

shift_probabilities = [0.87,0.1,0.03]
shift = [0,2,1]
probs = [0.90,0.04,0.03,0.03]

def add_gaps(rows,columns):
  while True:
          sequence_array = make_sequences(rows,columns)
          gap_number = 3
          block_size = 10
          gap = random.randint(10,int(columns/2))
          gap2 = random.randint(int(columns/2),(columns-10))
          sequences = [] #This is all familiar
          shift_sequences = [] #We create two sequences tis time!
          for i in range(rows):
              sequence = list(sequence_array[i])
              shift_sequence = list(sequence_array[i]) #Again, we need everything twice, this time.
              if i < 3:
                  gap_site = int(gap)
                  for j in range(block_size):
                    extra = ''.join(random.choices(nucleotides, weights=(next(random_numbers_adding_up_100())), k=1)) #Adds random nucleotides
                    sequence.insert(gap_site,extra)
                    shift_sequence.insert(gap_site,extra) #Again, we need everything twice, this time.
                  gap_site2 = int(gap2) #For the block on the right
                  for k in range(gap_number):    #Adds three extra nucleotides
                    extra = ''.join(random.choices(nucleotides, weights=(next(random_numbers_adding_up_100())), k=1)) #Adds random nucleotides
                    sequence.insert(gap_site2,extra)
                    shift_sequence.insert(gap_site2,extra)
              else:
                  random_shift = random.choices(shift, weights=shift_probabilities, k=1)[0] #A random number (0, 1, or 2) shifts the block to the left
                  gap_site = int(gap) - random_shift  #Position of the gap (middle) - Shift number
                  for j in range(block_size):
                    sequence.insert(int(gap),'-') #Important! We need to leave it unchanged for the "normal" sequence
                    shift_sequence.insert(gap_site,'-')
                  gap_site2 = int(gap2) - random_shift #For the block on the right
                  for k in range(gap_number):
                    sequence.insert(int(gap2),'-')
                    shift_sequence.insert(gap_site2,'-')
              sequence = list(sequence)
              sequence = ''.join(sequence)
              shift_sequence = list(shift_sequence)
              shift_sequence = ''.join(shift_sequence)
              sequences += [sequence]
              shift_sequences += [shift_sequence]
          sequences = np.array(sequences)
          shift_sequences = np.array(shift_sequences)
          yield sequences, shift_sequences

gap_sequence = add_gaps(96,83)

def add_unit_gaps(sequence_array): ##Now for the unit vector.
        sequences = [] #Build the array new.
        for i in range(sequence_array.shape[0]): #Loops through all the rows of the array.
            sequence = list(sequence_array[i]) #Access the rows individually.
            seq = recode_seq(sequence)
            seq = tf.keras.utils.to_categorical(seq, num_classes=5, dtype='uint8') #The extra class is necessary; we have four nucleotides + a gap.
            sequences += [seq]
        sequences = np.array(sequences)
        return sequences


train_no_shift = np.empty((25000,96,96,5),dtype='uint8') #For the unshifted gaps.
train_shift = np.empty((25000,96,96,5),dtype='uint8') #For the shifted gaps.

for i in range(25000):
  gap_sequences, shift_gap_sequences = next(gap_sequence)
  unit_gap_seq = add_unit_gaps(gap_sequences)
  unit_shift_gap_seq = add_unit_gaps(shift_gap_sequences)
  train_no_shift[i] = unit_gap_seq
  train_shift[i] = unit_shift_gap_seq

#with open('shift_sequences.npy', 'wb') as f:
 #   np.save(f, train_no_shift)
  #  np.save(f, train_shift)

np.savez_compressed('compressed_shift_sequences.npz',x=train_shift, y=train_no_shift)

#np.save('unshifted_small.npy', train_no_shift, allow_pickle=True, fix_imports=True)
#np.save('shifted_small.npy', train_shift, allow_pickle=True, fix_imports=True)