# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 08:19:38 2023

@author: Petar
"""

import tensorflow as tf
import numpy as np
import keras
#from sklearn.model_selection import train_test_split

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
model = keras.models.load_model('my_model.keras')



X = loaded['x']
y = loaded['y']

first_unit_sequence = X[0]
first_unit_shift_sequence = y[0]
first_sequence = make_predict_sequences(first_unit_sequence)
first_shift_sequence = make_predict_sequences(first_unit_shift_sequence)

first_unit_shift_seq = np.array([first_unit_shift_sequence])

prediction = model.predict(x=first_unit_shift_seq)

sequence_prediction = make_predict_sequences(prediction[0])

print("This what the first sequence of the last batch of 25,000 looks like:")

for s in first_sequence:
 print(s)

print("Now, here's the same sequence, but with the gaps shifted:")

for t in first_shift_sequence:
  print(t)

print("Pretty messy, eh? If only there was a program to shift those gaps back in the right place... Oh, wait!")

for w in sequence_prediction:
  print(w)

print("The same, again, but this time, the difference to the original will be highlighted.")

for n, u in enumerate(sequence_prediction): #This will get a little complicated.
  pred_seq = '' #Currently, make_predict_sequences gives us a string of letters. Here, we'll have to disassemble and reassemble it again.
  nucl1 = list(u) #Here's a list of the predicted sequences.
  nucl2 = list(first_sequence[n]) #And another. Here's a list of the initial sequence that it's being compared to.
  for m, v in enumerate(nucl1):
    if v == nucl2[m]: #This function is supposed to check if the predicted nucleotides and the initial nucleotides match.
      pred_seq += v #If they do, they are added to the current pred_seq without modification
    else: #If not, well, we do modifications
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
  print(pred_seq) #And print the whole thing (it'll be re-created during the next iteration as an empty string and re-filled)
