#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 20:26:55 2023

@author: Petar
"""

#This is copied from the videos by DigitalSreeni ("Image Segmentation using U-Net").
#I also tried Aladdin Persson's U-Net where he built it from scratch, but I didn't understand it so well.

import tensorflow as tf
import os
import random
import numpy as np

from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
seed = 42
np.random.seed = seed

#Let's adapt the net to our model

inputs = tf.keras.layers.Input(shape=(96, 96, 5))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s) #96
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1) #48

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1) #48
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2) #24

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2) #24
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3) #12

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3) #12
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4) #6

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5) #6

#Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5) #12
u6 = tf.keras.layers.concatenate([u6, c4]) #12 12
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6) #24
u7 = tf.keras.layers.concatenate([u7, c3]) #24 24
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7) #48
u8 = tf.keras.layers.concatenate([u8, c2]) #48 48
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8) #96
u9 = tf.keras.layers.concatenate([u9, c1], axis=3) #96 96
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(5, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs],outputs=[outputs])
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #Loss überlegen.
#Möglichkeit für starken loss: intersection_over_union
model.summary()

#train_no_shift = open('unshifted.npy', 'rb')
#train_shift = open('shifted.npy', 'rb')

#with open('shift_sequences.npy', 'rb') as f:
 #   train_no_shift = np.load(f)
  #  train_shift = np.load(f)

loaded = np.load('compressed_shift_sequences.npz')

from sklearn.model_selection import train_test_split
X = loaded['x']
y = loaded['y']


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
 #         print(position)
#      sequence += random.choices(nucleotide, weights=list(weight_profile[j]), k=1)
    sequence = ''.join(sequence)  ##I'll keep this as an on/off-switch in case we need the raw letters.
    sequences += [sequence]
  sequences = np.array(sequences)
  return sequences

first_unit_sequence = X[0]
first_unit_shift_sequence = y[0]
first_sequence = make_predict_sequences(first_unit_sequence)
first_shift_sequence = make_predict_sequences(first_unit_shift_sequence)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=3),
                tf.keras.callbacks.ModelCheckpoint(filepath="Checkpoints-dir",save_weights_only=True,
                                                   monitor='val_accuracy',save_best_only=True)]

history = model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=20,epochs=10,verbose=1,callbacks=my_callbacks)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Add curve to plot
plt.plot(acc, label='Training accuracy')
# Add curve to plot
plt.plot(val_acc, label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
# Create first plot for content specified above
plt.figure()
plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
# Create another plot and show all plots
plt.show()

first_unit_shift_seq = np.array([first_unit_shift_sequence])

prediction = model.predict(x=first_unit_shift_seq)

sequence_prediction = make_predict_sequences(prediction[0])

print("This what the first sequence of the last batch of 1,000 looks like:")

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
    if v == nucl2[m]: #This function is supposed to check if the predicted nucleotides and hte initial nucleotides match.
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
