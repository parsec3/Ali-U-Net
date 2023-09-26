#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 20:26:55 2023

@author: Petar
"""

#Here's where we get to the meat. We will use a modified version of the U-Net for biomedical image segmentation developed by the University of Freiburg for this task.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

class DepthwiseSeparableConv2D(keras.layers.Layer):
  def __init__(self, filters, kernel_size, activation, padding):
    super(DepthwiseSeparableConv2D, self).__init__()
    self.depthwise = tf.keras.layers.DepthwiseConv2D(kernel_size = kernel_size, padding = padding, activation = activation)
    self.pointwise = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1, 1), activation = activation)

  def call(self, input_tensor):
    x = self.depthwise(input_tensor)
    return self.pointwise(x)

seed = 42
np.random.seed = seed

#Let's adapt the net to our model
inputs = tf.keras.layers.Input(shape=(96, 96, 5))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = DepthwiseSeparableConv2D(32, (11, 2), activation='relu', padding='same')(s) #96x96; the notes here are mostly to keep track of the current size of the DNA-matrix
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = DepthwiseSeparableConv2D(32, (11, 2), activation='relu', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1) #48

c2 = DepthwiseSeparableConv2D(64, (11, 2), activation='relu', padding='same')(p1) #48
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = DepthwiseSeparableConv2D(64, (11, 2), activation='relu', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2) #24

c3 = DepthwiseSeparableConv2D(128, (11, 2), activation='relu', padding='same')(p2) #24
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = DepthwiseSeparableConv2D(128, (11, 2), activation='relu', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3) #12

c4 = DepthwiseSeparableConv2D(128, (11, 2), activation='relu', padding='same')(p3) #12
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = DepthwiseSeparableConv2D(128, (11, 2), activation='relu', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4) #6

c5 = DepthwiseSeparableConv2D(256, (11, 2), activation='relu', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = DepthwiseSeparableConv2D(256, (11, 2), activation='relu', padding='same')(c5) #6

#Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5) #12
u6 = tf.keras.layers.concatenate([u6, c4]) #12 12
c6 = DepthwiseSeparableConv2D(128, (11, 2), activation='relu', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = DepthwiseSeparableConv2D(128, (11, 2), activation='relu', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6) #24
u7 = tf.keras.layers.concatenate([u7, c3]) #24 24
c7 = DepthwiseSeparableConv2D(64, (11, 2), activation='relu', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = DepthwiseSeparableConv2D(64, (11, 2), activation='relu', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7) #48
u8 = tf.keras.layers.concatenate([u8, c2]) #48 48
c8 = DepthwiseSeparableConv2D(32, (11, 2), activation='relu', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = DepthwiseSeparableConv2D(32, (11, 2), activation='relu', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8) #96
u9 = tf.keras.layers.concatenate([u9, c1], axis=3) #96 96
c9 = DepthwiseSeparableConv2D(16, (11, 2), activation='relu', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = DepthwiseSeparableConv2D(16, (11, 2), activation='relu', padding='same')(c9)


outputs = tf.keras.layers.Conv2D(5, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs],outputs=[outputs])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

loaded = np.load('compressed_shift_sequences.npz') #We load what we previously created.

from sklearn.model_selection import train_test_split
X = loaded['x']
y = loaded['y']

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

model.save('my_model.keras')
