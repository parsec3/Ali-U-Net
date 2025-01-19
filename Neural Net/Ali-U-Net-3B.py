# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 18:59:51 2025

@author: Petar
"""

import argparse
import tensorflow as tf
import numpy as np

seed = 42
np.random.seed = seed

parser = argparse.ArgumentParser(description="Train the Ali-U-Net on simulated alignments.")

parser.add_argument("rows", type=int,  help="The rows in the training data.")
parser.add_argument("columns", type=int,  help="The columns in the training data.")
parser.add_argument("activation", type=str, help="The activation function.")
parser.add_argument("initialization", type=str, help="The initialization function.")
parser.add_argument("training_data", type=str, help="The training dataset.")
parser.add_argument("test_data", type=str, help="The test dataset.")
parser.add_argument("file_path", type=str, help="The filepath.")

args = parser.parse_args()

act_fun = args.activation      # "sigmoid" OR "relu"
act_init = args.initialization # "glorot_normal" OR "he_normal"

rows = args.rows       # 48 OR 96
columns = args.columns # 48 OR 96

#Let's adapt the net to our model
inputs = tf.keras.layers.Input(shape=(rows, columns, 5))

#The below output sizes all assume '48' is chosen as a "rows" and as a "columns" argument.

#Input 1

#Contraction path
c1 = tf.keras.layers.Conv2D(32, (11, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(inputs) # Output size: 48x48x32
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(32, (11, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1) # Output size: 24x24x32

c2 = tf.keras.layers.Conv2D(64, (7, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(p1)  # Output size: 24x24x64
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(64, (7, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2) # Output size: 12x12x64

c3 = tf.keras.layers.Conv2D(128, (5, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(p2) # Output size 12x12x128
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(128, (5, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3) # Output size 6x6x128

c4 = tf.keras.layers.Conv2D(128, (4, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(p3) # Output size 6x6x128
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (4, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4) # Output size 3x3x128

c5 = tf.keras.layers.Conv2D(256, (3, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(c5) # Output size is 3x3x256

#Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4]) # Output size: 48x48x32
c6 = tf.keras.layers.Conv2D(128, (4, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (4, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6) # Output size is 12x12x128
u7 = tf.keras.layers.concatenate([u7, c3])  # Output size is 12x12x256
c7 = tf.keras.layers.Conv2D(128, (5, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(128, (5, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7) # Output size is 24x24x64
u8 = tf.keras.layers.concatenate([u8, c2])  # Output size is 24x24x128
c8 = tf.keras.layers.Conv2D(64, (7, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(64, (7, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)  # Output size is 24x24x128
u9 = tf.keras.layers.concatenate([u9, c1], axis=3) # Output size is 48x48x64
c9 = tf.keras.layers.Conv2D(32, (11, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(32, (11, 2), activation=act_fun, kernel_initializer=act_init, padding='same')(c9)

outputs = tf.keras.layers.Conv2D(5, (1, 1), activation='sigmoid')(c9)

x = tf.keras.Model(inputs=inputs,outputs=outputs)


#Input 2

#Contraction path
c1 = tf.keras.layers.Conv2D(32, (2, 11), activation=act_fun, kernel_initializer=act_init, padding='same')(inputs) # Output size: 48x48x32
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(32, (2, 11), activation=act_fun, kernel_initializer=act_init, padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1) # Output size: 24x24x32

c2 = tf.keras.layers.Conv2D(64, (2, 7), activation=act_fun, kernel_initializer=act_init, padding='same')(p1)  # Output size: 24x24x64
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(64, (2, 7), activation=act_fun, kernel_initializer=act_init, padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2) # Output size: 12x12x64

c3 = tf.keras.layers.Conv2D(128, (2, 5), activation=act_fun, kernel_initializer=act_init, padding='same')(p2) # Output size 12x12x128
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(128, (2, 5), activation=act_fun, kernel_initializer=act_init, padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3) # Output size 6x6x128

c4 = tf.keras.layers.Conv2D(128, (2, 4), activation=act_fun, kernel_initializer=act_init, padding='same')(p3) # Output size 6x6x128
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (2, 4), activation=act_fun, kernel_initializer=act_init, padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4) # Output size 3x3x128

c5 = tf.keras.layers.Conv2D(256, (2, 3), activation=act_fun, kernel_initializer=act_init, padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (2, 3), activation=act_fun, kernel_initializer=act_init, padding='same')(c5) # Output size is 3x3x256

#Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4]) # Output size: 48x48x32
c6 = tf.keras.layers.Conv2D(128, (2, 4), activation=act_fun, kernel_initializer=act_init, padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (2, 4), activation=act_fun, kernel_initializer=act_init, padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6) # Output size is 12x12x128
u7 = tf.keras.layers.concatenate([u7, c3])  # Output size is 12x12x256
c7 = tf.keras.layers.Conv2D(128, (2, 5), activation=act_fun, kernel_initializer=act_init, padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(128, (2, 5), activation=act_fun, kernel_initializer=act_init, padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7) # Output size is 24x24x64
u8 = tf.keras.layers.concatenate([u8, c2])  # Output size is 24x24x128
c8 = tf.keras.layers.Conv2D(64, (2, 7), activation=act_fun, kernel_initializer=act_init, padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(64, (2, 7), activation=act_fun, kernel_initializer=act_init, padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)  # Output size is 24x24x128
u9 = tf.keras.layers.concatenate([u9, c1], axis=3) # Output size is 48x48x64
c9 = tf.keras.layers.Conv2D(32, (2, 11), activation=act_fun, kernel_initializer=act_init, padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(32, (2, 11), activation=act_fun, kernel_initializer=act_init, padding='same')(c9)

outputs = tf.keras.layers.Conv2D(5, (1, 1), activation='sigmoid')(c9)

y = tf.keras.Model(inputs=inputs,outputs=outputs)


#Input 3

#Contraction path
c1 = tf.keras.layers.Conv2D(32, (11, 11), activation=act_fun, kernel_initializer=act_init, padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1) # Output size: 48x48x32
c1 = tf.keras.layers.Conv2D(32, (11, 11), activation=act_fun, kernel_initializer=act_init, padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1) # Output size: 24x24x32

c2 = tf.keras.layers.Conv2D(64, (7, 7), activation=act_fun, kernel_initializer=act_init, padding='same')(p1)  # Output size: 24x24x64
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(64, (7, 7), activation=act_fun, kernel_initializer=act_init, padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2) # Output size: 12x12x64

c3 = tf.keras.layers.Conv2D(128, (5, 5), activation=act_fun, kernel_initializer=act_init, padding='same')(p2) # Output size 12x12x128
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(128, (5, 5), activation=act_fun, kernel_initializer=act_init, padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3) # Output size 6x6x128

c4 = tf.keras.layers.Conv2D(128, (4, 4), activation=act_fun, kernel_initializer=act_init, padding='same')(p3) # Output size 6x6x128
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (4, 4), activation=act_fun, kernel_initializer=act_init, padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4) # Output size 3x3x128

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=act_fun, kernel_initializer=act_init, padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=act_fun, kernel_initializer=act_init, padding='same')(c5) # Output size is 3x3x256

#Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4]) # Output size is 6x6x256
c6 = tf.keras.layers.Conv2D(128, (4, 4), activation=act_fun, kernel_initializer=act_init, padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (4, 4), activation=act_fun, kernel_initializer=act_init, padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6) # Output size is 12x12x128
u7 = tf.keras.layers.concatenate([u7, c3])  # Output size is 12x12x256
c7 = tf.keras.layers.Conv2D(128, (5, 5), activation=act_fun, kernel_initializer=act_init, padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(128, (5, 5), activation=act_fun, kernel_initializer=act_init, padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7) # Output size is 24x24x64
u8 = tf.keras.layers.concatenate([u8, c2])  # Output size is 24x24x128
c8 = tf.keras.layers.Conv2D(64, (7, 7), activation=act_fun, kernel_initializer=act_init, padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(64, (7, 7), activation=act_fun, kernel_initializer=act_init, padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)  # Output size is 24x24x128
u9 = tf.keras.layers.concatenate([u9, c1], axis=3) # Output size is 48x48x64
c9 = tf.keras.layers.Conv2D(32, (11, 11), activation=act_fun, kernel_initializer=act_init, padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(32, (11, 11), activation=act_fun, kernel_initializer=act_init, padding='same')(c9)

outputs = tf.keras.layers.Conv2D(5, (1, 1), activation='sigmoid')(c9)

z = tf.keras.Model(inputs=inputs,outputs=outputs)

combined = tf.keras.layers.concatenate([x.output, y.output, z.output])

a = tf.keras.layers.Conv2D(10, (1, 1), activation="relu")(combined)
a = tf.keras.layers.Conv2D(5, (1, 1), activation="softmax")(a)

from tensorflow.io import serialize_tensor, parse_tensor

def parse_record(example):
    print("parse_record_fn has been called.")
    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    x = tf.io.decode_raw(example['x'], tf.uint8)
    y = tf.io.decode_raw(example['y'], tf.uint8)
    x = tf.one_hot(x,5)
    y = tf.one_hot(y,5)
    x = tf.reshape(x, (rows, columns, 5))  # Assuming the shape of your data
    y = tf.reshape(y, (rows, columns, 5))   # Assuming the shape of your data
    return x, y

batch_size = 64
epochs = 50

# Create and compile the model

model = tf.keras.Model(inputs=[inputs],outputs=a)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001) # Default ist 0.001 statt 0.0001
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


training_dataset = tf.data.TFRecordDataset(args.training_data, buffer_size=1000000000)# tf.data.experimental.AUTOTUNE)
test_dataset = tf.data.TFRecordDataset(args.test_data, buffer_size=1000000000)

parsed_training_dataset = training_dataset.map(parse_record, num_parallel_calls=batch_size)
parsed_test_dataset = test_dataset.map(parse_record, num_parallel_calls=batch_size)

# Shuffle and batch the dataset
training_dataset = parsed_training_dataset.shuffle(buffer_size=1000).batch(batch_size)
training_dataset = training_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_dataset = parsed_test_dataset.shuffle(buffer_size=1000).batch(batch_size)
filepath = args.file_path +"/checkpoint-epoch-{epoch:02d}-{val_accuracy:.4f}-.hdf5"
my_callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=filepath,save_weights_only=False,
                                                    monitor='val_accuracy',save_best_only=True)]


# Train the model
history = model.fit(training_dataset,validation_data=(test_dataset), epochs=epochs, verbose=2, callbacks=my_callbacks)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

np.save('acc_3B_10M.npy', acc)
np.save('val_acc_3B_10M.npy', val_acc)
np.save('loss_3B_10M.npy', loss)
np.save('val_loss_3B_10M.npy', val_loss)

model.save('3B_M10.keras')  # The file needs to end with the .keras extension
