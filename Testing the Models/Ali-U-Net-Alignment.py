import numpy as np
import keras
import tensorflow as tf
import os
import argparse

#Tracking time

import time
start = time.time()

#This ensures comparable conditions to other alignment software.

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['openmp'] = 'True'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

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

def convert2unit_vector(sequence_array): ##Now for the unit vector.
        sequence_array = sequence_array[0]
        sequences = [] #Build the array new.
        for i in range(sequence_array.shape[0]): #Loops through all the rows of the array.
            sequence = list(sequence_array[i]) #Access the rows individually.
            seq = recode_seq(sequence)
            seq = tf.keras.utils.to_categorical(seq, num_classes=5, dtype='uint8') #The extra class is necessary; we have four nucleotides + a gap.
            sequences += [seq]
        sequences = np.array(sequences)
        return sequences

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

seed = 42
np.random.seed = seed

parser = argparse.ArgumentParser(description="Load the model.")

parser.add_argument("rows", type=int,  help="The rows in the training data.")
parser.add_argument("columns", type=int,  help="The columns in the training data.")
parser.add_argument("model", type=str,  help="The model file.")
parser.add_argument("input_file", type=str,  help="The input file.")
parser.add_argument("output_file", type=str,  help="The output file.")

args = parser.parse_args()

rows = args.rows       # 48 OR 96
columns = args.columns # 48 OR 96
model_path = args.model
input_file = args.input_file
output_file = args.output_file

model = tf.keras.models.load_model(model_path)

Accuracy_List = []
seq_num = 0

for i in range(1, 1001):
    unaligned_seq = []

    # Open and read the FASTA file manually
    with open(input_file+f"_{i}.fasta", "r") as handle:
        lines = handle.readlines()
        for line in lines:
            line = line.strip()
            # Ignore header lines starting with '>'
            if not line.startswith(">"):
                unaligned_seq.append(line.upper())

    # Convert the sequence list into the desired numpy array format
    unaligned_seq_array = np.array([unaligned_seq])
    unaligned_seq_array = np.array([unaligned_seq])
    unaligned_unit_seq = convert2unit_vector(unaligned_seq_array)
    unaligned_unit_seq = unaligned_unit_seq.reshape(1, rows, columns, 5)

    # Predict using the model
    prediction = model.predict(x=unaligned_unit_seq)
    prediction = make_predict_sequences(prediction[0])

    # Writing the predicted sequences to a new FASTA file
    list_seq = prediction
    list_name = []

    with open(output+f"_{i}.fasta", "w") as ofile:
        for t, seq in enumerate(list_seq):
            number = t + 1
            element = f"Seq{number}"
            list_name.append(element)
            # Write FASTA format: header and sequence
            ofile.write(f">{element}\n{seq}\n")

end = time.time()
print(end - start)
