import random
import numpy as np
import tensorflow as tf
import os
import sys
import argparse

parser = argparse.ArgumentParser(description="Train a neural network on the 48x48 net with internal gaps.")
parser.add_argument("rows", type=int, help="The number of rows.")
parser.add_argument("columns", type=int, help="The number of columns.")
parser.add_argument("arrays", type=int, help="The number of arrays.")
parser.add_argument("filename", type=str, help="The file name.")
args = parser.parse_args()


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

def DNA_profile(columns,probabilities):
  profile=[]
  for i in range(columns):
    prob = random.sample(probabilities, len(probabilities)) #Shuffle around so that the spotlight is always on a different nucleotide.
    profile+=[prob]
  return profile

nucleotides = ["A", "C", "G", "T"]
probabilities0 = [0.90,0.04,0.03,0.03]

def make_sequences(rows,columns,probabilities):
  profile = DNA_profile(columns,probabilities)
  sequences = []
  for i in range(rows):
    sequence = []
    for i in range(columns):
      sequence += random.choices(nucleotides, weights=profile[i], k=1)
    sequence = ''.join(sequence)
    sequences += [sequence]
  sequences = np.array(sequences)
  return sequences

sequences = make_sequences(96,96,probabilities0)

probabilities1 = [1,0,0,0]
probabilities2 = [0.90,0.04,0.03,0.03]
prob_list = [probabilities1,probabilities2]


def simulate_alignment_generator(rows, columns, probabilities):
    while True:
        sequence_array = make_sequences(rows, columns, probabilities)  # Generate base sequences
        sequences = []
        shift_sequences = []

        # Unstructured gap logic
        num_gaps = random.randint(1, 3)  # Random number of gaps, e.g., 1 to 3
        for i in range(rows):
            sequence = list(sequence_array[i])
            shift_sequence = list(sequence_array[i])  # Copy for shifted sequence variant
            skip2 = random.choices(range(rows),k=num_gaps)

            if i in skip2:
                gap_length = random.randint(1, int(columns * 0.2))  # Variable gap size, up to 20% of column length
                gap_position = random.randint(0, columns - gap_length)
                margin_position = gap_position + gap_length
                sequence = sequence[:gap_position] + ['-'] * gap_length + sequence[margin_position:]

                del shift_sequence[gap_position:margin_position]
                shift_sequence = shift_sequence + ['-'] * gap_length

            # Convert lists back to strings and store them
            sequence = ''.join(sequence)
            shift_sequence = ''.join(shift_sequence)
            sequences.append(sequence)
            shift_sequences.append(shift_sequence)

    # Convert to numpy arrays for consistency with the original code
    sequences = np.array(sequences)
    shift_sequences = np.array(shift_sequences)

    yield sequences, shift_sequences

rows = args.rows
columns = args.columns

from multiprocessing import Pool

def generate_alignment(_):
    gap_sequence = simulate_alignment_generator(
        rows, columns, prob_list[random.randint(0, 1)], gap_list[1],
        margin_list[random.randint(0, 1)], growth_list[random.randint(0, 1)], gap_types[1]
    )

    gap_sequences, shift_gap_sequences = next(gap_sequence)

    if gap_sequences.size == 0 or shift_gap_sequences.size == 0:
        print("Error: Generated sequences are empty.")

    recoded_gap_seq = np.array([convert2recode(seq) for seq in gap_sequences])
    recoded_shift_gap_seq = np.array([convert2recode(seq) for seq in shift_gap_sequences])
    return recoded_gap_seq, recoded_shift_gap_seq



numeric_mapping = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
    '-': 4
}

def convert2recode(sequence):
    # Create an array to hold the recoded values with the same length as the input sequence
    seq_len = len(sequence)
    recoded_array = np.empty(seq_len, dtype=np.uint8)

    # Map each nucleotide to its corresponding code
    for i, nucleotide in enumerate(sequence):
        if nucleotide in numeric_mapping:
            recoded_array[i] = numeric_mapping[nucleotide]

        else:
            raise ValueError(f"Invalid nucleotide '{nucleotide}' in sequence.")

    return recoded_array


def convert_to_TFRecord(x_partition,y_partition, filename_out):

    # Define a function to convert numpy arrays to tf.train.Feature
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))

    # Create TFRecord writer
    tfrecord_filename = filename_out
    with tf.io.TFRecordWriter(tfrecord_filename) as writer:
        for i in range(len(x_partition)):
            x_feature = _bytes_feature(x_partition[i])
            y_feature = _bytes_feature(y_partition[i])

            # Create a tf.train.Features object
            features = tf.train.Features(feature={
                'x': x_feature,
                'y': y_feature
            })

            # Create a tf.train.Example object

            example = tf.train.Example(features=features)

            # Serialize the tf.train.Example to a string
            serialized_example = example.SerializeToString()

            # Write the serialized example to the TFRecord file
            writer.write(serialized_example)

    print(f"TFRecord file '{tfrecord_filename}' has been created.")

if __name__ == "__main__":
  train_shift = []
  train_no_shift = []

  arrays = args.arrays #Decide the number of arrays here.
  num_processes = 50
  alignments_per_process = arrays // num_processes

  train_no_shift = np.empty((arrays,rows,columns),dtype='uint8') #For the unshifted gaps.
  train_shift = np.empty((arrays,rows,columns),dtype='uint8') #For the shifted gaps.
  with Pool(processes=num_processes) as pool:
    results = pool.starmap(generate_alignment, [(None,) for _ in range(arrays)])
  for k, (recoded_gap_seq, recoded_shift_gap_seq) in enumerate(results):
    train_no_shift[k] = recoded_gap_seq
    train_shift[k] = recoded_shift_gap_seq
  filename = args.filename
  convert_to_TFRecord(train_shift,train_no_shift,filename)
