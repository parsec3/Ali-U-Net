import argparse
import random
import numpy as np
import tensorflow as tf
import os
import sys



parser = argparse.ArgumentParser(description="Simulate a study case without internal gaps.")

parser.add_argument("columns", type=int, help="The number of columns.")
parser.add_argument("rows", type=int, help="The number of rows.")
parser.add_argument("margin", type=int, help="The margin size.")
parser.add_argument("skip_rows", type=int, help="The number of rows to be left without gaps.")
parser.add_argument("alignments", type=int, help="The number of alignments.")
parser.add_argument("filename", type=str, help="The file name.")

args = parser.parse_args()

rows = args.rows
columns = args.columns
margin = args.margin
skip_rows = args.skip_rows
arrays = args.alignments
filename = args.filename


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

def make_sequences(rows,columns):
  profile = DNA_profile(columns)
  sequences = []
  for i in range(rows):
    sequence = []
    for i in range(columns):
      sequence += random.choices(nucleotides, weights=profile[i], k=1)
    sequence = ''.join(sequence)
    sequences += [sequence]
  sequences = np.array(sequences)
  return sequences


sequences = make_sequences(rows,columns)

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

def create_unaligned_sequence(sequence, left_margin, right_margin, total_columns):
    """Create the unaligned version of a sequence."""
    total_margin = left_margin + right_margin

    # Remove left margin
    del sequence[0:left_margin]

    # Adjust right margin
    sequence = sequence[:total_columns - total_margin] + ['-'] * total_margin
    return sequence


def apply_margins(sequence, left_margin, right_margin):
    """Apply left and right margins to a sequence."""
    sequence = ['-'] * left_margin + sequence[left_margin:]
    sequence = sequence[:len(sequence) - right_margin] + ['-'] * right_margin
    return sequence


def simulate_alignment_generator(rows,columns,margin_size):
  while True:
    sequence_array = make_sequences(rows,columns) #Generate the base sequences

    aligned_sequences = []
    unaligned_sequences = []

    for i in range(rows):
      aligned_sequence = list(sequence_array[i])
      unaligned_sequence = list(sequence_array[i])

      # Skip random rows
      if i in random.sample(range(rows), k=skip_rows):
        aligned_sequences.append(''.join(aligned_sequence))
        unaligned_sequences.append(''.join(unaligned_sequence))
        continue

      # Apply margins
      left_margin = random.randint(0, margin_size)
      right_margin = random.randint(0, margin_size)
      aligned_sequence = apply_margins(aligned_sequence, left_margin, right_margin)

      # Create unaligned sequence
      unaligned_sequence = create_unaligned_sequence(
        unaligned_sequence, left_margin,
        right_margin, columns
      )

      # Save results
      aligned_sequences.append(''.join(aligned_sequence))
      unaligned_sequences.append(''.join(unaligned_sequence))

    # Convert lists to numpy arrays for output
    aligned_sequences = np.array(aligned_sequences)
    unaligned_sequences = np.array(unaligned_sequences)

    # Calculate min and max positions
    yield aligned_sequences, unaligned_sequences

from multiprocessing import Pool

def generate_alignment(_):
    aligned_sequences, unaligned_sequences = next(gap_sequence)
    recoded_aligned_seq = np.array([convert2recode(seq) for seq in aligned_sequences])
    recoded_unaligned_seq = np.array([convert2recode(seq) for seq in unaligned_sequences])
    return recoded_aligned_seq, recoded_unaligned_seq

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

      num_processes = 50
      alignments_per_process = arrays // num_processes

      train_aligned = np.empty((arrays,rows,columns),dtype='uint8') #For the unaligned gaps.
      train_unaligned = np.empty((arrays,rows,columns),dtype='uint8') #For the aligned gaps.
      gap_sequence = simulate_alignment_generator(rows,columns,margin) #Inserts the substitution probability a>
      with Pool(processes=num_processes) as pool:
        results = pool.starmap(generate_alignment, [(None,) for _ in range(arrays)])

      for k, (recoded_aligned_seq, recoded_unaligned_seq) in enumerate(results):
        train_aligned[k] = recoded_aligned_seq
        train_unaligned[k] = recoded_unaligned_seq

      convert_to_TFRecord(train_unaligned,train_aligned,filename)
