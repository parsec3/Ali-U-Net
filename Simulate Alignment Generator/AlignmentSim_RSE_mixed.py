import argparse
import random
import numpy as np
import tensorflow as tf
import os
import sys

parser = argparse.ArgumentParser(description="Train a neural network on the 48x48 net with internal gaps.")

parser.add_argument("rows", type=int, help="The number of rows.")
parser.add_argument("columns", type=int, help="The number of columns.")
parser.add_argument("margin", type=int, help="The margin size.")
parser.add_argument("gap_prob", type=int, help="The gap probability distribution (if the input is '0', the first distribution is chosen, if it is '1', the second is chosen.")
parser.add_argument("skip_rows", type=int, help="The number of rows to be left without gaps.")
parser.add_argument("alignments", type=int, help="The number of alignments.")
parser.add_argument("filename", type=str, help="The file name.")

args = parser.parse_args()

rows = args.rows
columns = args.columns
margin = args.margin
gap_prob = args.gap_prob
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
probabilities = [0.90,0.04,0.03,0.03]

def make_sequences(rows,columns):
  profile = DNA_profile(columns)
  sequences = []
  for i in range(rows):
    sequence = []
    for i in range(columns):
      sequence += random.choices(nucleotides, weights=profile[i], k=1)
    sequence = ''.join(sequence)  ##I'll keep this as an on/off-switch in case we need the raw letters.
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

# Configuration
gap_sizes = [2,3,4]

gap_probabilities1 = [0.5,0.25,0.25] # Gap probabilities for the 48x48 study case
gap_probabilities2 = [0.25,0.25,0.5] # Gap probabilities for the 96x96 study case

gap_probabilities_list = [gap_probabilities1, gap_probabilities2]

gap_probabilities = gap_probabilities_list[gap_prob]

growth_options = [0,1,2]
growth_probabilities = [0.80,0.1,0.1]

gap_types = ['IG','NOIG']

def apply_margins(sequence, left_margin, right_margin):
    """Apply left and right margins to a sequence."""
    sequence = ['-'] * left_margin + sequence[left_margin:]
    sequence = sequence[:len(sequence) - right_margin] + ['-'] * right_margin
    return sequence

def apply_gaps(sequence, gap_position, gap_size):
    """Insert gaps at specified positions in a sequence."""
    sequence = sequence[:gap_position] + ['-'] * gap_size + sequence[gap_position + gap_size:]
    return sequence

def create_unaligned_sequence_rse(sequence, left_margin, right_margin, total_columns):
    """Create the unaligned version of a sequence."""
    total_margin = left_margin + right_margin

    # Remove left margin
    del sequence[0:left_margin]

    # Adjust right margin
    sequence = sequence[:total_columns - total_margin] + ['-'] * total_margin
    return sequence


def create_unaligned_sequence_rse_ig(sequence, left_margin, gap_positions, gap_sizes, right_margin, total_columns):
    """Create the unaligned version of a sequence."""
    total_margin = sum(gap_sizes) + left_margin + right_margin

    # Remove left margin
    del sequence[0:left_margin]
    del sequence[(gap_positions[0]-left_margin):(gap_positions[0]+gap_sizes[0]-left_margin)]
    del sequence[(gap_positions[1]-left_margin-gap_sizes[0]):(gap_positions[1]+gap_sizes[1]-gap_sizes[0]-left_margin)]

    # Adjust right margin
    sequence = sequence[:total_columns - total_margin] + ['-'] * total_margin
    return sequence



def simulate_alignment_generator(rows,columns,margin_size,gap_type):
  max_gap_size = max(gap_sizes) + max(growth_options)
  while True:
    sequence_array = make_sequences(rows,columns) #Generate the base sequences

    left_gap_size = random.choices(gap_sizes, weights=gap_probabilities, k=1)[0] #Randomly pick gap sizes
    right_gap_size = random.choices(gap_sizes, weights=gap_probabilities, k=1)[0]

    left_gap_position = random.randint(max_gap_size,int(columns/2)-2*max(gap_sizes)) #Randomly generate gap positions
    right_gap_position = random.randint(int(columns/2)+max(gap_sizes),columns-2*max(gap_sizes)-max(growth_options))

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


      # Adjust gap positions and sizes using growth options
      growth_factors = random.choices(growth_options, weights=growth_probabilities, k=4)
      adjusted_left_gap_position = left_gap_position - growth_factors[0]
      adjusted_right_gap_position = right_gap_position - growth_factors[1]
      adjusted_left_gap_size = left_gap_size + growth_factors[2] + growth_factors[0]
      adjusted_right_gap_size = right_gap_size + growth_factors[3] + growth_factors[1]

      # Apply margins
      left_margin = random.randint(0, margin_size)
      right_margin = random.randint(0, margin_size)

      if gap_type == "IG":
        aligned_sequence = apply_margins(aligned_sequence, left_margin, right_margin)

        # Apply gaps
        aligned_sequence = apply_gaps(aligned_sequence, adjusted_left_gap_position, adjusted_left_gap_size)
        aligned_sequence = apply_gaps(aligned_sequence, adjusted_right_gap_position, adjusted_right_gap_size)

        # Create unaligned sequence
        unaligned_sequence = create_unaligned_sequence_rse_ig(
          unaligned_sequence, left_margin,
          [adjusted_left_gap_position, adjusted_right_gap_position],
          [adjusted_left_gap_size, adjusted_right_gap_size],
          right_margin, columns
        )

      else:
        aligned_sequence = apply_margins(aligned_sequence, left_margin, right_margin)

        # Create unaligned sequence
        unaligned_sequence = create_unaligned_sequence_rse(
          unaligned_sequence, left_margin,
          right_margin, columns
        )

      # Save results
      aligned_sequences.append(''.join(aligned_sequence))
      unaligned_sequences.append(''.join(unaligned_sequence))

    # Convert lists to numpy arrays for output
    aligned_sequences = np.array(aligned_sequences)
    unaligned_sequences = np.array(unaligned_sequences)

    yield (
      aligned_sequences, unaligned_sequences,
    )



from multiprocessing import Pool

def generate_alignment(_):
    gap_sequence = simulate_alignment_generator(rows,columns,margin,gap_types[random.randint(0, 1)])

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
      with Pool(processes=num_processes) as pool:
        results = pool.starmap(generate_alignment, [(None,) for _ in range(arrays)])

      for k, (recoded_aligned_seq, recoded_unaligned_seq) in enumerate(results):
        train_aligned[k] = recoded_aligned_seq
        train_unaligned[k] = recoded_unaligned_seq

      convert_to_TFRecord(train_unaligned,train_aligned,filename)

