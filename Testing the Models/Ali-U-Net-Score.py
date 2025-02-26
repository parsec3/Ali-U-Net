import tensorflow as tf
import numpy as np
from statistics import median, mean
import argparse

nucleotide = ["A", "C", "G", "T", "-"]

# Function to parse a FASTA file without using BioPython
def parse_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence.upper())
                    sequence = ''
            else:
                sequence += line
        if sequence:
            sequences.append(sequence.upper())
    return sequences

def num_to_string(pred_array):
    sequences = []
    rows = pred_array.shape[0]
    columns = pred_array.shape[1]
    for i in range(rows):
        sequence = []
        for j in pred_array[i]:
            sequence += nucleotide[j]
        sequence = ''.join(sequence)  ##I'll keep this as an on/off-switch in case we need the raw letters.
        sequences += [sequence]
    sequences = np.array(sequences)
    return sequences

def parse_tfrecord_fn(example):
    print("parse_tfrecord_fn has been called.")
    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    x = tf.io.decode_raw(example['x'], tf.int8)
    y = tf.io.decode_raw(example['y'], tf.int8)
    x = tf.reshape(x, (48, 48))  # Assuming the shape of your data
    y = tf.reshape(y, (48, 48))   # Assuming the shape of your data
    return x, y

def get_ref_alignment(data,N):
    count = 0
    aligned_sequences = []

    for x_data, y_data in data:
        aligned_sequences   += [num_to_string(y_data.numpy())]
        count += 1
        if count == N:
            break

    Y_seq = []

    for y in aligned_sequences:
        list_y = []
        for b in y:
            list_y += [b]
        Y_seq += [list_y]
    return Y_seq

def calculate_pairwise_identity(seq1, seq2):
    if len(seq1) != len(seq2):
        return None

    identical_matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-')
    num_pairs = sum(1 for a, b in zip(seq1, seq2) if a != '-' and b != '-')

    if num_pairs == 0:  # Avoid division by zero
        return None

    identity_score = (identical_matches / num_pairs) * 100
    return identity_score

def calculate_column_score(sequences1, sequences2):
    if len(sequences1) != len(sequences2):
        return None

    alignment_length1 = len(sequences1[0])
    alignment_length2 = len(sequences2[0])

    for seq in sequences1:
        if len(seq) != alignment_length1:
            return None
    for seq in sequences2:
        if len(seq) != alignment_length2:
            return None

    if alignment_length1 != alignment_length2:
        return None

    num_columns = len(sequences1[0])
    identical_columns = 0
    for i in range(num_columns):
        col1 = ''.join(seq[i] for seq in sequences1)
        col2 = ''.join(seq[i] for seq in sequences2)
        if col1 == col2:
            identical_columns += 1

    column_score = (identical_columns / num_columns) * 100
    return column_score

parser = argparse.ArgumentParser(description="Load the model.")

parser.add_argument("file", help="Path to tfrecord file file.")
parser.add_argument("aligned_file", type=str,  help="The aligned fasta file produced by the Ali-U-Net.")

args = parser.parse_args()

filename = args.file
filename = args.aligned_file

# Create a TFRecordDataset
dataset = tf.data.TFRecordDataset(filename, buffer_size=1000000)

# Map the parse function to the dataset
parsed_dataset = dataset.map(parse_tfrecord_fn).prefetch(100)
parsed_ref_aligned_seq = get_ref_alignment(parsed_dataset,1000)

Accuracy_List = []
Accuracy_List2 = []
Accuracy_List3 = []

for i in range(1, 1001):
    ref_aligned_seq  = parsed_ref_aligned_seq[i-1]

    AliU_aligned_seq = parse_fasta(aligned_file+f"_{i}.fasta")

    accurate_hits = 0
    inaccurate_hits = 0

    for n, u in enumerate(AliU_aligned_seq):
        pred_seq = ''
        nucl1 = list(u)
        nucl2 = list(ref_aligned_seq[n])
        for m, v in enumerate(nucl1):
            if v == nucl2[m]:
                pred_seq += v
                accurate_hits += 1
            else:
                inaccurate_hits += 1
                pred_seq += v.lower() if v in "ACGT" else '~'

    total_hits = accurate_hits + inaccurate_hits
    accuracy = (accurate_hits / total_hits) * 100
    Accuracy_List.append(accuracy)

    # Calculate sum-of-pairs score
    identity = []
    for n, u in enumerate(AliU_aligned_seq):
        remain = len(AliU_aligned_seq) - n
        for i in range(remain-1):
            identity_score = calculate_pairwise_identity(u,AliU_aligned_seq[n+i+1])
            if identity_score != None:
               identity += [identity_score]
    Accuracy_List2.append(mean(identity))
    
    # Calculate column score
    column_score = calculate_column_score(AliU_aligned_seq,ref_aligned_seq)
    if column_score != None:
      Accuracy_List3.append(column_score)

# Print results
print(Accuracy_List)
themean = mean(Accuracy_List)
themedian = median(Accuracy_List)
res = np.std(Accuracy_List)

print(f"The Ali-U-Net achieves a median reference identity score of {themedian:.3f}%, "
      f"a mean reference identity score of {themean:.3f}% and a standard deviation of {res:.3f}.")

print(Accuracy_List2)
themean2 = mean(Accuracy_List2)
themedian2 = median(Accuracy_List2)
res2 = np.std(Accuracy_List2)

print(f"The Ali-U-Net achieves a median sum-of-pairs score of {themedian2:.3f}%, "
      f"a mean sum-of-pairs score of {themean2:.3f}% and a standard deviation of {res2:.3f}.")

print(Accuracy_List3)
themean3 = mean(Accuracy_List3)
themedian3 = median(Accuracy_List3)
res3 = np.std(Accuracy_List3)

print(f"The Ali-U-Net achieves a median column score of {themedian3:.3f}%, "
      f"a mean column score of {themean3:.3f}% and a standard deviation of {res3:.3f}.")
