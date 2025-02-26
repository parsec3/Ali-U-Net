import numpy as np
import keras
import tensorflow as tf
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from  Bio import SeqIO
from statistics import median, mean
import argparse

parser = argparse.ArgumentParser(description="Load the model.")

parser.add_argument("rows", type=int, help="The rows in the training data.")
parser.add_argument("columns", type=int, help="The columns in the training data.")
parser.add_argument("file", help="Path to tfrecord file file.")
parser.add_argument("aligned_file", type=str, help="The aligned fasta file produced by the Ali-U-Net.")

args = parser.parse_args()

rows = args.rows
columns = args.columns
filename = args.file
aligned_file = args.aligned_file

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
        sequence = ''.join(sequence) ##I'll keep this as an on/off-switch in case we need the raw letters.
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
    x = tf.reshape(x, (rows, columns)) # Assuming the shape of your data
    y = tf.reshape(y, (rows, columns)) # Assuming the shape of your data
    return x, y

def get_unaligned_reference(data,N):
    count = 0
    unaligned_sequences = []
    for x_data, y_data in data:
        unaligned_sequences += [num_to_string(x_data.numpy())]
        count += 1
        if count == N:
            break
    x_seq = []
    for x in unaligned_sequences:
        list_x = []
        for b in x:
            list_x += [b]
        x_seq += [list_x]
    return x_seq

# Create a TFRecordDataset
dataset = tf.data.TFRecordDataset(filename, buffer_size=1000000)
# Map the parse function to the dataset
parsed_dataset = dataset.map(parse_tfrecord_fn).prefetch(100)
parsed_unaligned_seq = get_unaligned_reference(parsed_dataset,1000)

def calculate_pairwise_identity(seq1, seq2):
    alignment = pairwise2.align.globalms(seq1, seq2, 1, 0, -2, -1, one_alignment_only=True)[0]
    identical_positions = sum(1 for s1, s2 in zip(*alignment[:2]) if s1 != '-' and s1 == s2)
    total_positions = sum(1 for s1, s2 in zip(*alignment[:2]) if s1 != '-' or s2 != '-')
    identity_score = identical_positions / total_positions
    return identity_score

def shift_gaps_to_right(sequence):
    """Shift gaps in the sequence to the right."""
    parts = sequence.split('-')
    gap_count = len(sequence) - sum(len(part) for part in parts)
    shifted_sequence = ''.join(parts) + '-' * gap_count
    return shifted_sequence

def gapless_sequence(sequence):
    """Shift gaps in the sequence to the right."""
    parts = sequence.split('-')
    gap_count = len(sequence) - sum(len(part) for part in parts)
    shifted_sequence = ''.join(parts)
    return shifted_sequence

Accuracy_List = []

for i in range(1,1001):
    unaligned_seq = parsed_unaligned_seq[i-1] #x

    gapless_seq = parse_fasta(aligned_file+f"_{i}.fasta")

    identity_gapless = []
    for n, u in enumerate(gapless_seq):
      identity_score = calculate_pairwise_identity(u,unaligned_seq[n])
      identity_score = identity_score * 100
      identity_gapless += [identity_score]
    Accuracy_List.append(mean(identity_gapless))

themean = 100 - mean(Accuracy_List)
themedian = 100 - median(Accuracy_List)
res = np.std(Accuracy_List)

print("The Ali-U-Net achieves a median hallucination rate score of ",'{0:.3f}'.format(themedian),"%, a mean rate of ",'{0:.3f}'.format(themean),"% and a standard deviation of ",'{0:.3f}'.format(res))
