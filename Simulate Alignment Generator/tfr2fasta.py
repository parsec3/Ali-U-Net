#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import sys
import os
import argparse
from pathlib import Path
from timeit import default_timer as timer

nucleotide = ["A", "C", "G", "T", "-"]
np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser(description="Print fasta files for first N records of TFRecord file. Assumes partition names \'x\' and \'y\'.")
parser.add_argument("file", help="Path to tfrecord file file.")
parser.add_argument("rows", type=int,  help="The rows in the training data.")
parser.add_argument("columns", type=int,  help="The columns in the training data.")
parser.add_argument("N", type=int, default=10, help="Number of record pairs to print.")
args = parser.parse_args()


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
    x = tf.reshape(x, (args.rows, args.columns))  # Assuming the shape of your data
    y = tf.reshape(y, (args.rows, args.columns))   # Assuming the shape of your data
    return x, y

if __name__ == '__main__':
    filename = args.file
    N = int(args.N)

    if not os.path.exists(filename):
        print("Input file ", filename, " does not exist. Exiting.")
        exit()

    start = timer()
    # Create a TFRecordDataset
    dataset = tf.data.TFRecordDataset(filename, buffer_size=1000000)

    # Map the parse function to the dataset
    parsed_dataset = dataset.map(parse_tfrecord_fn).prefetch(100)

    # Iterate through the dataset and print the shape of the NumPy arrays
    count = 0
    unaligned_sequences = []
    aligned_sequences = []

    all_x_data = []
    all_y_data = []

    for x_data, y_data in parsed_dataset:
        print("x NumPy array shape:", type(x_data), " ", x_data.shape)
        print("y NumPy array shape:", type(y_data), " ", y_data.shape)

        unaligned_sequences += [num_to_string(x_data.numpy())]
        aligned_sequences   += [num_to_string(y_data.numpy())]
        count += 1
        if count == N:
            break

    print(unaligned_sequences)

    end1 = timer()
    print("Time used to read tfrecord file: ", end1 - start)

    X_seq = []

    for x in unaligned_sequences:
        list_x = []
        for a in x:
            list_x += [a]
        X_seq += [list_x]

    Y_seq = []

    for y in aligned_sequences:
        list_y = []
        for b in y:
            list_y += [b]
        Y_seq += [list_y]

    end = timer()
    print("Time used to convert to sequences: ", end - end1)
    print("Time used to read and convert: ", end - start)

    basename_without_extension = Path(filename).stem

    for i, s in enumerate(X_seq):
        list_name = []
        outfilename = basename_without_extension + "_x_" + str(i) + ".fas"
        print("Writing file: ", outfilename)
        ofile = open(outfilename, "w")
        number = 0
        for t in range(len(s)):
                number = t + 1
                element = "Seq" + str(number)
                list_name += [element]
                ofile.write(">" + list_name[t] + "\n" +s[t] + "\n") #General format of a FASTA file.
        ofile.close()


    for i, u in enumerate(Y_seq):
        list_name = []
        outfilename = basename_without_extension + "_y_" + str(i) + ".fas"
        print("Writing file: ", outfilename)
        ofile = open(outfilename, "w")
        number = 0
        for v in range(len(u)):
                number = v + 1
                element = "Seq" + str(number)
                list_name += [element]
                ofile.write(">" + list_name[v] + "\n" +u[v] + "\n") #General format of a FASTA file.
        ofile.close()
