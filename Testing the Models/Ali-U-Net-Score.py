import numpy as np
from statistics import median, mean

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

def make_predict_sequences(pred_array):
    sequences = []
    rows = pred_array.shape[0]
    columns = pred_array.shape[1]
    for i in range(rows):
        weight_profile = pred_array[i]
        sequence = []
        for j in range(columns):
            probs = list(weight_profile[j])
            position = probs.index(max(probs))
            sequence += nucleotide[position]
        sequence = ''.join(sequence)  # Create sequence string
        sequences.append(sequence)
    return np.array(sequences)

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

# Load the data
loaded = np.load('filename.npz')
y = loaded['y']

Accuracy_List = []
Accuracy_List2 = []
Accuracy_List3 = []

for i in range(1, 1001):
    first_unit_shift_sequence = y[i-1]
    first_shift_sequence = make_predict_sequences(first_unit_shift_sequence)
    aligned_seq = parse_fasta(f"filepath/to/aligned_sequence_name_{i}.fasta")

    accurate_hits = 0
    inaccurate_hits = 0

    for n, u in enumerate(aligned_seq):
        pred_seq = ''
        nucl1 = list(u)[0:96]
        nucl2 = list(first_shift_sequence[n])
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
    for n, u in enumerate(aligned_seq):
        remain = 96 - n
        for i in range(remain-1):
            identity_score = calculate_pairwise_identity(u, aligned_seq[n + i + 1])
            if identity_score is not None:
                identity.append(identity_score)
    if identity:
        Accuracy_List2.append(mean(identity))

    # Calculate column score
    column_score = calculate_column_score(aligned_seq, first_shift_sequence)
    if column_score is not None:
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
