import numpy as np
import keras
import tensorflow as tf
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from statistics import median, mean

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

loaded = np.load('filename.npz')

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


x = loaded['x']
y = loaded['y']

Accuracy_List = []


for i in range(1,1001):
    first_unit_sequence = x[i-1]
    first_unit_shift_sequence = y[i-1]
    first_sequence = make_predict_sequences(first_unit_sequence)
    first_shift_sequence = make_predict_sequences(first_unit_shift_sequence)
    gapless_seq = []

    handle = open(r"aligned_sequence_name_{}.fasta".format(i))
    for seq_record_al in SeqIO.parse(handle, "fasta"):
        seq = str(seq_record_al.seq)
        gapless = gapless_sequence(seq.upper())
        gapless_seq += [gapless]
    handle.close()

    identity_gapless = []
    for n, u in enumerate(gapless_seq):
      identity_score = calculate_pairwise_identity(u,first_sequence[n])
      identity_score = identity_score * 100
      identity_gapless += [identity_score]
    Accuracy_List.append(mean(identity_gapless))



themean = 100 - mean(Accuracy_List)
themedian = 100 - median(Accuracy_List)
res = np.std(Accuracy_List)

print("The Ali-U-Net achieves a median hallucination rate score of ",'{0:.3f}'.format(themedian),"%, a mean rate of ",'{0:.3f}'.format(themean),"% and a standard deviation of ",'{0:.3f}'.format(res))
