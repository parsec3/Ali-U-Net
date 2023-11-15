# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:14:51 2023

@author: Petar
"""

import numpy as np
from hmmlearn import hmm
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def simulate_msa(hidden_states, emission_states, transitions, emissions, n_sequences, seq_length):
    # Create an HMM model
    model = hmm.CategoricalHMM(n_components=len(hidden_states), n_iter=100)#, n_symbols=len(emission_states))

    # Set model parameters
    model.startprob_ = transitions['start']
    model.transmat_ = transitions['trans']
    model.emissionprob_ = emissions

    # Generate sequences
    sequences, states = model.sample(n_sequences * seq_length)
    sequences = sequences.reshape((n_sequences, seq_length))

    # Convert numeric sequences to letters
    sequence_symbols = []

    for i in range(n_sequences):
      subseq = []
      for s in sequences[i]:
        subseq += emission_states[s]
      subseq = "".join(subseq)
      sequence_symbols += [subseq]
    sequence_symbols = np.array(sequence_symbols)
    return sequence_symbols

def write_fasta(sequences, filename):
    records = [SeqRecord(Seq(''.join(seq)), id=f'Seq_{i+1}', description='') for i, seq in enumerate(sequences)]
    SeqIO.write(records, filename, "fasta")

if __name__ == "__main__":
    # Define HMM parameters
    hidden_states = ['H1', 'H2', 'H3']  # Hidden states
    emission_states = ['A', 'C', 'G', 'T']  # Emission states (nucleotides)
    n_sequences = 96
    seq_length = 96

    # Define transition probabilities
    transitions = {
        'start': np.array([0.5, 0.3, 0.2]),
        'trans': np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.3, 0.1, 0.6],
        ])
    }

    # Define emission probabilities
    emissions = np.array([
        [0.25, 0.25, 0.25, 0.25],  # H1 emissions
        [0.1, 0.4, 0.4, 0.1],      # H2 emissions
        [0.2, 0.3, 0.3, 0.2],      # H3 emissions
    ])

    # Simulate MSA
    simulated_sequences = simulate_msa(hidden_states, emission_states, transitions, emissions, n_sequences, seq_length)

    # Write simulated sequences to a FASTA file
    output_filename = 'simulated_msa.fasta'
    print(simulated_sequences)
 #   write_fasta(simulated_sequences, output_filename)

 #   print(f"Simulated MSA saved to {output_filename}")
