# Alig-Net
A neural network for solving DNA alignment problems in bioinformatics.

Some background for non-biologists:
A large part of the field of bioinformatics consists of comparing DNA or protein sequences and seeing how similar they are in order to determine evolutionary relationships. Finding the arrangement with the greatest overlap of nucleotides is called alignment. Common reasons why different DNA strands will not overlap, even if they are of closely related specimens, are genetic mutations. Common genetic mutations include, point mutations, where one nucleotide is replaced by another, indels, where nucleotides are being added or removed. The latter are generally more severe because, while point mutations can cause mismatch among one or two nucleotides, having one nucleotide too much or too little can shift the entire reading order sequence.

It is common for DNA alignment software to maximize the overlap by adding gaps to the sequences. This is meant to account for differences in the reading order caused by indels.
E.g.

AACCTT

AATT

These two sequences are poorly aligned, but alignment can be improved.

AACCTT

AA--TT

Naturally, adding gaps makes the alignment worse in other ways, so biologists must evaluate whether the presence of gaps creates more matches than its absence.

The neural net presented here will deal with two problems.

First, it will deal with a multiple-sequence alignment where the gaps have been placed incorrectly and place them correctly.
Second, it will be presented with unaligned DNA sequences and it will be forced to correctly place gaps in a way that maximizes overlap.

For that end, the neural net will receive matrices of correctly and incorrectly aligned DNA multiple-sequence alignments as training data to perform supervised learning.
