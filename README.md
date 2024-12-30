# Ali-U-Net
The field of bioinformatics concerns itself with assessing DNA or protein sequences for their similarity, as finding the arrangements with the greatest overlap of residues helds to determine evolutionary relationships. Mismatches can be caused by point mutations, but especially by insertions or deletions (indels) of nucleotides where the entire reading order of the sequence might be shifted.

It is common for DNA alignment software to maximize the overlap by adding gaps to the sequences to account for indels of nucleotides.
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
