# Ali-U-Net
The field of bioinformatics concerns itself with assessing DNA or protein sequences for their similarity, as finding the arrangements with the greatest overlap of residues helds to determine evolutionary relationships. Mismatches can be caused by point mutations, but especially by insertions or deletions (indels) of nucleotides where the entire reading order of the sequence might be shifted.

It is common for DNA alignment software to maximize the overlap by adding gaps to the sequences to account for indels of nucleotides.
E.g.

AACCTT

AATT

These two sequences are poorly aligned, but alignment can be improved.

AACCTT

AA--TT

As each insertion of gaps adds matches and mismatches, algorithms must be assessed for optimality which becomes more difficult with increasing size of the alignment.

The Ali-U-Net is a transformer which learns multiple-sequence alignment through supervised learning. The network along with the simulator for the training data are presented in this repository.
