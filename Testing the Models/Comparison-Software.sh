##Job settings

conda activate biotool


#time for i in {1..1000}; do mafft --thread 1 --localpair "path/to/unaligned_sequence_name_$i.fasta" > "aligned_sequence_name_$i.fasta"; done
#time for i in {1..1000}; do muscle -threads 1 -align path/to/unaligned_sequence_name_$i.fasta -output aligned_sequence_name_$i.fasta; done
#time for i in {1..1000}; do clustalo --in="path/to/unaligned_sequence_name_$i.fasta" -o aligned_sequence_name_$i.fasta --threads=1 --force; done
#time for i in {1..1000}; do t_coffee "path/to/unaligned_sequence_name_$i.fasta" -multi_core=no; done
