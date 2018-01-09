from random import choice
import gzip
from pysster.Motif import Motif


def rand_dna(length):
    return "".join(choice("ACGT") for x in range(length))


num = 5000
with gzip.open("artifical_pos.fasta.gz", "wt") as handle:
    seqs = [rand_dna(20) + "CCCCCCCCCC" + rand_dna(20) + "GGGGGGGGGG" + rand_dna(80) for x in range(num)]
    for x in range(num):
        handle.write(">1\n{}\n".format(seqs[x]))
    Motif("ACGT", seqs).plot().save("pos_half1.png")
    seqs = [rand_dna(80) + "AAAAAAAAAA" + rand_dna(20) + "TTTTTTTTTT" + rand_dna(20) for x in range(num)]
    for x in range(num):
        handle.write(">1\n{}\n".format(seqs[x]))
    Motif("ACGT", seqs).plot().save("pos_half2.png")


with gzip.open("artifical_neg.fasta.gz", "wt") as handle:
    seqs = [rand_dna(140) for x in range(num*2)]
    Motif("ACGT", seqs).plot().save("neg.png")
    for x in range(num*2):
        handle.write(">1\n{}\n".format(seqs[x]))