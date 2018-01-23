# this script prepares the positive and negative sets for the RBPs
# the data is based on CLIP-seq data sets provided in: https://github.molgen.mpg.de/heller/ssHMM_data
# positive sets contain CLIP-seq binding sites from the protein of interest
# negative sets contain randomly selected CLIP-seq binding sites from 24 other proteins
# all sequences are of length 200 and centered at a binding site
# positive and negative sets are balanced w.r.t. the number of sequences


from urllib.request import urlretrieve
from subprocess import call
from Bio import SeqIO
from Bio.Alphabet import IUPAC
import gzip
import shutil
import os
import random

random.seed(42)

# download human genome
urlretrieve("ftp://ftp.sanger.ac.uk/pub/gencode/Gencode_human/release_19/GRCh37.p13.genome.fa.gz", "grch37.fasta.gz")
with gzip.open("grch37.fasta.gz", "rb") as f_in:
    with open("grch37.fasta", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
os.remove("grch37.fasta.gz")


RBPs = [("https://github.molgen.mpg.de/raw/heller/ssHMM_data/master/clip-seq/fasta/PUM2/positive.fasta",
         "https://github.molgen.mpg.de/raw/heller/ssHMM_data/master/clip-seq/fasta/PUM2/negative_clip.fasta", "pum2"),
        ("https://github.molgen.mpg.de/raw/heller/ssHMM_data/master/clip-seq/fasta/QKI/positive.fasta",
         "https://github.molgen.mpg.de/raw/heller/ssHMM_data/master/clip-seq/fasta/QKI/negative_clip.fasta", "qki")]

for entry in RBPs:
    # get the files
    urlretrieve(entry[0], "positive.fasta")
    urlretrieve(entry[1], "negative.fasta")

    # extract coordinates from fasta headers, save as bed
    # extend sequences by +-100 centered at the middle of the binding site
    for file_name in ["positive", "negative"]:
        headers = [rec.id for rec in SeqIO.parse("{}.fasta".format(file_name), "fasta")]
        with open("{}.bed".format(file_name), "wt") as handle:
            for rec in headers:
                chrom, rest = rec.split(":")
                start = int(rest.split("-")[0])
                rest = rest.split("-")[1:]
                end = int(rest[0].split("(")[0])
                mid = int((start+end)/2)
                if mid-100 < 0: continue
                if len(rest) == 1:
                    strand = '+'
                else:
                    strand = '-'
                handle.write("{}\t{}\t{}\t.\t.\t{}\n".format(chrom, mid-100, mid+100, strand))
        # get sequences based on the bed file
        call("bedtools getfasta -s -fo {}.fasta -fi grch37.fasta -bed {}.bed".format(file_name, file_name), shell = True)
    
    # randomly split into train and test, first positives...
    seqs = [rec for rec in SeqIO.parse("positive.fasta", "fasta")]
    for i in range(len(seqs)):
        seqs[i].seq = seqs[i].seq.transcribe()
    num_train = int(len(seqs) * 0.75)
    num_test = len(seqs) - num_train
    random.shuffle(seqs)
    SeqIO.write(seqs[:num_train], "{}.train.positive.fasta".format(entry[2]), "fasta")
    SeqIO.write(seqs[num_train:], "{}.test.positive.fasta".format(entry[2]), "fasta")

    # ... and then negatives (classes are now balanced)
    seqs = [rec for rec in SeqIO.parse("negative.fasta", "fasta")]
    for i in range(len(seqs)):
        seqs[i].seq = seqs[i].seq.transcribe()
    random.shuffle(seqs)
    SeqIO.write(seqs[:num_train], "{}.train.negative.fasta".format(entry[2]), "fasta")
    SeqIO.write(seqs[num_train:(num_train+num_test)], "{}.test.negative.fasta".format(entry[2]), "fasta")

    # clean up
    os.remove("positive.bed")
    os.remove("negative.bed")
    os.remove("positive.fasta")
    os.remove("negative.fasta")

# clean up
os.remove("grch37.fasta")
os.remove("grch37.fasta.fai")