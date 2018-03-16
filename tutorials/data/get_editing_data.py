# the files alu.fa.gz, rep.fa.gz and nonrep.fa.gz can be created using this script.
# it will download the reference genome, editing sites and then randomly sample 50k sequences
# of length 300 per class and predict their secondary structures.
#
# !! bedtools must be installed for this script to work !!


from urllib.request import urlretrieve
from subprocess import call
from Bio import SeqIO
import gzip
import shutil
import os
import random
from pysster import utils


print("download human genome...")
urlretrieve("ftp://ftp.sanger.ac.uk/pub/gencode/Gencode_human/release_19/GRCh37.p13.genome.fa.gz", "grch37.fasta.gz")
with gzip.open("grch37.fasta.gz", "rb") as f_in:
    with open("grch37.fasta", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
os.remove("grch37.fasta.gz")


print("download editing sites...")
urlretrieve("http://srv00.recas.ba.infn.it/webshare/rediportalDownload/table1_full.txt.gz", "table1_full.txt.gz")


print("split into bed files...")
h_alu = gzip.open("alu.bed.gz", "wt")
h_rep = gzip.open("rep.bed.gz", "wt")
h_non = gzip.open("nonrep.bed.gz", "wt")
with gzip.open("table1_full.txt.gz", "rt") as handle:
    for line in handle:
        line = line.split()
        mid = int(line[1])
        if (mid-151) < 0 or line[2] == "C" or line[2] == "G": continue
        out = "{}\t{}\t{}\t.\t.\t{}\n".format(line[0], mid-151, mid+150, line[4])
        if line[6] == "ALU":
            h_alu.write(out)
        elif line[6] == "REP":
            h_rep.write(out)
        elif line[6] == "NONREP":
            h_non.write(out)
h_alu.close()
h_rep.close()
h_non.close()


print("get sequences as fasta...")
call("bedtools getfasta -s -fo alu.fasta -fi grch37.fasta -bed alu.bed.gz", shell = True)
call("bedtools getfasta -s -fo rep.fasta -fi grch37.fasta -bed rep.bed.gz", shell = True)
call("bedtools getfasta -s -fo nonrep.fasta -fi grch37.fasta -bed nonrep.bed.gz", shell = True)


print('randomly sample balanced classes...')
# there is a very small number of sequences without an 'A' in the center, error in editing database?
alu = [rec for rec in SeqIO.parse("alu.fasta", "fasta") if rec.seq[150] == "A"]
for i in range(len(alu)):
    alu[i].seq = alu[i].seq.transcribe()
rep = [rec for rec in SeqIO.parse("rep.fasta", "fasta") if rec.seq[150] == "A"]
for i in range(len(rep)):
    rep[i].seq = rep[i].seq.transcribe()
nonrep = [rec for rec in SeqIO.parse("nonrep.fasta", "fasta") if rec.seq[150] == "A"]
for i in range(len(nonrep)):
    nonrep[i].seq = nonrep[i].seq.transcribe()
random.seed(42)
SeqIO.write(random.sample(alu, 50000), "alu.fasta", "fasta")
SeqIO.write(random.sample(rep, 50000), "rep.fasta", "fasta")
SeqIO.write(random.sample(nonrep, 50000), "nonrep.fasta", "fasta")


print("predict secondary structures...")
utils.predict_structures("alu.fasta", "alu.fa.gz", 15, True)
utils.predict_structures("rep.fasta", "rep.fa.gz", 15, True)
utils.predict_structures("nonrep.fasta", "nonrep.fa.gz", 15, True)


print("clean up...")
os.remove("grch37.fasta")
os.remove("grch37.fasta.fai")
os.remove("rep.bed.gz")
os.remove("alu.fasta")
os.remove("rep.fasta")
os.remove("nonrep.fasta")
os.remove("alu.bed.gz")
os.remove("nonrep.bed.gz")
os.remove("rep.bed.gz")
os.remove("table1_full.txt.gz")
