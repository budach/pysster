import os
import numpy as np
from subprocess import call
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from time import time
from pysster.Motif import Motif


def get_performance():
    predictions = []
    with open("pos.predictions", "rt") as handle:
        for line in handle:
            predictions.append(float(line.split()[2]))
    num_pos = len(predictions)
    with open("neg.predictions", "rt") as handle:
        for line in handle:
            predictions.append(float(line.split()[2]))
    num_neg = len(predictions) - num_pos
    predictions = np.array(predictions)
    labels = np.array([1]*num_pos + [-1]*num_neg)
    fpr, tpr, _ = roc_curve(labels, predictions)
    precision, recall, _ = precision_recall_curve(labels, predictions)
    return auc(fpr, tpr), average_precision_score(labels, predictions)


def plot_nicer_motifs():
    seqs = []
    with open("GraphProt.sequence_motif", "rt") as handle:
        for line in handle:
            seqs.append(line.strip())
    Motif("ACGU", seqs).plot().save("moitf_sequence.png")
    seqs = []
    with open("GraphProt.structure_motif", "rt") as handle:
        for line in handle:
            seqs.append(line.strip())
    colors = {'H':(212,0,0), 'I':(255,204,0), 'M':(68,170,0), 'S':(204,0,255)}
    Motif("HIMSE", seqs).plot(colors).save("motif_structure.png")


def main():
    RBPs = [("../data/pum2.train.positive.fasta",
             "../data/pum2.train.negative.fasta",
             "../data/pum2.test.positive.fasta",
             "../data/pum2.test.negative.fasta",
             "PUM2"),
            ("../data/qki.train.positive.fasta",
             "../data/qki.train.negative.fasta",
             "../data/qki.test.positive.fasta",
             "../data/qki.test.negative.fasta",
             "QKI")]
    
    for entry in RBPs:
        output_folder = entry[4] + "_graphprot/"
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        os.chdir(output_folder)

        start = time()
        # train and predict, GraphProt predicts secondary structures internally
        call("GraphProt.pl -mode classification -action train -fasta {} -negfasta {}".format(entry[0], entry[1]), shell = True)
        call("GraphProt.pl -action predict -fasta {} -model GraphProt.model -prefix pos".format(entry[2]), shell = True)
        call("GraphProt.pl -action predict -fasta {} -model GraphProt.model -prefix neg".format(entry[3]), shell = True)
        stop = time()

        with open("time.txt", "wt") as handle:
            handle.write("{}, time in seconds: {}".format(entry[4], stop-start))
        # create motifs
        call("GraphProt.pl -action motif -model GraphProt.model -fasta {} -motif_len 8".format(entry[2]), shell = True)
        plot_nicer_motifs()
        with open("performance.txt", "wt") as handle:
            roc, prec = get_performance()
            handle.write("{}, {}".format(roc, prec))
        os.chdir("..")


if __name__ == "__main__":
    main()