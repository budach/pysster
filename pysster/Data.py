import re
import numpy as np
from random import choice
from collections import defaultdict
from itertools import chain


import pysster.utils as io
from pysster.One_Hot_Encoder import One_Hot_Encoder
from pysster.Alphabet_Encoder import Alphabet_Encoder


class Data:
    """
    The Data class provides a convenient way to handle DNA and RNA sequence and structure data for 
    multiple classes. Sequence and structure data are automatically converted into one-hot
    encoded matrices and split into training/validation/test sets. The data object can then
    be passed to Grid_Search or Model objects for easy training and evaluation.
    """

    def __init__(self, class_files, alphabet):
        """ Load the sequences and split the data into 70%/15%/15% training/validation/test.

        If the goal is to do single-label classification a list of fasta files must be provided
        (one file per class, the first file will correspond to 'class_0'). In this case
        fasta headers are ignored. If the goal is multi-label classification a single fasta file
        must be provided and headers must indicate class membership as a comma-separated list
        (e.g. header '>0,2' means that the sequence belongs to class 0 and 2).

        For sequence-only files fasta entries have no format restrictions. For sequence-structure
        files each sequence and structure must span a single line, e.g.:

        '>0,2
        'CCCCAUAGGGG
        '((((...)))) (-3.3)
        'SSSSHHHSSSS

        This kind of format is the default output of RNAfold. The third line containing the
        annotated structure string can be omitted if you want to do the training on the dot-bracket
        strings (RNAfold will not output the annotated structure string, but we provide
        a helper function in the utils file to annotate an existing fasta file).
        **Important: All sequences in all files must have the same length.**

        The provided alphabet must match the content of the fasta files. For sequence-only files
        a single string ('ACGT' or 'ACGU') should be provided and for sequence-structure files a 
        tuple should be provided (('ACGU', 'HIMS') to use the annotated structures or ('ACGU', '().')
        to use dot-bracket structures). Characters that are not part of the provided alphabets will
        be randomly replaced with an alphabet character.

        Parameters
        ----------
        class_files: str or [str]
            A fasta file (multi-label) or a list of fasta files (single-label).
        
        alphabet: str or tuple(str,str)
            A string for sequence-only files and a tuple for sequence-structure files.
        """
        if isinstance(alphabet, tuple):
            self.is_rna = True
            self.alpha_coder = Alphabet_Encoder(alphabet[0], alphabet[1])
            alphabet = self.alpha_coder.alphabet
            data_loader = self._load_encode_rna
        else:
            self.is_rna = False
            data_loader = self._load_encode_dna
        if not isinstance(class_files, list):
            class_files = [class_files]
            self.multilabel = True
        else:
            self.multilabel = False
        self.one_hot_encoder = One_Hot_Encoder(alphabet)
        data_loader(class_files)
        self._process_labels()
        self.train_val_test_split(0.7, 0.15)


    def train_val_test_split(self, portion_train, portion_val, seed = None):
        """ Randomly split the data into training, validation and test set.

        Example: setting portion_train = 0.6 and portion_val = 0.3 will set aside 60% of the data
        for training, 30% for validation and the remaining 10% for testing. Use the seed parameter
        to get reproducible splits.
        
        Parameters
        ----------
        portion_train: float
            Portion of data that should be used for training (<1.0) 
        
        portion_val: float
            Portion of data that should be used for validation (<1.0)
        
        seed: int
            Seed for the random number generator.
        """
        if seed:
            np.random.seed(seed)
        num_sequences = len(self.data)
        break_train = int(num_sequences * portion_train)
        break_val = int(num_sequences * (portion_train + portion_val))
        splits = np.random.permutation(np.arange(num_sequences))
        splits = np.split(splits, [break_train, break_val])
        self.splits = {"train": splits[0], "val": splits[1], "test": splits[2]}


    def get_labels(self, group):
        """ Get the labels for a subset of the data.

        The 'group' argument can have the value 'train', 'val', 'test' or 'all'. The returned
        array has the shape (number of sequences, number of classes).

        Parameters
        ----------
        group : str
            A string indicating for which subset the labels should be returned.
        
        Returns
        -------
        labels : numpy.ndarray
            An array filled with 0s and 1s indicating class membership.
        """
        idx = self._get_idx(group)
        return np.array([self.labels[x] for x in idx])


    def get_summary(self):
        """ Get an overview of the training/validation/test data for each class.

        Returns
        -------
        summary : str
            A tabular overview of every class.
        """
        summary = ""
        class_ids = list(range(len(self.labels[0])))
        output = {}
        for group in ["train", "val", "test"]:
            idx = self._get_idx(group)
            labels = np.array([self.labels[x] for x in idx])
            output[group] = labels.sum(axis=0)
        output["all"] = output["train"] + output["val"] + output["test"]
        formatter = lambda xs: "  ".join("{:>9}".format(str(x)) for x in xs)
        summary += "            {}\n".format(formatter(["class_{}".format(x) for x in class_ids]))
        summary += "all data:   {}\n".format(formatter(output["all"]))
        summary += "training:   {}\n".format(formatter(output["train"]))
        summary += "validation: {}\n".format(formatter(output["val"]))
        summary += "test:       {}".format(formatter(output["test"]))
        return summary


    def _load_encode_dna(self, class_files):
        self.data, self.labels = [], []
        replacer = lambda x: choice(self.one_hot_encoder.alphabet)
        for class_id, file_name in enumerate(class_files):
            handle = io.get_handle(file_name, "rt")
            for header, sequence in io.parse_fasta(handle):
                sequence = re.sub(r"[NYMRWK]", replacer, sequence.upper())
                self.data.append(self.one_hot_encoder.encode(sequence))
                if self.multilabel:
                    self.labels.append(list(map(int, header.split(','))))
                else:
                    self.labels.append([class_id])
            handle.close()


    def _load_encode_rna(self, class_files):
        self.data, self.labels = [], []
        replacer_seq = lambda x: choice(self.alpha_coder.alph0)
        replacer_struct = lambda x: choice(self.alpha_coder.alph1)
        if self.alpha_coder.alph1 == "HIMS":
            idx = 2
        else:
            idx = 1
        for class_id, file_name in enumerate(class_files):
            handle = io.get_handle(file_name, "rt")
            for header, block in io.parse_fasta(handle, "_"):
                lines = block.split("_")
                sequence = re.sub(r"[NYMRWK]", replacer_seq, lines[0])
                structure = re.sub(r"[FT]", replacer_struct, lines[idx].split(" ")[0].upper())
                joined = self.alpha_coder.encode((sequence, structure))
                self.data.append(self.one_hot_encoder.encode(joined))
                if self.multilabel:
                    self.labels.append(list(map(int, header.split(','))))
                else:
                    self.labels.append([class_id])
            handle.close()


    def _process_labels(self):
        n_classes = max(max(entry) for entry in self.labels) + 1
        for x in range(len(self.labels)):
            label = np.zeros(n_classes, dtype=np.uint32)
            label[self.labels[x]] = 1
            self.labels[x] = label


    def _data_generator(self, group, batch_size, shuffle, labels = True, select = None, seed = None):
        idx = self._get_idx(group)
        if select is not None:
            idx = idx[select]
        while 1:
            if shuffle:
                np.random.seed(seed)
                np.random.shuffle(idx)
            for i in range(0, len(idx), batch_size):
                if labels:
                    yield (np.array([self.data[x] for x in idx[i:(i+batch_size)]]),
                           np.array([self.labels[x] for x in idx[i:(i+batch_size)]]))
                else:
                    yield np.array([self.data[x] for x in idx[i:(i+batch_size)]])


    def _get_data(self, group):
        idx = self._get_idx(group)
        return np.array([self.data[x]for x in idx]), np.array([self.labels[x] for x in idx])


    def _get_idx(self, group):
        if group == "all":
            return np.array(list(range(len(self.data))))
        return self.splits[group]


    def _shape(self):
        return self.data[0].shape


    def _get_class_weights(self):
        counts = sum(self.labels)
        counts = float(len(self.labels)) / counts
        counts = counts / counts.min()
        return {i: val for i, val in enumerate(counts)}



    def _get_sequences(self, class_id, group, select = None):
        idx = self._get_idx(group)
        labels = np.array([self.labels[x] for x in idx])
        idx = idx[np.nonzero(labels[:, class_id])[0]]
        sequences = []
        if select is None:
            select = range(len(idx))
        for x in select:
            sequences.append(self.one_hot_encoder.decode(self.data[idx[x]]))
        return sequences


