import re
import numpy as np
from random import choice
from collections import OrderedDict


import pysster.utils as io
from pysster.One_Hot_Encoder import One_Hot_Encoder
from pysster.Alphabet_Encoder import Alphabet_Encoder


class Data:
    """
    The Data class provides a convenient way to handle biological sequence and structure data for 
    multiple classes. Sequence and structure data are automatically converted into one-hot
    encoded matrices and split into training/validation/test sets. The data object can then
    be passed to Grid_Search or Model objects for easy training and evaluation.

    Input format: Data objects accept raw strings in fasta format as input for sequence and structure
    data or optionally position-weight matrices for structure data (see __init__ function). Strings
    can contain all uppercase alphanumeric characters and the following special characters: "()[]{}<>,.|*".
    Additional handcrafted features may be added using the load_additional_data function.
    """

    def __init__(self, class_files, alphabet, structure_pwm=False):
        """ Load the sequences and split the data into 70%/15%/15% training/validation/test.

        If the goal is to do single-label classification a list of fasta files must be provided
        (one file per class, the first file will correspond to 'class_0'). In this case
        fasta headers are ignored. If the goal is multi-label classification a single fasta file
        must be provided and headers must indicate class membership as a comma-separated list
        (e.g. header '>0,2' means that the entry belongs to class 0 and 2).

        For sequence-only files fasta entries have no format restrictions. For sequence-structure
        files each sequence and structure must span a single line, e.g.:

        '>header
        'CCCCAUAGGGG
        '((((...))))

        in which the second line contains the sequence and the third line the structure.
        **Important: All sequences in all files must have the same length.**

        The provided alphabet must match the content of the fasta files. For sequence-only files
        a single string (e.g. 'ACGT' or 'ACGU') should be provided and for sequence-structure files a 
        tuple should be provided (e.g. ('ACGU', '().')). Characters that are not part of the 
        provided alphabets will be randomly replaced with an alphabet character.

        We support all uppercase alphanumeric characters and the following additional characters
        for alphabets: "()[]{}<>,.|*". Thus, it is possible to use and combine (in the sequence-structure
        case) arbitrarily defined alphabets as long as the data is provided in the described fasta format.
        In particular, this means the usage of the package is not restricted to RNA secondary
        structure (this is only an example). If you have structure information for DNA or protein data
        that can be encoded by some alphabet, similar to RNA structure information, you can apply
        the package to this kind of data as well.

        If you don't want to work with a single minimum free energy structure (as some RNA structure
        prediction tools can output multiple predictions) you can also provide a position-weight
        matrix representing the structure instead of a single string (matrix entries must be
        separated by a space or tab):

        '>header
        'GGGGUUCCCC
        '0.9 0.8 0.7 0.9 0.0 0.0 0.0 0.0 0.0 0.0
        '0.0 0.0 0.0 0.0 0.0 0.0 0.2 0.7 0.8 0.9
        '0.1 0.2 0.3 0.1 1.0 1.0 0.8 0.3 0.2 0.1

        If you provide "()." as the alphabet the first line of the matrix given above will correspond to
        "(", the second to ")" and the third to ".". Each column of the matrix must add up to 1. Again,
        we don't restrict the usage of the package to RNA, therefore the matrix given above can represent
        whatever you want it to represent, as long as you provide a valid alphabet.

        Parameters
        ----------
        class_files: str or [str]
            A fasta file (multi-label) or a list of fasta files (single-label).
        
        alphabet: str or tuple(str,str)
            A string for sequence-only files and a tuple for sequence-structure files.
        
        structure_pwm: bool
            Are structures provided as single strings (False) or as PWMs (True)?
        """
        self.meta = OrderedDict()
        self.is_rna_pwm = False
        if isinstance(alphabet, tuple):
            self.is_rna = True
            self.is_rna_pwm = structure_pwm
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
        # check if all sequences have the same length
        length = self.data[0].shape[0]
        for x in range(1, len(self.data)):
            if length != self.data[x].shape[0]:
                raise RuntimeError('All sequences must have the same length.')
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


    def load_additional_data(self, class_files, is_categorical=False, categories=None, standardize=False):
        """ Add additional numerical or categorical features to the network (for each sequence as a whole).

        For every input sequence additional data can be added to the network (e.g. location,
        average sequence conservation, etc.). The data will be concatenated to the input of the
        first dense layer (i.e. additional neurons in the first dense layer will be created). Input
        files are text files and must contain one value per line, e.g.:
        
        '0.679
        '0.961
        '0.065
        '0.871
        '...
        
        The number of provided files must match the fasta files provided to the __init__
        function (e.g. if you provided a list of 3 files to __init__ you must provide a list
        of 3 files here as well) and the number of lines in each file must match the number of
        entries in the corresponding fasta file. If you want to add multiple features simply
        call this function multiple times.

        Interpreting the influence of arbitrary additional data for a neural network is hard and at
        the moment we don't provide any means to do so. You should run your model with and without the
        additional data and check if the predictive performance improves. In general, if you have
        many handcrafted features you might want to consider using a different machine learning
        technique.

        Parameters
        ----------
        class_files: str or [str]
            A text file (multi-label) or a list of text files (single-label).
        
        is_categorical: bool
            Is the provided data categorical or numerical?

        categories: [str]
            A list containing all possible categories (only needed if is_categorial == True).

        standardize: bool
            Should the z-score be computed for numerical data?
        """
        if not isinstance(class_files, list):
            class_files = [class_files]
        # load raw data
        idx = len(self.meta)
        self.meta[idx] = {"data":[], "is_categorical": is_categorical}
        for _class_id, file_name in enumerate(class_files):
            handle = io.get_handle(file_name, "rt")
            if True == is_categorical:
                for line in handle:
                    self.meta[idx]['data'].append(line.strip())
            else:
                for line in handle:
                    self.meta[idx]['data'].append(float(line))
            handle.close()
        if len(self.labels) != len(self.meta[idx]['data']):
            raise RuntimeError("Number of additional data ({}) doesn't match number of main data ({}).".format(
                len(self.meta[idx]['data']), len(self.labels)
            ))
        # one hot encode categorical data
        if True == is_categorical:
            if not isinstance(categories, list):
                raise RuntimeError("is_categorical set to True, but no categories list provided.")
            categories = sorted(categories)
            mapping = {val: i for i, val in enumerate(categories)}
            for i, _val in enumerate(self.meta[idx]['data']):
                one_hot = np.zeros(len(categories), dtype=np.uint8)
                one_hot[mapping[self.meta[idx]['data'][i]]] = 1
                self.meta[idx]['data'][i] = one_hot
        # standardize numerical data if desired
        else:
            if True == standardize:
                from scipy import stats
                self.meta[idx]['data'] = stats.zscore(self.meta[idx]['data'])


    def load_additional_positionwise_data(self, class_files, identifier, standardize=False):
        """ Add additional numerical features to the network (for each nucleotide in a sequence).

        For every position in an input sequence additional numerical data can be added to
        the network (e.g. ChIP-seq signal, conservation for every nucleotide).
        The data will be added to the input matrix. E.g.: Using sequences of length 200
        over the alphabet "ACGT" results in input matrices of size 4x200. Additional position-wise
        data will be added to these matrices as a new row resulting in matrices of size 5x200.
        
        Input files are text files and must contain as many whitespace-separated values 
        in each line as the sequences are long, e.g.:
        
        '0.679 1.223 -0.296  ...
        '0.961 0.532 0.112   ...
        '0.065 -0.333 -0.256 ...
        '...
        
        The number of provided files must match the fasta files provided to the __init__
        function (e.g. if you provided a list of 3 files to __init__ you must provide a list
        of 3 files here as well) and the number of lines in each file must match the number of
        entries in the corresponding fasta file. If you want to add multiple features simply
        call this function multiple times.

        Input features should be standardized in some way prior to adding them to the
        network, as this tends to improve the predictive performance.

        In the same way network kernels are visualized as sequence motifs after the network
        training (based on the first 4 rows of the input matrices and using the visualize_kernel()
        Model function), the rows corresponding to additional features are summarized
        as line plots as well.

        Parameters
        ----------
        class_files: str or [str]
            A text file (multi-label) or a list of text files (single-label).
        
        identifier: str
            A short feature name (will be shown in kernel output plots).

        standardize: bool
            Scale each column according to the interquartile range.
        """
        if not "positionwise" in dir(self):
            self.positionwise = OrderedDict()
        if identifier in self.positionwise:
            raise RuntimeError("Identifier '{}' already exists.".format(identifier))
        if not isinstance(class_files, list):
            class_files = [class_files]
        len_sequence = self.data[0].shape[0]
        
        new_data = np.empty((len(self.labels), len_sequence), dtype=np.float32)
        row = 0
        for file_name in class_files:
            handle = io.get_handle(file_name, 'rt')
            for i, line in enumerate(handle):
                try:
                    new_data[row,:] = [float(x) for x in line.split()]
                except ValueError as err:
                    raise RuntimeError("ValueError: {} (in line {} in {}).".format(
                        err, i+1, file_name
                    ))
                row += 1
            handle.close()
        if row != len(self.labels):
            raise RuntimeError("Amount of additional data ({}) doesn't match number of sequences ({}).".format(
                row, len(self.labels)
            ))
        if True == standardize:
            from sklearn.preprocessing import robust_scale
            self.positionwise[identifier] = robust_scale(new_data, axis=0)
            if not "positionwise_unscaled" in dir(self):
                self.positionwise_unscaled = OrderedDict()
            self.positionwise_unscaled[identifier] = new_data
        else:
            self.positionwise[identifier] = new_data



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
                sequence = re.sub(r"[^{}]".format(self.one_hot_encoder.alphabet),
                                  replacer, sequence.upper())
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
        pattern_seq = r"[^{}]".format(re.escape(self.alpha_coder.alph0))
        pattern_struct = r"[^{}]".format(re.escape(self.alpha_coder.alph1))
        for class_id, file_name in enumerate(class_files):
            handle = io.get_handle(file_name, "rt")
            for header, block in io.parse_fasta(handle, "_"):
                lines = block.split("_")
                sequence = re.sub(pattern_seq, replacer_seq, lines[0].upper())
                if True == self.is_rna_pwm:
                    pwm = np.zeros((len(sequence), len(self.alpha_coder.alph1)), dtype=np.float32)
                    for x in range(1, pwm.shape[1]+1):
                        pwm[:, x-1] = list(map(float, lines[x].split()))
                    self.data.append(self._join_seq_pwm(sequence, pwm))
                else:
                    structure = re.sub(pattern_struct, replacer_struct, lines[1].split(" ")[0].upper())
                    joined = self.alpha_coder.encode((sequence, structure))
                    self.data.append(self.one_hot_encoder.encode(joined))
                if self.multilabel:
                    self.labels.append(list(map(int, header.split(','))))
                else:
                    self.labels.append([class_id])
            handle.close()


    def _join_seq_pwm(self, sequence, pwm):
        joined = np.zeros((len(sequence), len(self.alpha_coder.alphabet)), np.float32)
        for i, symbol in enumerate(sequence):
            pos = self.alpha_coder.alph0.find(symbol) * len(self.alpha_coder.alph1)
            joined[i, pos:(pos+len(self.alpha_coder.alph1))] = pwm[i,:]
        return joined


    def _process_labels(self):
        n_classes = max(max(entry) for entry in self.labels) + 1
        for x in range(len(self.labels)):
            label = np.zeros(n_classes, dtype=np.uint32)
            label[self.labels[x]] = 1
            self.labels[x] = label


    def _data_generator(self, group, batch_size, shuffle, labels=True, select=None, seed=None, meta=True):
        idx = self._get_idx(group)
        if select is not None:
            idx = idx[select]
        if "positionwise" in dir(self) and len(self.positionwise) > 0:
            use_positionwise = True
        else:
            use_positionwise = False
        while 1:
            if shuffle:
                np.random.seed(seed)
                np.random.shuffle(idx)
            for i in range(0, len(idx), batch_size):
                if use_positionwise:
                    out_data = self._get_positionwise_data(idx, i, batch_size)
                    if meta == True and len(self.meta) > 0:
                        out_data = [out_data, self._get_additional_data(idx, i, batch_size)]
                else:
                    out_data = np.array([self.data[x] for x in idx[i:(i+batch_size)]])
                    if meta == True and len(self.meta) > 0:
                        out_data = [out_data, self._get_additional_data(idx, i, batch_size)]
                if labels:
                    out_labels = np.array([self.labels[x] for x in idx[i:(i+batch_size)]])
                    yield (out_data, out_labels) 
                else:
                    yield out_data


    def _data_gen_no_labels_meta(self, group, batch_size, select):
        idx = self._get_idx(group)[select]
        while 1:
            if "positionwise" in dir(self) and len(self.positionwise) > 0:
                for i in range(0, len(idx), batch_size):
                    yield self._get_positionwise_data(idx, i, batch_size)
            else:
                for i in range(0, len(idx), batch_size):
                    yield np.array([self.data[x] for x in idx[i:(i+batch_size)]])


    def _get_positionwise_data(self, idx, i, batch_size):
        shape_data = self._shape()
        dim_seq = self.data[0].shape[1]
        result = []
        for x in idx[i:(i+batch_size)]:
            data = np.empty(shape_data, dtype=np.float32)
            data[:,:dim_seq] = self.data[x]
            for i, identifier in enumerate(self.positionwise):
                data[:,dim_seq+i] = self.positionwise[identifier][x]
            result.append(data)
        return np.array(result)


    def _get_additional_data(self, idx, i, batch_size):
        result = []
        for x in idx[i:(i+batch_size)]:
            additional = np.empty((0,), dtype=np.float32)
            for idx_add in range(len(self.meta)):
                additional = np.append(additional, self.meta[idx_add]['data'][x])
            result.append(additional)
        return np.array(result)


    def _get_data(self, group):
        idx = self._get_idx(group)
        return np.array([self.data[x] for x in idx]), np.array([self.labels[x] for x in idx])


    def _get_idx(self, group):
        if group == "all":
            return np.array(list(range(len(self.data))))
        return self.splits[group]


    def _shape(self):
        # to be backward compatible
        if 'positionwise' in dir(self):
            return (self.data[0].shape[0], self.data[0].shape[1] + len(self.positionwise))
        return self.data[0].shape


    def _get_class_weights(self):
        counts = sum(self.labels)
        counts = float(len(self.labels)) / counts
        counts = counts / counts.min()
        counts = {i: val for i, val in enumerate(counts)}
        if len(counts) == 2 and counts[0] == counts[1]:
            counts[0] = 1.5
        return counts


    def _get_sequences(self, class_id, group, select = None):
        idx = self._get_idx(group)
        labels = np.array([self.labels[x] for x in idx])
        idx = idx[np.nonzero(labels[:, class_id])[0]]
        sequences = []
        if select is None:
            select = range(len(idx))
        if True == self.is_rna_pwm:
            for x in select:
                sequences.append(self.data[idx[x]])
        else:
            for x in select:
                sequences.append(self.one_hot_encoder.decode(self.data[idx[x]]))
        return sequences


    def _get_positionwise_for_plots(self, class_id, group, select = None):
        idx = self._get_idx(group)
        labels = np.array([self.labels[x] for x in idx])
        idx = idx[np.nonzero(labels[:, class_id])[0]]
        if select is None:
            select = range(len(idx))
        data = []
        for identifier in self.positionwise:
            if "positionwise_unscaled" in dir(self) and identifier in self.positionwise_unscaled:
                source = self.positionwise_unscaled[identifier]
            else:
                source = self.positionwise[identifier]
            feature = []
            for x in select:
                feature.append(source[idx[x],:])
            data.append(feature)
        return data

