import numpy as np
import re
import heapq
import string
import random
from os import remove
from copy import deepcopy
from tempfile import gettempdir
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.models import Model as KModel
from keras.layers import Dropout, Conv1D, MaxPooling1D, Flatten, Dense
from keras.constraints import max_norm
from keras.optimizers import Adam
from keras.initializers import RandomUniform, Constant
from keras.utils import multi_gpu_model
import keras.activations
from collections import OrderedDict


import pysster.utils as utils
from pysster.Motif import Motif


class Model:
    """
    The Model class represents a convolutional neural network and provides functions for
    network training and visualization of learned features (sequence/structure motifs). The
    basic architecture of the network consists of a variable number of convolutional and max pooling
    layers followed by a variable number of dense layers. These layers are interspersed by dropout
    layers after the input layer and after every max pooling and dense layer.
    
    The network uses the Adam optimizer. In case of a single-label classification a softmax
    activation is used for the output layer together with a categorical crossentropy loss. In case
    of a multi-label classification a sigmoid activation and a binary crossentropy loss is used.
    
    The network can be tuned using the following hyperparameters which can be provided through
    the 'params' parameter of the __init__ function:

    '| parameter         | default | description |
    '|:-                 |:-       |:-           |
    '| conv_num          | 2       | number of convolutional/pooling layers |
    '| kernel_num        | 50      | number of kernels in each conv layer |
    '| kernel_len        | 25      | length of kernels |
    '| pool_size         | 2       | size of pooling windows |
    '| pool_stride       | 2       | step size of pooling operation |
    '| dense_num         | 1       | number of dense layers |
    '| neuron_num        | 100     | number of neurons in each dense layer |
    '| dropout_input     | 0.1     | dropout portion after input |
    '| dropout_conv      | 0.3     | dropout portion after pooling layers |
    '| dropout_dense     | 0.6     | dropout portion after dense layers |
    '| batch_size        | 256     | batch size during training |
    '| learning_rate     | 0.0005  | learning rate of Adam optimizer |
    '| patience_lr       | 5       | number of epochs without validation loss improvement before halving learning rate |
    '| patience_stopping | 15      | number of epochs without validation loss improvement before stopping training |
    '| epochs            | 500     | maximum number of training epochs |
    '| kernel_constraint | 3       | max-norm weight constraint |

    Not all parameters are equally important when doing a hyperparameter grid search. The ones
    with a strong influence are usually conv_num (range 1-3), kernel_num (range 50-300), 
    neuron_num (50-1000) and the dropout parameters (around 0.1 for the input and 0.2-0.6 otherwise).

    Note: with each convolutional/pooling stack the length of your sequences will be reduced.
    E.g. starting with sequences of length 300 and kernels of length 25  will result in sequences
    of length 300-25+1=276 after the first convolutional layer. A default pooling layer will halve
    this number further to 138. If you use too many convolutional/pooling stacks you will get an
    error, because your sequence length will be <= 0.
    """

    def __init__(self, params, data, seed = None):
        """ Initialize the model with the given parameters.

        Example: providing the params dict {'conv_num': 1, 'kernel_num': 20, 'dropout_input': 0.0}
        will set these 3 parameters to the provided values. All other parameters will 
        have default values. A data object must be provided to infer the input shape and
        number of classes.

        Parameters
        ----------
        params : dict
            A dict containing hyperparameter values.
        
        data : pysster.Data
            The Data object the model should be trained on.
        
        seed : int
            Seed for the random initialization of network weights.
        """
        self.params = deepcopy(params)
        if data != None:
            self.params["input_shape"] = data._shape()
            self.params["class_num"] = len(data.labels[0])
            if data.multilabel == True:
                self.params["activation"] = "sigmoid"
                self.params["loss"] = "binary_crossentropy"
        if seed != None:
            self.params['seed'] = seed
        self.temp_file = "{}/{}.hdf5".format(
            gettempdir(),
            ''.join(random.choice(string.ascii_uppercase) for _ in range(15))
        )
        self._check_params()
        self._prepare_callbacks()
        self._prepare_model()


    def print_summary(self):
        """ Print an overview of the network architecture.
        """
        self.model.summary()


    def train(self, data, verbose = True):
        """ Train the model.

        The model will be trained and validated on the training and validation set provided
        by the Data object.

        Parameters
        ----------
        data : pysster.Data
            The Data object the model should be trained on.
        
        verbose : bool
            If True, progress information will be printed throughout the training. 
        """
        np.random.seed(self.params["seed"])
        random.seed(self.params["seed"])
        n_train = len(data._get_idx('train'))
        n_train = n_train//self.params['batch_size'] + (n_train%self.params['batch_size'] != 0)
        n_val = len(data._get_idx('val'))
        n_val = n_val//self.params['batch_size'] + (n_val%self.params['batch_size'] != 0)
        self.model.fit_generator(generator = data._data_generator('train',self.params['batch_size'],
                                                                  True, seed=self.params["seed"]),
                                 steps_per_epoch = n_train,
                                 epochs = self.params['epochs'],
                                 callbacks = self.callbacks,
                                 verbose = verbose,
                                 validation_data = data._data_generator('val',self.params['batch_size'],
                                                                        True, seed=self.params["seed"]),
                                 validation_steps = n_val,
                                 class_weight = data._get_class_weights())
        self.model = load_model(self.temp_file)
        remove(self.temp_file)


    def predict(self, data, group):
        """ Get model predictions for a subset of a Data object.

        The 'group' argument can have the value 'train', 'val', 'test' or 'all'. The returned
        array has the shape (number of sequences, number of classes) and contains
        predicted probabilities.

        Parameters
        ----------
        data : pysster.Data 
            A Data object.

        group : str
            The subset of the Data object that should be used for prediction.
        
        Returns
        -------
        predictions : numpy.ndarray
            An array containing predicted probabilities.
        """
        data_gen = data._data_generator(group, self.params['batch_size'], False, False)
        idx = data._get_idx(group)
        n = max(len(idx)//self.params['batch_size'] + (len(idx)%self.params['batch_size'] != 0), 1)
        return self.model.predict_generator(data_gen, n)


    def get_max_activations(self, data, group, index = 1):
        """ Get the network output of the first convolutional layer.

        The function returns the maximum activation (the maximum output of a kernel) for
        every kernel - input sequence pair. The return value is a dict containing the
        entries 'activations' (an array of shape (number of sequences, number of kernels)), 'labels'
        (an array of shape (number of sequences, number of classes)) and 'group' (the
        subset of the Data object used). 

        The 'group' argument can have the value 'train', 'val', 'test' or 'all'.

        Parameters
        ----------
        data : pysster.Data 
            A Data object.
        
        group : str
            The subset of the Data object that should be used.
        
        index : int
            The index of the network layer for which the output should be returned.
        
        Returns
        -------
        results : dict
            A dict with 3 values ('activations', 'labels, 'group', see above)
        """
        tmp_model = KModel(self.model.input, self.model.layers[index].output)
        data_gen = data._data_generator(group, self.params['batch_size'], False, False)
        idx = data._get_idx(group)
        n = max(len(idx)//self.params['batch_size'] + (len(idx)%self.params['batch_size'] != 0), 1)
        activations = []
        for _ in range(n):
            tmp = tmp_model.predict_on_batch(next(data_gen))
            activations.append(tmp.max(axis=1))
        return {'activations': np.vstack(activations), 
                'labels': np.array([data.labels[x] for x in idx]),
                'group': group}


    def visualize_kernel(self, activations, data, kernel, folder):
        """ Get a number of visualizations and an importane score for a convolutional kernel.

        This function creates three output files: 1) a sequence(/structure) motif that the
        kernel has learned to detect, 2) a histogram/mean activation plot showing the 
        positional enrichment of said motif for every class and 3) violin plots showing 
        the maximum activation distributions for every class (higher values == better, 
        this is a proxy for general class enrichment).

        The output files are named "motif_kernel_x.png", "position_kernel_x.png" and
        "activations_kernel_x.png"

        How it works:
        Given an input sequence, a first layer kernel produces an output vector (called activations)
        of length sequence_length - kernel_length + 1. The position of the maximum activation can
        therefore be directly mapped back to the input sequence and a subsequence of the length
        of the kernel can be extracted from the input sequence. Applying this approach to every
        input sequence yields a number of subsequences that can be used for the construction
        of a motif. Subsequences are only considered if the maximum activation exceeds a certain
        threshold, in this case the maximum of the mean maximum activations per class. Only 
        subsequences from the top class are used to construct the motif (up to 500 subsequences).

        The histograms show the positions of the maximum activation, i.e. the positions the
        subsequences were extracted from. The mean activation plots show the mean activation
        for all sequence positions. Both plots are only based on sequences that led to a maximum
        activation higher than the threshold. Histogram and mean activation plot are usually
        identical, but in case the histogram is very sparse the mean activation plot might
        be easier to look at.

        The violin plots show how the maximum activation values are distributed for each class,
        indicating class enrichment.

        The function returns a Motif object (or a tuple of Motif objects for RNA sequence/structure
        motifs) and an importance score that indicates how important this kernel was for the
        classification (higher values == more important). The score is computed as maximum of the
        mean maximum activations per class minus minimum of the mean maximum activations per class.
        The idea is that kernels that show a big differences across classes (i.e. kernels that are
        strongly enriched in some classes and little to none in other classes) are more important
        for the network to deliver correct predictions.

        Parameters
        ----------
        activations : dict
            The return value of the get_max_activations function.
        
        data: pysster.Data 
            The Data object that was used to compute the maximum activations.
        
        kernel: int
            The kernel that should be visualized (first kernel is 0)
        
        folder: str
            A valid folder path. Plots will be saved here.
        
        Returns
        -------
        results: tuple(pysster.Motif, float) or tuple((tuple(pysster.Motif, pysster.Motif), float)
            A Motif object (or a tuple of Motifs for sequence/structure motifs) and the importance score.
        """
        # this function is kind of messy to keep the memory usage low.
        # i am avoiding to compute the complete first conv layer output because
        # that would be a matrix of shape (num of sequences, length of sequences, num of kernels)
        # therefore I have to do lots of index mapping...
        if folder[-1] != "/":
            folder += "/"
        # get activation threshold (the max average activation per class)
        n_classes = activations['labels'].shape[1]
        max_per_class = []
        for class_id in range(n_classes):
            max_per_class.append(activations['activations'][activations['labels'][:,class_id] == 1, kernel])
        mean_max = [np.mean(x) for x in max_per_class]
        threshold, thresh_class = max(mean_max), np.argmax(mean_max)
        # collect the positions of the max activations per class (histograms)
        # and the average activation of every position per class (mean_acts)
        histograms, mean_acts = [], []
        idx_above_thresh = np.where( (activations['activations'][:, kernel] > threshold) | 
                                      np.isclose(activations['activations'][:, kernel], threshold) )[0]
        acts = self._get_activations_idx_kernel(data, idx_above_thresh, activations['group'], kernel)
        for class_id in range(n_classes):
            idx_labels = np.where(activations['labels'][:,class_id] == 1)[0]
            idx_class_seq = np.intersect1d(idx_labels, idx_above_thresh, True)
            # histogram and mean activations
            idx_class = np.in1d(idx_above_thresh, idx_class_seq)
            if idx_class.sum() > 0:
                histograms.append(np.argmax(acts[idx_class], axis=1))
                mean_acts.append(np.mean(acts[idx_class], axis=0))
            else:
                histograms.append([])
                mean_acts.append([])
            # get the sequence logo from sequences from the threshold class (max 500 sequences)
            if class_id == thresh_class:
                select = np.in1d(idx_labels, idx_class_seq).nonzero()[0]
                if len(select) > 500:
                    value500 = heapq.nlargest(500, max_per_class[class_id][select])[-1]
                    select_seqs = np.where(max_per_class[class_id] >= value500)[0]
                    select = np.in1d(select, select_seqs)
                    sequences = data._get_sequences(class_id, activations["group"], select_seqs)
                    logo = self._plot_motif(data, self._get_subseq(sequences, histograms[-1][select]))
                else:
                    sequences = data._get_sequences(class_id, activations["group"], select)
                    logo = self._plot_motif(data, self._get_subseq(sequences, histograms[-1]))
        # plot everything
        utils.plot_motif(logo, "{}motif_kernel_{}.png".format(folder, kernel))
        utils.plot_motif_summary(histograms, mean_acts, kernel, "{}position_kernel_{}.png".format(folder, kernel))
        utils.plot_violins(max_per_class, kernel, "{}activations_kernel_{}.png".format(folder, kernel))
        return logo, max(mean_max) - min(mean_max)



    def plot_clustering(self, activations, output_file, classes = None):
        """ Perform a hierarchical clustering on both sequences and kernels.

        Given the maximum activations for each sequence - kernel pair (the output of the
        get_max_activations() method) a hierarchical clustering using Ward's method and 
        the Euclidean distance is performed. Values are standardized before clustering. To compute
        the clustering only for a subset of classes (often it looks quite messy for all classes) you
        can provide a list of integers through the 'classes' argument (e.g. [0, 3] to only plot
        sequences belonging to class 0 and 3). By default all sequences of all classes are used.
        Clustering is only possible for single-label classifications.

        Parameters
        ----------
        activations : dict
            A dict with keys 'activations' and 'labels' (the return value of get_max_activations()).
        
        output_file : str
            Path of the PNG output file.
        
        classes : [int]
            List of integers indicating which classes should be clustered (default: all).
        """
        if (activations['labels'].sum(axis=1) > 1).any():
            print("Warning: Clustering for multi-label data not supported. Plot was not created.")
            return
        if classes == None:
            utils._plot_heatmap(output_file,
                                      activations['activations'],
                                      np.argmax(activations['labels'], axis=1))
        else:
            if not isinstance(classes, list):
                raise ValueError("'classes' must be a list of integers.")
            labels = activations["labels"][:, classes]
            labels.shape = (labels.shape[0], len(classes))
            select = np.where(labels.sum(axis=1))[0]
            utils._plot_heatmap(output_file,
                                      activations['activations'][select,:],
                                      np.argmax(labels[select,:], axis=1), classes)




    def visualize_optimized_inputs(self, data, layer_name, output_file, bound=0.1, lr=0.02, steps=600, nodes=None):
        """ Visualize what every node in the network has learned.

        Given fixed network parameters it is possible to visualize what individual nodes (e.g.
        kernels in conv layers and neurons in dense layers) have learned during model training by
        specifically maximizing the output of these nodes with respect to an input sequence
        (starting with a random PWM of the length of an input sequence). In brief: this function
        learns a single input sequence (in the form of a PWM) that maximizes the output of a specific
        network node using a l2-norm penalized gradient ascent optimization.

        Warning: This kind of visualization has been applied before to image classification networks
        and while the resulting images are usually somewhat recognizable they are still very hard
        to interpret (e.g. https://distill.pub/2017/feature-visualization/).
        For a PWM to be useful it has to be very precise, but this is unfortunately not the case
        for many data sets and results are very messy, especially for RNA secondary structure motifs.
        Therefore this function should not be considered for any biological interpretations.
        Please use the visualize_kernel() method for more reliable visualizations. Nevertheless,
        visualization of all layers of a network can be interesting if you are interested
        in how the neural networks works per se.

        If needed, the bound, lr and steps parameters can be used to tune the information content of the PWM
        and the convergence of the optimization (higher values == higher information content).

        Each row in the output file corresponds to a node of the layer.

        Parameters
        ----------
        data : pysster.Data
            The Data object used to train the model.
        
        layer_name : str
            Name of the network layer that should be optimized (see print_summary())
        
        output_file : str
            Path of the PNG output file.

        bound : float
            A float > 0. The PWM will be initialized by drawing from a uniform distribution with lower and upper bounds - and + bound.
        
        lr : float
            A float > 0. Learning rate of the gradient ascent optimization.
        
        steps : int
            An int > 0. Number of optimization iterations.
        
        nodes : [int]
            List of integers indicating which nodes of the layer should be optimized (default: all).
        """
        if nodes == None:
            nodes = list(range(self.model.get_layer(layer_name).output_shape[-1]))
        if layer_name == self.model.layers[-1].name:
            model = self._change_activation()
        else:
            model = self.model
        motif_plots = []
        for node in nodes:
            print("Optimize node {}...".format(node))
            motif_plots += self._get_optimized_input(model, data, layer_name, node, bound, lr, steps)
        utils.combine_images(motif_plots, output_file)



    def _check_params(self):
        default_params = {'conv_num': 2, 'kernel_num': 50, 'kernel_len': 25, 
                          'dense_num': 1, 'neuron_num': 100, 'batch_size': 256,
                          'pool_size': 2, 'pool_stride': 2, "kernel_constraint": 3,
                          'dropout_input': 0.1, 'dropout_conv': 0.3, 'dropout_dense': 0.6,
                          'learning_rate': 0.0005, 'patience_lr': 5, 'patience_stopping': 15,
                          'epochs': 500, 'activation': "softmax", 'loss': "categorical_crossentropy",
                          'seed': None}
        for key in default_params:
            if not key in self.params:
                self.params[key] = default_params[key]
        if not "class_num" in self.params:
            raise RuntimeError("Number of classes not specified.")
        if not "input_shape" in self.params:
            raise RuntimeError("Input shape not specified.")


    def _prepare_model(self):
        np.random.seed(self.params["seed"])
        random.seed(self.params["seed"])
        self.model = Sequential()
        self.model.add(Dropout(input_shape = self.params["input_shape"],
                               rate = self.params["dropout_input"]))
        for x in range(self.params["conv_num"]):
            self.model.add(Conv1D(filters = self.params["kernel_num"],
                                  kernel_size = self.params["kernel_len"],
                                  padding = "valid",
                                  kernel_initializer = RandomUniform(),
                                  kernel_constraint = max_norm(self.params["kernel_constraint"]),
                                  activation = "relu"))
            self.model.add(MaxPooling1D(pool_size = self.params["pool_size"],
                                        strides = self.params["pool_stride"]))
            self.model.add(Dropout(rate = self.params["dropout_conv"]))
        self.model.add(Flatten())
        for x in range(self.params["dense_num"]):
            self.model.add(Dense(units = self.params["neuron_num"],
                                 kernel_initializer = RandomUniform(),
                                 kernel_constraint = max_norm(self.params["kernel_constraint"]),
                                 activation = "relu"))
            self.model.add(Dropout(rate = self.params["dropout_dense"]))
        self.model.add(Dense(units = self.params["class_num"],
                             kernel_initializer = RandomUniform(),
                             activation = self.params['activation']))
        self.model.compile(loss = self.params['loss'],
                           optimizer = Adam(lr = self.params["learning_rate"]))


    def _prepare_callbacks(self):
        reduce_lr = ReduceLROnPlateau('val_loss', 0.5, self.params["patience_lr"], verbose = 0)
        stopper = EarlyStopping('val_loss', patience = self.params["patience_stopping"])
        checkpoints = ModelCheckpoint(self.temp_file, "val_loss", save_best_only = True)
        self.callbacks = [reduce_lr, stopper, checkpoints]


    def _plot_motif(self, data, subseqs):
        if data.is_rna:
            rnas, structs = zip(*(data.alpha_coder.decode(seq) for seq in subseqs))
            logo_rna = Motif(data.alpha_coder.alph0, sequences = rnas)
            logo_struct = Motif(data.alpha_coder.alph1, sequences = structs)
            return (logo_rna, logo_struct)
        return Motif(data.one_hot_encoder.alphabet, sequences = subseqs)


    def _get_subseq(self, sequences, histogram):
        subseqs = []
        for i, seq in enumerate(sequences):
            subseqs.append(seq[histogram[i]:histogram[i] + self.params["kernel_len"]])
        return subseqs


    def _get_activations_idx_kernel(self, data, idx, group, kernel):
        get_out = K.function([self.model.layers[0].input, K.learning_phase()],
                             [self.model.layers[1].output[:,:,kernel]])
        data_gen = data._data_generator(group, self.params['batch_size'], False, False, idx)
        n = max(len(idx)//self.params['batch_size'] + (len(idx)%self.params['batch_size'] != 0), 1)
        activations = []
        for _ in range(n):
            activations.append(get_out([next(data_gen), 0])[0])
        if len(activations) == 1:
            return activations[0]
        else:
            return np.vstack(activations)


    def _optimize_input(self, model, layer_name, node_index, input_data, lr, steps):
        model_input = model.layers[0].input
        loss = K.max(model.get_layer(layer_name).output[...,node_index])
        grads = K.gradients(loss, model_input)[0]
        grads = K.l2_normalize(grads, axis = 1)
        iterate = K.function([model_input, K.learning_phase()], [loss, grads])
        for _ in range(steps):
            loss_value, grads_value = iterate([input_data, 0])
            input_data += grads_value * lr
        return input_data[0], loss_value > 2


    def _extract_pwm(self, input_data, annotation, alphabet):
        pwm = []
        for char in alphabet:
            if char in "().": 
                pattern = "\{}".format(char)
            else:
                pattern = char
            idx = [m.start() for m in re.finditer(pattern, annotation)]
            pwm.append(np.sum(input_data[:,idx], axis = 1))
        return np.transpose(np.array(pwm))


    def _get_optimized_input(self, model, data, layer_name, node_index, boundary, lr, steps):
        for attempt in range(5):
            input_data = np.random.uniform(-boundary, +boundary,
                                           (1, self.params["input_shape"][0], self.params["input_shape"][1]))
            input_data, success = self._optimize_input(model, layer_name, node_index, input_data, lr, steps)
            if success: break
        if not success:
            print("Warning: loss did not converge for node {} in layer '{}'".format(node_index, layer_name))
        input_data = np.apply_along_axis(utils.softmax, 1, input_data)
        if not data.is_rna:
            return [Motif(data.one_hot_encoder.alphabet, pwm = input_data).plot(scale=0.25)]
        else:
            annotation_seq, annotation_struct = data.alpha_coder.decode(data.alpha_coder.alphabet)
            pwm_struct = self._extract_pwm(input_data, annotation_struct, data.alpha_coder.alph1)
            pwm_seq = self._extract_pwm(input_data, annotation_seq, data.alpha_coder.alph0)
            motif_struct = Motif(data.alpha_coder.alph1, pwm = pwm_struct).plot(scale=0.25)
            motif_seq = Motif(data.alpha_coder.alph0, pwm = pwm_seq).plot(scale=0.25)
            return [motif_seq, motif_struct]


    def _change_activation(self):
        path = "{}/temp_model_file".format(gettempdir())
        self.model.save(path, overwrite = True)
        model = load_model(path)
        model.layers[-1].activation = keras.activations.linear
        model.save(path, overwrite = True)
        model = load_model(path)
        remove(path)
        return model