import numpy as np
from os import remove
from tempfile import gettempdir
from itertools import product
from keras import backend as K
import string
import random


from pysster.Data import Data
from pysster.Model import Model
import pysster.utils as utils


class Grid_Search:
    """
    The Grid_Search class provides a simple way to execute a hyperparameter tuning for
    the convolutional neural network model. Have a look at the Model documentation for an overview
    of all available hyperparameters. The tuning returns the best model (highest ROC-AUC
    on the validation data) and an overview of all trained models.
    """

    def __init__(self, params):
        """ Initialize the object with a collection of parameter values.

        For example: providing {'conv_num': [1,2,3], 'kernel_num': [20,50]} will result in
        training 6 different models (all possible combinations of the provided values) when
        the train() method is called later on. Parameters that are not provided here will hold
        their default values in all 6 models.

        Parameters
        ----------
        params: dict
            A dict containing parameter names as keys and corresponding values as lists.
        """
        for x in params:
            if not isinstance(params[x], list) or params[x] == []:
                raise RuntimeError("All params entries must be non-empty lists.")
        self.params = params
        self.candidates = [dict(zip(params.keys(), x)) for x in product(*params.values())]


    def train(self, data, verbose = True):
        """ Train all models and return the best one.

        Models are evaluated and ranked according to their ROC-AUC on a validation data set.

        Parameters
        ----------
        data: pysster.Data
            A Data object providing training and validation data sets.
        
        verbose: bool
            If True, progress information (train/val loss) will be printed throughout the training.

        Returns
        -------
        results: tuple(pysster.Model, str)
            The best performing model and an overview table of all models are returned.
        """
        best_model_path = "{}/{}".format(
            gettempdir(),
            ''.join(random.choice(string.ascii_uppercase) for _ in range(20))
        )
        aucs = []
        max_auroc = -1
        for i, candidate in enumerate(self.candidates):
            model = Model(candidate, data)
            model.train(data, verbose)
            predictions = model.predict(data, "val")
            labels = data.get_labels("val")
            report = utils.performance_report(labels, predictions)
            roc_auc = np.sum(report[:,0:-1] * report[:,-1, np.newaxis], axis=0)
            roc_auc = (roc_auc / np.sum(report[:,-1]))[3]
            aucs.append(roc_auc)
            if aucs[-1] > max_auroc:
                max_auroc = aucs[-1]
                utils.save_model(model, best_model_path)
            K.clear_session()
            K.reset_uids()
            if not verbose: continue
            print("\n=== Summary ===")
            print("Model {}/{} = {:.5f} weighted avg roc-auc".format(i+1, len(self.candidates), aucs[i]))
            for param in candidate:
                if not param in ["input_shape"]:
                    print(" - {}: {}".format(param, candidate[param]))
        # load the best model (and remove it from disc)
        model = utils.load_model(best_model_path)
        remove(best_model_path)
        remove("{}.h5".format(best_model_path))
        # save a formatted summary of all trained models
        table = self._grid_search_table(aucs)
        return model, table


    def _grid_search_table(self, aucs):
        order = sorted(((x, i) for i, x in enumerate(aucs)), reverse = True)
        format_str = ""
        table = ""
        for key, value in self.params.items():
            format_str += "{{:>{}}} ".format(len(key))
            table += "# {}: {}\n".format(key, value)
        format_str += "{:.5f}\n"
        table += " ".join(self.params.keys()) + " roc-auc\n"
        for tup in order:
            table += format_str.format(*(self.candidates[tup[1]][key] for key in self.params), tup[0])
        return table