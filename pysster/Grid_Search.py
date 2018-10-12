import numpy as np
from os import remove
from tempfile import gettempdir
from itertools import product
from keras import backend as K
from copy import deepcopy
import string
import random

from pysster.Model import Model
import pysster.utils as utils


class Grid_Search:
    """
    The Grid_Search class provides a simple way to execute a hyperparameter tuning for
    the convolutional neural network model. Have a look at the Model documentation for an overview
    of all available hyperparameters. The tuning returns the best model (highest ROC-AUC or PR-AUC
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
        self.params = deepcopy(params)
        self.candidates = [dict(zip(params.keys(), x)) for x in product(*params.values())]


    def train(self, data, pr_auc = False, verbose = True):
        """ Train all models and return the best one.

        Models are evaluated and ranked according to their ROC-AUC or PR-AUC (precision-recall)
        on a validation data set.

        Parameters
        ----------
        data: pysster.Data
            A Data object providing training and validation data sets.
        
        pr_auc: bool
            If True, the area under the precision-recall curve will be maximized instead of the area under the ROC curve

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
        if True == pr_auc:
            metric_idx = 4
            metric_name = "pre-auc"
        else:
            metric_idx = 3
            metric_name = "roc-auc"
        metric = []
        max_metric = -1
        for i, candidate in enumerate(self.candidates):
            model = Model(candidate, data)
            model.train(data, verbose)
            predictions = model.predict(data, "val")
            labels = data.get_labels("val")
            report = utils.performance_report(labels, predictions)
            metric_val = np.sum(report[:,0:-1] * report[:,-1, np.newaxis], axis=0)
            metric_val = (metric_val / np.sum(report[:,-1]))[metric_idx]
            metric.append(metric_val)
            if metric[-1] > max_metric:
                max_metric = metric[-1]
                utils.save_model(model, best_model_path)
            K.clear_session()
            K.reset_uids()
            if not verbose: continue
            print("\n=== Summary ===")
            print("Model {}/{} = {:.5f} weighted avg {}".format(i+1, len(self.candidates), metric[i], metric_name))
            for param in candidate:
                if not param in ["input_shape"]:
                    print(" - {}: {}".format(param, candidate[param]))
        # load the best model (and remove it from disc)
        model = utils.load_model(best_model_path)
        remove(best_model_path)
        remove("{}.h5".format(best_model_path))
        # save a formatted summary of all trained models
        table = self._grid_search_table(metric, metric_name)
        return model, table


    def _grid_search_table(self, metric, metric_name):
        order = sorted(((x, i) for i, x in enumerate(metric)), reverse = True)
        format_str = ""
        table = ""
        for key, value in self.params.items():
            format_str += "{{:>{}}} ".format(len(key))
            table += "# {}: {}\n".format(key, value)
        format_str += "{:.5f}\n"
        table += " ".join(self.params.keys()) + " {}\n".format(metric_name)
        for tup in order:
            table += format_str.format(*(str(self.candidates[tup[1]][key]) for key in self.params), tup[0])
        return table