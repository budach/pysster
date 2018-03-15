from os.path import dirname
import unittest
import warnings


from pysster.Data import Data
from pysster.Grid_Search import Grid_Search
from pysster.Model import Model


class Test_Grid_Search(unittest.TestCase):


    def setUp(self):
        folder = dirname(__file__)
        files = [folder + "/data/dna_pos.fasta", folder + "/data/dna_neg.fasta"]
        self.data = Data(files, "ACGT")
        self.params = {'conv_num': [1], 'kernel_num': [2, 4], 'epochs': [1,2]}
        self.searcher = Grid_Search(self.params)
    

    def test_grid_search_init(self):
        self.assertTrue(self.searcher.params == self.params)
        self.assertTrue(len(self.searcher.candidates) == 4)
        self.assertTrue({'conv_num':1,'kernel_num':2,'epochs':1} in self.searcher.candidates)
        self.assertTrue({'conv_num':1,'kernel_num':2,'epochs':2} in self.searcher.candidates)
        self.assertTrue({'conv_num':1,'kernel_num':4,'epochs':1} in self.searcher.candidates)
        self.assertTrue({'conv_num':1,'kernel_num':4,'epochs':2} in self.searcher.candidates)
    

    def test_grid_search_train_roc(self):
        # filter tensorflow deprecation warnings to not clutter the unittest output 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, table = self.searcher.train(self.data, verbose = False)
        self.assertTrue(isinstance(model, Model))
        self.assertTrue(isinstance(table, str))
        table = table.split('\n')
        self.assertTrue(len(table) == 9)
        self.assertTrue(table[0] in ["# conv_num: [1]", "# kernel_num: [2, 4]", "# epochs: [1, 2]"])
        self.assertTrue(table[1] in ["# conv_num: [1]", "# kernel_num: [2, 4]", "# epochs: [1, 2]"])
        self.assertTrue(table[2] in ["# conv_num: [1]", "# kernel_num: [2, 4]", "# epochs: [1, 2]"])
        for word in table[3].split():
            self.assertTrue(word in ["conv_num", "kernel_num", "epochs", "roc-auc"])
        for line in table[4:8]:
            self.assertTrue(len(line.split()) == 4)
        self.assertTrue(table[8] == '')


    def test_grid_search_train_pre(self):
        # filter tensorflow deprecation warnings to not clutter the unittest output 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, table = self.searcher.train(self.data, pr_auc = True, verbose = False)
        self.assertTrue(isinstance(model, Model))
        self.assertTrue(isinstance(table, str))
        table = table.split('\n')
        self.assertTrue(len(table) == 9)
        self.assertTrue(table[0] in ["# conv_num: [1]", "# kernel_num: [2, 4]", "# epochs: [1, 2]"])
        self.assertTrue(table[1] in ["# conv_num: [1]", "# kernel_num: [2, 4]", "# epochs: [1, 2]"])
        self.assertTrue(table[2] in ["# conv_num: [1]", "# kernel_num: [2, 4]", "# epochs: [1, 2]"])
        for word in table[3].split():
            self.assertTrue(word in ["conv_num", "kernel_num", "epochs", "pre-auc"])
        for line in table[4:8]:
            self.assertTrue(len(line.split()) == 4)
        self.assertTrue(table[8] == '')
