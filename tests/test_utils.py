import unittest
import numpy as np
from tempfile import gettempdir
from os.path import dirname, isfile
from os import remove
from shutil import which


from pysster.Data import Data
from pysster.Model import Model
from pysster.Motif import Motif
from pysster import utils


class Test_utils(unittest.TestCase):


    def setUp(self):
        self.folder = dirname(__file__)
        file_name = self.folder + "/data/rna.fasta"
        self.data = Data(file_name, ("ACGU", "()."))
        self.params = {"conv_num":1, "kernel_num":3, "kernel_len":5,
                       "neuron_num":2, "epochs":3}
        self.m1 = Model(self.params, self.data, seed = 2)
        self.m1.train(self.data, verbose = False)


    def test_utils_save_load_model(self):
        utils.save_model(self.m1, gettempdir()+"/model")
        self.assertTrue(isfile(gettempdir()+"/model"))
        self.assertTrue(isfile(gettempdir()+"/model.h5"))
        model = utils.load_model(gettempdir()+"/model")
        self.assertTrue(self.m1.params == model.params)
        self.assertTrue(self.m1.model.get_config() == model.model.get_config())
        for x in range(6):
            self.assertTrue(np.allclose(self.m1.model.get_weights()[x],
                                        model.model.get_weights()[x]))
        remove(gettempdir()+"/model")
        remove(gettempdir()+"/model.h5")


    def test_utils_save_load_data(self):
        utils.save_data(self.data, gettempdir()+"/data")
        self.assertTrue(isfile(gettempdir()+"/data"))
        data = utils.load_data(gettempdir()+"/data")
        self.assertTrue(isinstance(data, Data))
        remove(gettempdir()+"/data")

    
    def test_utils_annotate_structures(self):
        utils.annotate_structures(self.folder+"/data/rna_annot.fasta",
                                  gettempdir()+"/test.fasta")
        with open(self.folder+"/data/rna_annot_ref.fasta", 'rt') as handle:
            ref = handle.read()
        with open(gettempdir()+"/test.fasta", 'rt') as handle:
            comp = handle.read()
        self.assertTrue(ref == comp)
        remove(gettempdir()+"/test.fasta")
    

    def test_utils_predict_structures(self):
        # RNAfold and RNAlib bindings not available
        skip = False
        try:
            from RNA import fold
        except:
            if which("RNAfold") == None:
                try:
                    utils.predict_structures(self.folder+"/data/rna_pred.fasta",
                                             gettempdir()+"/test2.fasta", 2, False)
                    raise RuntimeError('predict_structures should have raised an error at this point, but did not')
                except:
                    skip = True # we got an error, as expected
        #annotate=False
        if skip == True:
            return
        utils.predict_structures(self.folder+"/data/rna_pred.fasta",
                                 gettempdir()+"/test2.fasta", 2, False)
        if not isfile(gettempdir()+"/test2.fasta"): return
        with open(self.folder+"/data/rna_pred_ref.fasta", 'rt') as handle:
            ref = handle.read()
        with open(gettempdir()+"/test2.fasta", 'rt') as handle:
            comp = handle.read()
        self.assertTrue(ref == comp)
        remove(gettempdir()+"/test2.fasta")
        #annotate=True
        utils.predict_structures(self.folder+"/data/rna_pred.fasta",
                                 gettempdir()+"/test2.fasta", 2, True)
        if not isfile(gettempdir()+"/test2.fasta"): return
        with open(self.folder+"/data/rna_pred_ref_annot.fasta", 'rt') as handle:
            ref = handle.read()
        with open(gettempdir()+"/test2.fasta", 'rt') as handle:
            comp = handle.read()
        self.assertTrue(ref == comp)
        remove(gettempdir()+"/test2.fasta")

    def test_utils_save_as_meme(self):
        logos = [Motif('ACGT', ['GATTACA']), Motif('ACGT', ['AAAA'])]
        utils.save_as_meme(logos, gettempdir()+"/test.meme")
        with open(self.folder+"/data/ref.meme", 'rt') as handle:
            ref = handle.read()
        with open(gettempdir()+"/test.meme", 'rt') as handle:
            comp = handle.read()
        self.assertTrue(ref == comp)
        remove(gettempdir()+"/test.meme")
