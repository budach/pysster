import unittest
import numpy as np
from tempfile import gettempdir
from os.path import dirname, isfile
from os import remove
from PIL import Image


from pysster.Data import Data
from pysster.Model import Model
from pysster.Motif import Motif
from pysster import utils


class Test_Model(unittest.TestCase):


    def setUp(self):
        folder = dirname(__file__)
        file_name = folder + "/data/rna.fasta"
        self.data = Data(file_name, ("ACGU", "()."))
        self.params = {"conv_num":1, "kernel_num":3, "kernel_len":5,
                       "neuron_num":2, "epochs":3}
        self.m1 = Model(self.params, self.data, seed = 2)
        self.m2 = Model(self.params, self.data, seed = 13)
        self.m3 = Model(self.params, self.data, seed = 2)
    

    def test_model_init(self):
        self.assertTrue(self.m1.params["conv_num"] == 1)
        self.assertTrue(self.m1.params["kernel_num"] == 3)
        self.assertTrue(self.m1.params["kernel_len"] == 5)
        self.assertTrue(self.m1.params["neuron_num"] == 2)
        self.assertTrue(self.m1.params["activation"] == "sigmoid")
        self.assertTrue(self.m1.model.layers[2].get_weights()[0].shape == (5,12,3))
        self.assertTrue(np.allclose(self.m1.model.layers[2].get_weights()[0], 
                                    self.m3.model.layers[2].get_weights()[0]))
        self.assertFalse(np.allclose(self.m2.model.layers[2].get_weights()[0], 
                                     self.m3.model.layers[2].get_weights()[0]))
        self.assertTrue(np.allclose(self.m1.model.layers[6].get_weights()[0], 
                                    self.m3.model.layers[6].get_weights()[0]))
        self.assertFalse(np.allclose(self.m2.model.layers[6].get_weights()[0], 
                                     self.m3.model.layers[6].get_weights()[0]))


    def test_model_train_predict(self):
        for obj in [self.m1, self.m2, self.m3]:
            obj.train(self.data, verbose = False)
            predictions = obj.predict(self.data, "test")
            self.assertTrue(predictions.shape == (3,3))
            self.assertTrue((predictions > 0.49).all())
            self.assertTrue((predictions < 0.51).all())
            predictions = obj.predict(self.data, "all")
            self.assertTrue(predictions.shape == (20,3))
            self.assertTrue((predictions > 0.49).all())
            self.assertTrue((predictions < 0.51).all())
        self.assertTrue(np.allclose(self.m1.model.layers[2].get_weights()[0], 
                                    self.m3.model.layers[2].get_weights()[0], atol=0.001))
        self.assertFalse(np.allclose(self.m2.model.layers[2].get_weights()[0], 
                                     self.m3.model.layers[2].get_weights()[0], atol=0.001))
        self.assertTrue(np.allclose(self.m1.model.layers[6].get_weights()[0], 
                                    self.m3.model.layers[6].get_weights()[0], atol=0.001))
        self.assertFalse(np.allclose(self.m2.model.layers[6].get_weights()[0], 
                                     self.m3.model.layers[6].get_weights()[0], atol=0.001))
    

    def test_model_get_max_activations(self):
        acts = self.m1.get_max_activations(self.data, 'test')
        self.assertTrue(acts['activations'].shape == (3,3))
        self.assertTrue(acts['labels'].shape == (3,3))
        self.assertTrue(acts['group'] == 'test')
    

    def test_model_visualize_kernel(self):
        acts = self.m1.get_max_activations(self.data, 'all')
        folder = gettempdir() + '/'
        # individual kernels
        for kernel in range(self.params['kernel_num']):
            motif, score = self.m1.visualize_kernel(acts, self.data, kernel, folder)
            self.assertTrue(isfile(folder+"motif_kernel_{}.png".format(kernel)))
            self.assertTrue(isfile(folder+"position_kernel_{}.png".format(kernel)))
            self.assertTrue(isfile(folder+"activations_kernel_{}.png".format(kernel)))
            remove(folder+"motif_kernel_{}.png".format(kernel))
            remove(folder+"position_kernel_{}.png".format(kernel))
            remove(folder+"activations_kernel_{}.png".format(kernel))
            self.assertTrue(isinstance(motif, tuple))
            self.assertTrue(isinstance(motif[0], Motif))
            self.assertTrue(np.isclose(score, 0) or score > 0)
        # all kernels
        motifs = self.m1.visualize_all_kernels(acts, self.data, folder)
        self.assertTrue(len(motifs) == 3)
        for x in range(3):
            self.assertTrue(isinstance(motifs[x], tuple))
            self.assertTrue(isinstance(motifs[x][0], Motif))
        for kernel in range(self.params['kernel_num']):
            self.assertTrue(isfile(folder+"motif_kernel_{}.png".format(kernel)))
            self.assertTrue(isfile(folder+"position_kernel_{}.png".format(kernel)))
            self.assertTrue(isfile(folder+"activations_kernel_{}.png".format(kernel)))
            remove(folder+"motif_kernel_{}.png".format(kernel))
            remove(folder+"position_kernel_{}.png".format(kernel))
            remove(folder+"activations_kernel_{}.png".format(kernel))
        self.assertTrue(isfile(folder+"summary.html"))
        remove(folder+"summary.html")


    def test_model_plot_clustering(self):
        acts = self.m1.get_max_activations(self.data, 'test')
        self.m1.plot_clustering(acts, gettempdir()+"/clust.png")
        self.assertFalse(isfile(gettempdir()+"/clust.png"))
    

    def test_model_optimized_inputs(self):
        self.m1.visualize_optimized_inputs(self.data, self.m1.model.layers[2].name, gettempdir()+"/test.png")
        self.m1.visualize_optimized_inputs(self.data, self.m1.model.layers[2].name, gettempdir()+"/test2.png", nodes = [0])
        with Image.open(gettempdir()+"/test.png") as img:        
            self.assertTrue(img.size == (1998,1128))
        with Image.open(gettempdir()+"/test2.png") as img:        
            self.assertTrue(img.size == (1998,376))
        remove(gettempdir()+"/test.png")
        remove(gettempdir()+"/test2.png")