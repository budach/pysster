import unittest
import numpy as np
from PIL import Image


from pysster.Motif import Motif


class Test_Motif(unittest.TestCase):


    def setUp(self):
        self.ref_pwm = np.array([[0,0,0.25,0],
                                 [0.25,0,0,0],
                                 [0,0,0,0.25],
                                 [0,0,0,0.25],
                                 [0.25,0,0,0],
                                 [0,0.25,0,0],
                                 [0.25,0,0,0]], dtype = np.float32)
        self.m = Motif("ACGT", ["GATTACA"])
        self.m2 = Motif("ACGT", pwm = self.ref_pwm)


    def test_motif_init(self):
        self.assertTrue(self.m.alphabet == "ACGT")
        self.assertTrue(self.m2.alphabet == "ACGT")


    def test_motif_valid_pwm(self):
        self.assertTrue(self.m.pwm.shape == (7, 4))
        self.assertTrue((self.m.pwm >= 0).all() and (self.m.pwm <= 1).all())
        self.assertTrue(np.allclose(np.sum(self.m.pwm, axis = 1), [1] * 7))

        self.assertTrue(self.m2.pwm.shape == (7, 4))
        self.assertTrue((self.m2.pwm >= 0).all() and (self.m2.pwm <= 1).all())
        self.assertTrue(np.allclose(np.sum(self.m2.pwm, axis = 1), [1] * 7))

        self.assertTrue(np.allclose(self.m.pwm, self.m2.pwm))


    def test_motif_valid_entropies(self):
        self.assertTrue(self.m.entropies.shape == (7,))
        self.assertTrue((self.m.entropies >= 0).all() and (self.m.entropies <= 2).all())

        self.assertTrue(self.m2.entropies.shape == (7,))
        self.assertTrue((self.m2.entropies >= 0).all() and (self.m2.entropies <= 2).all())

        self.assertTrue(np.allclose(self.m.entropies, self.m2.entropies))


    def test_motif_plot(self):
        self.assertTrue(isinstance(self.m.plot(), Image.Image))
        self.assertTrue(isinstance(self.m2.plot(), Image.Image))