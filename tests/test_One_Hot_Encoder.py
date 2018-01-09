import unittest
import numpy as np


from pysster.One_Hot_Encoder import One_Hot_Encoder


class Test_One_Hot_Encoder(unittest.TestCase):


    def setUp(self):
        self.one = One_Hot_Encoder("ACGT")
        self.reference_seq = "GATTACA"
        self.reference_one_hot = np.array([[0,0,1,0],
                                           [1,0,0,0],
                                           [0,0,0,1],
                                           [0,0,0,1],
                                           [1,0,0,0],
                                           [0,1,0,0],
                                           [1,0,0,0]], dtype = np.uint8)


    def test_one_hot_encoder_init(self):
        self.assertTrue(self.one.alphabet == "ACGT")
        self.assertTrue(len(self.one.table) == 4)
        self.assertTrue(len(self.one.table_rev) == 4)


    def test_one_hot_encoder_encode(self):
        encoded = self.one.encode(self.reference_seq)
        self.assertTrue(np.array_equal(encoded, self.reference_one_hot))


    def test_one_hot_encoder_decode(self):
        decoded = self.one.decode(self.reference_one_hot)
        self.assertTrue(np.array_equal(decoded, self.reference_seq))