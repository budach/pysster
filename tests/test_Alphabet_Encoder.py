import unittest


from pysster.Alphabet_Encoder import Alphabet_Encoder


class Test_Alphabet_Encoder(unittest.TestCase):


    def setUp(self):
        self.dot = Alphabet_Encoder('ACGU', '().')
        self.ref_dot = ('ACGUUGCA', '(((..)))')
        self.decoded_dot = self.dot.encode(self.ref_dot)

        self.ann = Alphabet_Encoder('ACGU', 'HIMS')
        self.ref_ann = ('ACGUUGCA', 'SSHHSSIM')
        self.decoded_ann = self.ann.encode(self.ref_ann)


    def test_alphabet_encoder_init(self):
        self.assertTrue(self.dot.alph0 == 'ACGU')
        self.assertTrue(self.dot.alph1 == '().')

        self.assertTrue(len(self.dot.alphabet) == 12)
        self.assertTrue(len(self.ann.alphabet) == 16)
    

    def test_alphabet_encoder_encode(self):
        self.assertTrue(len(self.decoded_dot) == 8)
        self.assertTrue(len(self.decoded_ann) == 8)


    def test_alphabet_encoder_decode(self):
        self.assertTrue(self.dot.decode(self.decoded_dot) == self.ref_dot)
        self.assertTrue(self.ann.decode(self.decoded_ann) == self.ref_ann)