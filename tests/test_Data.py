import unittest
import numpy as np
from os.path import dirname


from pysster.Data import Data
from pysster.Model import Model


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class Test_Data(unittest.TestCase):


    def setUp(self):
        folder = dirname(__file__)
        dna_files = [folder + "/data/dna_pos.fasta", folder + "/data/dna_neg.fasta"]
        rna_files = folder + "/data/rna.fasta"
        rna_pwm = [folder + '/data/rna_pwm1.fasta', folder + '/data/rna_pwm2.fasta']
        rna_pwm_add = [folder + '/data/rna_pwm1_add.txt', folder + '/data/rna_pwm2_add.txt']
        self.data_dna = Data(dna_files, "ACGT")
        self.data_rna_dot = Data(rna_files, ("ACGU", "()."))
        self.data_pwm = Data(rna_pwm, ('ACGU', '().'), structure_pwm=True)
        self.data_pwm.load_additional_data(rna_pwm_add, is_categorical=False)
        self.data_pwm.load_additional_data(rna_pwm_add, is_categorical=True)

    
    def test_data_init_dna(self):
        self.assertFalse(self.data_dna.is_rna_pwm)
        self.assertFalse(self.data_dna.is_rna)
        self.assertFalse(self.data_dna.multilabel)
        self.assertTrue(len(self.data_dna.data) == 100)
        self.assertTrue(self.data_dna.data[0].shape == (32, 4))
        self.assertTrue(self.data_dna.one_hot_encoder.alphabet == 'ACGT')

        self.assertTrue(len(self.data_dna.labels) == 100)
        self.assertTrue(self.data_dna.labels[0].shape == (2,))
        for x in range(40):
            self.assertTrue((self.data_dna.labels[x] == [1,0]).all())
        for x in range(40, 100):
            self.assertTrue((self.data_dna.labels[x] == [0,1]).all())


    def test_data_init_rna(self):
        self.assertFalse(self.data_rna_dot.is_rna_pwm)
        self.assertTrue(len(self.data_rna_dot.data) == 20)
        self.assertTrue(self.data_rna_dot.data[0].shape == (40, 12))
        self.assertTrue(self.data_rna_dot.alpha_coder.alph1 == '().')

        idx_0 = [0,2,4,10,11,14,18,19]
        idx_1 = [1,2,5,6,9,10,12,15,16,17,19]
        idx_2 = [0,2,3,6,7,8,9,10,13,14,15,16]
        for obj in [self.data_rna_dot]:
            self.assertTrue(obj.is_rna)
            self.assertTrue(obj.multilabel)
            self.assertTrue(obj.alpha_coder.alph0 == 'ACGU')
            self.assertTrue(obj.one_hot_encoder.alphabet == obj.alpha_coder.alphabet)

            self.assertTrue(len(obj.labels) == 20)
            self.assertTrue(obj.labels[0].shape == (3,))
            for x in idx_0: self.assertTrue(obj.labels[x][0] == 1)
            for x in idx_1: self.assertTrue(obj.labels[x][1] == 1)
            for x in idx_2: self.assertTrue(obj.labels[x][2] == 1)


    def test_data_train_val_test_split(self):
        for obj in [self.data_rna_dot]:
            self.assertTrue(len(obj.splits["train"]) == 14)
            self.assertTrue(len(obj.splits["val"]) == 3)
            self.assertTrue(len(obj.splits["test"]) == 3)
            self.assertTrue(set(obj.splits["train"]) &
                            set(obj.splits["val"])   &
                            set(obj.splits["test"])  == set())

        self.assertTrue(len(self.data_dna.splits["train"]) == 70)
        self.assertTrue(len(self.data_dna.splits["val"]) == 15)
        self.assertTrue(len(self.data_dna.splits["test"]) == 15)
        self.assertTrue(set(self.data_dna.splits["train"]) &
                        set(self.data_dna.splits["val"])   &
                        set(self.data_dna.splits["test"])  == set())


    def test_data_shape(self):
        self.assertTrue(self.data_dna._shape() == (32,4))
        self.assertTrue(self.data_rna_dot._shape() == (40,12))


    def test_data_get_sequences(self):
        num_seqs = {'train': 70, 'val': 15, 'test': 15, 'all': 100}
        for group in ['train', 'val', 'test', 'all']:
            seqs = []
            for class_id in [0, 1]:
                seqs += self.data_dna._get_sequences(class_id, group)
            self.assertTrue(len(seqs) == num_seqs[group])
            for seq in seqs:
                self.assertTrue(seq == "ACGTACGTACGTACGTACGTACGTACGTACGT")


    def test_data_get_data(self):
        num_seqs = {'train': 70, 'val': 15, 'test': 15, 'all': 100}
        for group in ['train', 'val', 'test', 'all']:
            one_hot = self.data_dna._get_data(group)
            self.assertTrue(one_hot[0].shape == (num_seqs[group], 32, 4))
            self.assertTrue(one_hot[1].shape == (num_seqs[group], 2))

        num_seqs = {'train': 14, 'val': 3, 'test': 3, 'all': 20}
        for group in ['train', 'val', 'test', 'all']:
            one_hot = self.data_rna_dot._get_data(group)
            self.assertTrue(one_hot[0].shape == (num_seqs[group], 40, 12))
            self.assertTrue(one_hot[1].shape == (num_seqs[group], 3))


    def test_data_get_class_weights(self):
        weights = self.data_dna._get_class_weights()
        for key, val in {0: 1.5, 1: 1.0}.items():
            self.assertTrue(isclose(weights[key], val))
        weights = self.data_rna_dot._get_class_weights()
        for key, val in {0: 1.5, 1: 1.0909090909090908, 2: 1.0}.items():
            self.assertTrue(isclose(weights[key], val))
    

    def test_data_get_summary(self):
        classes = [["class_0", "class_1"],
                   ["class_0", "class_1", "class_2"],
                   ["class_0", "class_1", "class_2"]]
        for i, obj in enumerate([self.data_dna, self.data_rna_dot]):
            text = obj.get_summary()
            text = text.split("\n")
            self.assertTrue(text[0].split() == classes[i])
            self.assertTrue(text[1].split()[:2] == ["all", "data:"])
            self.assertTrue(text[2].split()[0] == "training:")
            self.assertTrue(text[3].split()[0] == "validation:")
            self.assertTrue(text[4].split()[0] == "test:")
            for x in range(len(classes[i])):
                self.assertTrue(int(text[2].split()[x+1]) + 
                                int(text[3].split()[x+1]) +
                                int(text[4].split()[x+1]) == int(text[1].split()[x+2]))
    

    def test_data_get_labels(self):
        labels = self.data_dna.get_labels("test")
        self.assertTrue(labels.shape == (15, 2))
        self.assertTrue((labels.sum(axis=1) == 1).all())
        labels = self.data_dna.get_labels("train")
        self.assertTrue(labels.shape == (70, 2))
        self.assertTrue((labels.sum(axis=1) == 1).all())
        labels = self.data_dna.get_labels("val")
        self.assertTrue(labels.shape == (15, 2))
        self.assertTrue((labels.sum(axis=1) == 1).all())
        labels = self.data_dna.get_labels("all")
        self.assertTrue(labels.shape == (100, 2))
        self.assertTrue((labels.sum(axis=1) == 1).all())
        self.assertTrue((labels.sum(axis=0) == [40, 60]).all())

        labels = self.data_rna_dot.get_labels("test")
        self.assertTrue(labels.shape == (3, 3))
        self.assertTrue((labels.sum(axis=1) <= 3).all())
        labels = self.data_rna_dot.get_labels("all")
        self.assertTrue(labels.shape == (20, 3))
        self.assertTrue((labels.sum(axis=1) <= 3).all())
        self.assertTrue((labels.sum(axis=0) == [8,11,12]).all())
     

    def test_data_pwm_struct(self):
        self.assertTrue(self.data_pwm.is_rna_pwm == True)
        self.assertTrue(len(self.data_pwm.data) == 32)
        self.assertTrue(self.data_pwm.data[0].shape == (10, 12))
        ref = np.array([0.9,0,0.1,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0.8,0,0.2,
                        0.7,0,0.3,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0.9,0,0.1,
                        0.0,0,1.0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0.0,0,1.0,
                        0.0,0.2,0.8,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0.0,0.7,0.3,
                        0.0,0.8,0.2,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0.0,0.9,0.1])
        ref.shape = (10,12)
        self.assertTrue(np.allclose(self.data_pwm.data[0], ref))
        seqs = self.data_pwm._get_sequences(0, 'all')
        self.assertTrue(len(seqs) == 16)
        self.assertTrue(np.allclose(ref, seqs[15]))


    def test_data_additional(self):
        self.assertTrue(len(self.data_pwm.meta) == 2)
        self.assertTrue(self.data_pwm.meta[0]['is_categorical'] == False)
        self.assertTrue(self.data_pwm.meta[1]['is_categorical'] == True)
        self.assertTrue(self.data_pwm.meta[0]['data'] == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1])
        self.assertTrue(len(self.data_pwm.meta[1]['data']) == 32)
        for x in self.data_pwm.meta[1]['data']:
            self.assertTrue(sum(x) == 1)
        self.assertTrue((self.data_pwm.meta[1]['data'][0] == self.data_pwm.meta[1]['data'][31]).all())
        self.assertTrue((self.data_pwm.meta[1]['data'][13] == self.data_pwm.meta[1]['data'][18]).all())
        addi = self.data_pwm._get_additional_data([0,1,15,16], 0, 4)
        self.assertTrue(len(addi) == 4)
        self.assertTrue(np.allclose(addi[0], [1,*self.data_pwm.meta[1]['data'][0]]))
        self.assertTrue(np.allclose(addi[1], [2,*self.data_pwm.meta[1]['data'][1]]))
        self.assertTrue(np.allclose(addi[2], [16,*self.data_pwm.meta[1]['data'][15]]))
        self.assertTrue(np.allclose(addi[3], [16,*self.data_pwm.meta[1]['data'][16]]))
        mod = Model({"conv_num":1, "kernel_num":2, "kernel_len":4, "neuron_num":2, "epochs":2}, self.data_pwm)
        mod.train(self.data_pwm, verbose=True)
        predictions = mod.predict(self.data_pwm, "all")
        self.assertTrue(predictions.shape == (32,2))