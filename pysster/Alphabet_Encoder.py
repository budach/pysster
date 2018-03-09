import numpy as np
from itertools import product


class Alphabet_Encoder:


    def __init__(self, alph0, alph1):
        self._encodeTable = {tup: chr(i+65) for i, tup in enumerate(product(alph0, alph1))}
        self._decodeTable = {val: key for key, val in self._encodeTable.items()}
        self.alphabet = "".join(sorted(self._decodeTable.keys()))
        self.alph0 = alph0
        self.alph1 = alph1


    def encode(self, record):
        encoded = []
        for tup in zip(record[0], record[1]):
            encoded.append(self._encodeTable[tup])
        return "".join(encoded)


    def decode(self, encoded):
        alph0Encoded, alph1Encoded = [], []
        for x in encoded:
            alph0Encoded.append(self._decodeTable[x][0])
            alph1Encoded.append(self._decodeTable[x][1])
        return "".join(alph0Encoded), "".join(alph1Encoded)